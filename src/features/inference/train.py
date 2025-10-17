import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim, save
from torch.utils.data import DataLoader

from features.DETR_data.data import DETRData
from loss.loss import DETRLoss, HungarianMatcher
from features.model.model import DETR
from features.boxes.boxes import stacker
from features.logger.logger import get_logger
from features.rich_handler.rich_handlers import rich_training_context

if __name__ == "__main__":
    logger = get_logger("training")
    logger.print_banner()

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    train_dataset = DETRData("data/train")
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=stacker, drop_last=True)

    test_dataset = DETRData("data/test", train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=stacker, drop_last=True)

    num_classes = 11
    model = DETR(num_classes=num_classes)

    ckpt_path = ""
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained from {ckpt_path} with strict=False")
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
    except Exception as e:
        logger.warning(f"Could not load pretrained weights from {ckpt_path}: {e}")

    if hasattr(model, "class_embed") and isinstance(getattr(model, "class_embed"), nn.Linear):
        old_head = getattr(model, "class_embed")
        hidden_dim = old_head.in_features
        new_out = num_classes + 1
        new_head = nn.Linear(hidden_dim, new_out)
        nn.init.normal_(new_head.weight, std=0.01)
        nn.init.constant_(new_head.bias, 0.0)

        setattr(model, "class_embed", new_head)
        logger.info(f"Reinitialized class head to output {new_out} (num_classes + no-object)")

    model.log_model_info()
    model.train()

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and "backbone" not in n and "class_embed" not in n
            ],
            "lr": 1e-5,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n
            ],
            "lr": 1e-6,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and "class_embed" in n
            ],
            "lr": 5e-5,
        },
    ]
    opt = optim.AdamW(param_groups, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10 * len(train_dataloader), T_mult=2
    )

    weights = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(
        num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1
    )

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    epochs = 1000

    training_config = {
        "Total Epochs": epochs,
        "Batch Size": 4,
        "Train Batches": train_batches,
        "Test Batches": test_batches,
        "Optimizer": "AdamW",
        "Weight Decay": 1e-4,
        "LR (others)": 1e-5,
        "LR (backbone)": 1e-6,
        "LR (class_head)": 5e-5,
        "Scheduler": "CosineAnnealingWarmRestarts (per-iteration)",
        "Grad Clip (max_norm)": 0.1,
    }
    logger.print_table(
        "🏋️ Training Configuration", list(training_config.keys()), [list(training_config.values())]
    )

    with rich_training_context() as training_handler:
        for epoch in range(epochs):
            with training_handler.create_training_progress() as epoch_progress:
                epoch_task = epoch_progress.add_task(
                    f"[bold blue] Progress {epoch + 1}/{epochs}",
                    train_loss=0.0,
                    test_loss=0.0,
                    total=train_batches,
                )

                model.train()
                train_epoch_loss = 0.0

                for batch_idx, batch in enumerate(train_dataloader):
                    X, y = batch
                    try:
                        yhat = model(X)
                        yhat_classes = yhat["pred_logits"]
                        yhat_bb = yhat["pred_boxes"]

                        loss_dict = criterion(yhat, y)
                        weight_dict = criterion.weight_dict
                        losses = (
                            loss_dict["labels"]["loss_ce"] * weight_dict["class_weighting"]
                            + loss_dict["boxes"]["loss_bbox"] * weight_dict["bbox_weighting"]
                            + loss_dict["boxes"]["loss_giou"] * weight_dict["giou_weighting"]
                        )

                        train_epoch_loss += losses.item()

                        opt.zero_grad()
                        losses.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                        opt.step()

                        scheduler.step(epoch + batch_idx / len(train_dataloader))

                        epoch_progress.update(
                            epoch_task,
                            advance=1,
                            train_loss=round(train_epoch_loss / train_batches, 5),
                        )

                    except Exception as e:
                        logger.error(
                            f"Training error at epoch {epoch}, batch {batch_idx}: {str(e)}"
                        )
                        logger.error(f"Batch targets: {str(y)}")
                        sys.exit()

                model.eval()
                test_epoch_loss = 0.0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_dataloader):
                        X, y = batch
                        yhat = model(X)
                        loss_dict = criterion(yhat, y)
                        weight_dict = criterion.weight_dict
                        losses = (
                            loss_dict["labels"]["loss_ce"] * weight_dict["class_weighting"]
                            + loss_dict["boxes"]["loss_bbox"] * weight_dict["bbox_weighting"]
                            + loss_dict["boxes"]["loss_giou"] * weight_dict["giou_weighting"]
                        )
                        test_epoch_loss += losses.item()
                        epoch_progress.update(
                            epoch_task,
                            advance=0,
                            test_loss=round(test_epoch_loss / test_batches, 5),
                        )

                if epoch % 10 == 0 and epoch != 0:
                    checkpoint_path = f"checkpoints/{epoch}_model.pt"
                    save(model.state_dict(), checkpoint_path)
                    training_handler.save_checkpoint_status(checkpoint_path, epoch)

    save(model.state_dict(), f"checkpoints/{epoch}_model.pt")
