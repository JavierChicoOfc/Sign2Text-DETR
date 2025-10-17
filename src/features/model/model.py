import math
from typing import Dict

import torch
from torch import nn
from torchinfo import summary
from torchvision.models import ResNet50_Weights, resnet50

from features.logger.logger import get_logger
from features.rich_handler.rich_handlers import ModelHandler


def _get_1d_sincos_pos_embed(
    length: int, dim: int, temperature: float = 10000.0, device=None
) -> torch.Tensor:
    """
    Generate 1D sine-cosine positional embedding.
    Args:
        length (int): Sequence length.
        dim (int): Embedding dimension (must be even).
        temperature (float): Temperature scaling for frequencies.
        device: Torch device.
    Returns:
        torch.Tensor: Positional embedding of shape (length, dim).
    """
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(temperature) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_2d_sincos_position_embedding(
    height: int, width: int, dim: int, device=None
) -> torch.Tensor:
    """
    Generate 2D sine-cosine positional encoding.
    Args:
        height (int): Height of grid.
        width (int): Width of grid.
        dim (int): Embedding dimension (must be even).
        device: Torch device.
    Returns:
        torch.Tensor: Positional encoding of shape (1, height*width, dim).
    """
    assert dim % 2 == 0
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    pos = pos.view(1, height * width, dim)
    return pos


class DETR(nn.Module):
    """
    DETR (Detection Transformer) model for object detection.
    Backbone: ResNet-50. Transformer encoder-decoder. Predicts class logits and bounding boxes.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        num_queries: int = 25,
    ) -> None:
        super().__init__()
        self.logger = get_logger("model")
        self.model_handler = ModelHandler()
        model_config = {
            "Model Type": "DETR (Detection Transformer)",
            "Number of Classes": num_classes,
            "Hidden Dimension": hidden_dim,
            "Attention Heads": nheads,
            "Encoder Layers": num_encoder_layers,
            "Decoder Layers": num_decoder_layers,
            "Object Queries": num_queries,
            "Backbone": "ResNet-50 (ImageNet pretrained)",
        }
        self.model_handler.log_model_architecture(model_config)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.num_queries = num_queries
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        feat = self.conv(x)
        bsz, d_model, Hf, Wf = feat.shape
        src = feat.flatten(2).permute(0, 2, 1)
        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)
        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)
        hs = self.transformer(src=src, tgt=tgt)
        return {"pred_logits": self.linear_class(hs), "pred_boxes": self.linear_bbox(hs).sigmoid()}

    def log_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.model_handler.log_parameters_count(total_params, trainable_params)

    def load_pretrained(self, checkpoint_path: str) -> None:
        try:
            self.load_state_dict(torch.load(checkpoint_path))
            self.model_handler.log_model_loading(checkpoint_path, success=True)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.model_handler.log_model_loading(checkpoint_path, success=False)


if __name__ == "__main__":
    model = DETR(num_classes=11)
    summary(model, (5, 3, 224, 224))
