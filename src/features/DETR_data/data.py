import os
from typing import Any, Dict, Tuple

import albumentations as A
import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from features.boxes.boxes import rescale_bboxes, stacker
from features.logger.logger import get_logger
from features.rich_handler.rich_handlers import DataLoaderHandler
from features.classes.setup import get_classes


class DETRData(Dataset):
    """
    PyTorch Dataset for DETR-style object detection.
    Loads images and YOLO-format labels, applies augmentations, and returns tensors for training/testing.
    """

    def __init__(self, path: str, train: bool = True) -> None:
        """
        Args:
            path (str): Path to dataset folder containing 'images' and 'labels' subfolders.
            train (bool): Whether to use training augmentations.
        """
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, "labels")
        self.images_path = os.path.join(self.path, "images")
        self.label_files = os.listdir(self.labels_path)
        self.labels = [x for x in self.label_files if x.endswith(".txt")]
        self.train = train

        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()

        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path,
        }
        self.data_handler.log_dataset_stats(dataset_info)

        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor",
        ]
        self.data_handler.log_transform_info(transform_list)

    def safe_transform(
        self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, max_attempts: int = 50
    ) -> Dict[str, Any]:
        """
        Apply augmentations safely, retrying if bboxes are lost.

        Args:
            image (np.ndarray): Input image.
            bboxes (np.ndarray): Bounding boxes in YOLO format.
            labels (np.ndarray): Class labels.
            max_attempts (int): Max attempts to keep bboxes after augmentation.

        Returns:
            Dict[str, Any]: Augmented image, bboxes, and labels.
        """
        self.transform = A.Compose(
            [
                A.Resize(500, 500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []),
                A.Resize(224, 224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                *(
                    [A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)]
                    if self.train
                    else []
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        for attempt in range(max_attempts):
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                if len(transformed["bboxes"]) > 0:
                    return transformed
            except Exception:
                continue
        return {"image": image, "bboxes": bboxes, "class_labels": labels}

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Augmented image tensor and target dict with 'labels' and 'boxes'.
        """
        label_path = os.path.join(self.labels_path, self.labels[idx])
        image_name = self.labels[idx].split(".")[0]
        image_path = os.path.join(self.images_path, f"{image_name}.jpg")

        img = Image.open(image_path)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        class_labels = []
        bounding_boxes = []
        for annotation in annotations:
            annotation = annotation.split("\n")[:-1][0].split(" ")
            class_labels.append(annotation[0])
            bounding_boxes.append(annotation[1:])
        class_labels = np.array(class_labels).astype(int)
        bounding_boxes = np.array(bounding_boxes).astype(float)

        augmented = self.safe_transform(
            image=np.array(img), bboxes=bounding_boxes, labels=class_labels
        )
        augmented_img_tensor = augmented["image"]
        augmented_bounding_boxes = np.array(augmented["bboxes"])
        augmented_classes = augmented["class_labels"]

        labels = torch.tensor(augmented_classes, dtype=torch.long)
        boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
        return augmented_img_tensor, {"labels": labels, "boxes": boxes}


if __name__ == "__main__":
    dataset = DETRData("data/test", train=False)
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET)
    CLASSES = get_classes()
    fig, ax = plt.subplots(2, 2)
    axs = ax.flatten()
    for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)):
        ax.imshow(img.permute(1, 2, 0))
        box_classes = annotations["labels"]
        boxes = rescale_bboxes(annotations["boxes"], (224, 224))
        for box_class, bbox in zip(box_classes, boxes):
            if box_class != 3:
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                print(xmin, ymin, xmax, ymax)
                ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        color=(0.000, 0.447, 0.741),
                        linewidth=3,
                    )
                )
                text = f"{CLASSES[box_class]}"
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))

    fig.tight_layout()
    plt.show()
