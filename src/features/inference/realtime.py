import time
from typing import Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from features.model.model import DETR
from features.logger.logger import get_logger
from features.rich_handler.rich_handlers import DetectionHandler
from features.classes.setup import get_classes, get_colors


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2), all normalized in [0, 1].
    Args:
        boxes (torch.Tensor): Tensor of shape [N, 4].
    Returns:
        torch.Tensor: Converted boxes of shape [N, 4].
    """
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def scale_xyxy_to_image(boxes_xyxy_norm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Scale normalized (x1, y1, x2, y2) boxes to absolute pixel coordinates.
    Args:
        boxes_xyxy_norm (torch.Tensor): Normalized boxes [N, 4].
        width (int): Image width.
        height (int): Image height.
    Returns:
        torch.Tensor: Absolute pixel coordinates [N, 4].
    """
    scale = torch.tensor(
        [width, height, width, height], dtype=boxes_xyxy_norm.dtype, device=boxes_xyxy_norm.device
    )
    return boxes_xyxy_norm * scale


def ensure_color_tuple(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Ensure color is a tuple of 3 ints for OpenCV.
    Args:
        c (Tuple[int, int, int]): Color input.
    Returns:
        Tuple[int, int, int]: Color as tuple.
    """
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return int(c[0]), int(c[1]), int(c[2])
    return (255, 255, 255)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("realtime")
    detection_handler = DetectionHandler()
    logger.print_banner()
    logger.realtime("Initializing real-time sign language detection...")
    transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    model = DETR(num_classes=11)
    model.load_pretrained("checkpoints/970_model.pt")
    model.to(device)
    model.eval()
    CLASSES = get_classes()
    COLORS = get_colors()
    logger.realtime("Starting camera capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    frame_count = 0
    fps_start_time = time.time()
    try:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                H, W = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                inference_start = time.time()
                transformed = transforms(image=frame_rgb)
                input_tensor = transformed["image"].unsqueeze(0).to(device)
                result = model(input_tensor)
                inference_time_ms = (time.time() - inference_start) * 1000.0
                logits = result["pred_logits"]
                probs = logits.softmax(-1)[..., :-1]
                max_probs, max_classes = probs.max(-1)
                keep_mask = max_probs > 0.7
                boxes_cxcywh = result["pred_boxes"][0]
                boxes_cxcywh = boxes_cxcywh.clamp(0, 1)
                kept_idx = keep_mask[0].nonzero(as_tuple=True)[0]
                boxes_cxcywh_kept = boxes_cxcywh[kept_idx]
                classes_kept = max_classes[0][kept_idx]
                probas_kept = max_probs[0][kept_idx]
                boxes_xyxy_norm = box_cxcywh_to_xyxy(boxes_cxcywh_kept).clamp(0, 1)
                bboxes_px = scale_xyxy_to_image(boxes_xyxy_norm, W, H)
                detections = []
                for cls_idx_t, conf_t, bbox_t in zip(classes_kept, probas_kept, bboxes_px):
                    x1, y1, x2, y2 = bbox_t.detach().cpu().numpy().tolist()
                    cls_idx = int(cls_idx_t.detach().cpu().item())
                    conf = float(conf_t.detach().cpu().item())
                    detections.append(
                        {"class": CLASSES[cls_idx], "confidence": conf, "bbox": [x1, y1, x2, y2]}
                    )
                    color = ensure_color_tuple(COLORS[cls_idx])
                    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{CLASSES[cls_idx]} - {conf:.4f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                    y1_text = max(int(y1) - th - 6, 0)
                    cv2.rectangle(
                        frame_bgr,
                        (int(x1), y1_text),
                        (int(x1) + tw + 6, y1_text + th + 6),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame_bgr,
                        label,
                        (int(x1) + 3, y1_text + th + 3),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - fps_start_time
                    fps = 30.0 / max(elapsed_time, 1e-6)
                    if detections:
                        detection_handler.log_detections(detections, frame_id=frame_count)
                    detection_handler.log_inference_time(inference_time_ms, fps)
                    fps_start_time = time.time()
                cv2.imshow("Frame", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.realtime("Stopping real-time detection...")
                    break
    except KeyboardInterrupt:
        logger.realtime("Interrupted by user. Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
