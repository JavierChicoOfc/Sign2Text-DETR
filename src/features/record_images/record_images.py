import os
import time
import uuid
from typing import Any, Dict

import cv2

from features.classes import get_classes
from features.logger import logger

classes = get_classes()


class CaptureImages:
    """
    Class for capturing images from a camera and saving them to disk, organized by class name.
    Handles camera initialization, image capture, and session management with logging and progress display.
    """

    def __init__(self, path: str, classes: Dict[str, Any], camera_id: int) -> None:
        """
        Initialize the CaptureImages object.

        Args:
            path (str): Directory where images will be saved.
            classes (Dict[str, Any]): Dictionary of class names.
            camera_id (int): Camera device ID.
        Raises:
            Exception: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path
        self.classes = classes

        logger.print_banner()
        logger.capture("Image capture system initialized")

        if not self.cap.isOpened():
            logger.capture_error("Camera", f"Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        logger.success(f"Camera {camera_id} connected successfully")

        os.makedirs(self.path, exist_ok=True)
        logger.info(f"Output directory: {self.path}")

    def capture(self, class_name: str) -> bool:
        """
        Capture a single image for the specified class and save it to disk.

        Args:
            class_name (str): The class label for the image.

        Returns:
            bool: True if capture was successful, False otherwise.
        """
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception("Failed to read from camera")
            raw_frame = frame.copy()

            logger.capture("Capturing in 2 seconds...")
            time.sleep(2)
            image = cv2.putText(
                frame,
                f"Capturing {class_name}",
                (0, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                3,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Image Capture", image)

            filename = f"{class_name}-{uuid.uuid1()}.jpg"
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.warning("Quit key pressed - stopping capture")
                return False

            return True
        except Exception as e:
            logger.capture_error(class_name, str(e))
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10) -> None:
        """
        Run the image capture session for all classes.

        Args:
            sleep_time (int, optional): Time to wait between captures (seconds). Defaults to 1.
            num_images (int, optional): Number of images to capture per class. Defaults to 10.
        """
        logger.capture_session_start(self.classes, num_images, sleep_time)
        total_captured = 0

        for class_idx, img_class in enumerate(self.classes):
            logger.capture_class_start(img_class, num_images)
            with logger.create_capture_progress(num_images, img_class) as progress:
                for countdown in range(3, 0, -1):
                    logger.capture(f"Starting capture for class: {img_class} in {countdown}")
                    time.sleep(1)
                class_task = progress.add_task(f"Capturing {img_class}", total=num_images)

                class_captured = 0
                for idx in range(num_images):
                    success = self.capture(img_class)
                    if success:
                        class_captured += 1
                        total_captured += 1
                        logger.capture_success(img_class, idx + 1)
                        progress.update(class_task, advance=1)
                    else:
                        logger.capture_error(img_class, f"Image {idx + 1}")
                        progress.update(class_task, advance=1)
                    time.sleep(sleep_time)

                logger.success(
                    f"Completed {img_class}: {class_captured}/{num_images} images captured"
                )

        logger.capture_session_complete(total_captured, len(self.classes))
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")


if __name__ == "__main__":
    cap = CaptureImages("./data/test", classes, 0)
    cap.run(num_images=11)
