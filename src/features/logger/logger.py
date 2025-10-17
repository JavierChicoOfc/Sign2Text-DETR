import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "debug": "dim blue",
        "data": "magenta",
        "model": "blue",
        "training": "green",
        "test": "yellow",
        "realtime": "bright_cyan",
        "detection": "bright_green",
        "bbox": "orange3",
        "loss": "red3",
        "accuracy": "green3",
    }
)


class SignLanguageLogger:
    """
    Enhanced logger with Rich formatting for Sign Language Detection project.
    Provides methods for logging, progress bars, and rich console output.
    """

    def __init__(self, name: str = "sign_language", level: str = "INFO") -> None:
        """
        Initialize the logger with a custom theme and handlers.

        Args:
            name (str): Logger name.
            level (str): Logging level (e.g., 'INFO', 'DEBUG').
        """
        self.console = Console(theme=custom_theme)
        self.name = name
        self.level = getattr(logging, level.upper())
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Setup the logging configuration with rich formatting and file output.
        """
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Rich handler for console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(self.level)

        # File handler for persistent logs
        log_file = self.logs_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(rich_handler)
        self.logger.addHandler(file_handler)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with rich formatting."""
        self.logger.info(f"[info]{message}[/info]", **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with rich formatting."""
        self.logger.warning(f"[warning]{message}[/warning]", **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with rich formatting."""
        self.logger.error(f"[error]{message}[/error]", **kwargs)

    def success(self, message: str, **kwargs) -> None:
        """Log success message with rich formatting."""
        self.logger.info(f"[success]✅ {message}[/success]", **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with rich formatting."""
        self.logger.debug(f"[debug]{message}[/debug]", **kwargs)

    def data(self, message: str, **kwargs) -> None:
        """Log data-related message with rich formatting."""
        self.logger.info(f"[data]📊 {message}[/data]", **kwargs)

    def model(self, message: str, **kwargs) -> None:
        """Log model-related message with rich formatting."""
        self.logger.info(f"[model]🤖 {message}[/model]", **kwargs)

    def training(self, message: str, **kwargs) -> None:
        """Log training-related message with rich formatting."""
        self.logger.info(f"[training]🏋️ {message}[/training]", **kwargs)

    def test(self, message: str, **kwargs) -> None:
        """Log test-related message with rich formatting."""
        self.logger.info(f"[test]🧪 {message}[/test]", **kwargs)

    def realtime(self, message: str, **kwargs) -> None:
        """Log realtime-related message with rich formatting."""
        self.logger.info(f"[realtime]📹 {message}[/realtime]", **kwargs)

    def detection(self, message: str, **kwargs) -> None:
        """Log detection-related message with rich formatting."""
        self.logger.info(f"[detection]🎯 {message}[/detection]", **kwargs)

    def print_panel(self, title: str, content: str, style: str = "blue") -> None:
        """
        Print content in a rich panel.

        Args:
            title (str): Panel title.
            content (str): Panel content.
            style (str): Panel style.
        """
        panel = Panel(content, title=title, style=style, border_style=style)
        self.console.print(panel)

    def print_table(self, title: str, headers: List[str], rows: List[List[Any]]) -> None:
        """
        Print data in a rich table.

        Args:
            title (str): Table title.
            headers (List[str]): List of column headers.
            rows (List[List[Any]]): Table rows.
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")
        for header in headers:
            table.add_column(header, style="cyan")
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self.console.print(table)

    def print_status(self, status: str, message: str, style: str = "blue") -> None:
        """
        Print a status message with icon.

        Args:
            status (str): Status type (e.g., 'info', 'success').
            message (str): Status message.
            style (str): Rich style for the message.
        """
        status_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "loading": "⏳",
            "done": "🎉",
            "data": "📊",
            "model": "🤖",
            "training": "🏋️",
            "test": "🧪",
            "realtime": "📹",
            "detection": "🎯",
        }
        icon = status_icons.get(status, "•")
        self.console.print(f"[{style}]{icon} {message}[/{style}]")

    def create_progress(self, description: str = "Processing...") -> Progress:
        """
        Create a rich progress bar.

        Args:
            description (str): Description for the progress bar.

        Returns:
            Progress: Rich Progress object.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

    def create_training_progress(self, loss_string: str, type: str = "Train") -> Progress:
        """
        Create a specialized progress bar for training.

        Args:
            loss_string (str): Description of the loss metric.
            type (str): Type of progress (e.g., 'Train', 'Validation').

        Returns:
            Progress: Rich Progress object.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{type} Progress"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )

    def print_banner(self) -> None:
        """
        Print the Sign Language Detection project banner.
        """
        banner = """
        ███████╗██╗ ██████╗ ███╗   ██╗██████╗ ████████╗███████╗██╗  ██╗████████╗
        ██╔════╝██║██╔════╝ ████╗  ██║╚════██╗╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
        ███████╗██║██║  ███╗██╔██╗ ██║ █████╔╝   ██║   █████╗   ╚███╔╝    ██║   
        ╚════██║██║██║   ██║██║╚██╗██║██╔═══╝    ██║   ██╔══╝   ██╔██╗    ██║   
        ███████║██║╚██████╔╝██║ ╚████║███████╗   ██║   ███████╗██╔╝ ██╗   ██║   
        ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝                                                                  
        """
        self.console.print(Panel(banner, style="bold cyan", border_style="blue", expand=False))

    def print_model_summary(self, model_info: Dict[str, Any]) -> None:
        """
        Print model architecture summary.

        Args:
            model_info (Dict[str, Any]): Model configuration parameters.
        """
        table = Table(
            title="🤖 DETR Model Configuration", show_header=True, header_style="bold blue"
        )
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        for key, value in model_info.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def print_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """
        Print dataset information.

        Args:
            dataset_info (Dict[str, Any]): Dataset metrics and info.
        """
        table = Table(title="📊 Dataset Information", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        for key, value in dataset_info.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def print_detection_results(self, detections: List[Dict[str, Any]]) -> None:
        """
        Print detection results in a formatted table.

        Args:
            detections (List[Dict[str, Any]]): List of detection results.
        """
        if not detections:
            self.console.print("[yellow]No detections found[/yellow]")
            return
        table = Table(title="🎯 Detection Results", show_header=True, header_style="bold green")
        table.add_column("Class", style="cyan")
        table.add_column("Confidence", style="yellow")
        table.add_column("BBox (x1,y1,x2,y2)", style="orange3")
        for detection in detections:
            table.add_row(
                detection.get("class", "Unknown"),
                f"{detection.get('confidence', 0):.3f}",
                f"({detection.get('bbox', [0, 0, 0, 0])[0]:.1f}, "
                f"{detection.get('bbox', [0, 0, 0, 0])[1]:.1f}, "
                f"{detection.get('bbox', [0, 0, 0, 0])[2]:.1f}, "
                f"{detection.get('bbox', [0, 0, 0, 0])[3]:.1f})",
            )
        self.console.print(table)

    def print_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        test_loss: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """
        Print training metrics in a formatted way.

        Args:
            epoch (int): Current epoch.
            train_loss (float): Training loss.
            test_loss (Optional[float]): Test loss.
            lr (Optional[float]): Learning rate.
        """
        metrics_text = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"
        if test_loss is not None:
            metrics_text += f" | Test Loss: {test_loss:.4f}"
        if lr is not None:
            metrics_text += f" | LR: {lr:.2e}"
        self.console.print(f"[training]{metrics_text}[/training]")

    def capture(self, message: str, **kwargs) -> None:
        """
        Log capture-related message with rich formatting.

        Args:
            message (str): Message to log.
        """
        self.logger.info(f"[realtime]📸 {message}[/realtime]", **kwargs)

    def capture_success(self, class_name: str, image_count: int, **kwargs) -> None:
        """
        Log successful image capture.

        Args:
            class_name (str): Name of the class.
            image_count (int): Image number.
        """
        self.console.print(f"[success]✅ Captured {class_name} image #{image_count}[/success]")

    def capture_error(self, class_name: str, error: str, **kwargs) -> None:
        """
        Log capture error with rich formatting.

        Args:
            class_name (str): Name of the class.
            error (str): Error message.
        """
        self.console.print(f"[error]❌ Failed to capture {class_name}: {error}[/error]")

    def capture_class_start(self, class_name: str, total_images: int) -> None:
        """
        Print when starting to capture a new class.

        Args:
            class_name (str): Name of the class.
            total_images (int): Number of images to capture.
        """
        self.console.print(f"\n[bold cyan]🎯 Starting capture for class: {class_name}[/bold cyan]")
        self.console.print(f"[dim]Target: {total_images} images[/dim]")

    def capture_session_start(
        self, classes: List[str], images_per_class: int, sleep_time: int
    ) -> None:
        """
        Print capture session information.

        Args:
            classes (List[str]): List of class names.
            images_per_class (int): Images per class.
            sleep_time (int): Sleep time between captures.
        """
        session_info = [
            ["Total Classes", str(len(classes))],
            ["Images per Class", str(images_per_class)],
            ["Total Images", str(len(classes) * images_per_class)],
            ["Sleep Time", f"{sleep_time}s"],
            ["Classes", ", ".join(classes)],
        ]
        self.print_table("📸 Image Capture Session", ["Parameter", "Value"], session_info)

    def capture_session_complete(self, total_captured: int, total_classes: int) -> None:
        """
        Print capture session completion.

        Args:
            total_captured (int): Total images captured.
            total_classes (int): Number of classes processed.
        """
        self.console.print("\n[bold green]🎉 Capture session completed![/bold green]")
        self.console.print(f"[success]✅ Total images captured: {total_captured}[/success]")
        self.console.print(f"[success]✅ Classes processed: {total_classes}[/success]")

    def create_capture_progress(self, total_images: int, class_name: str) -> Progress:
        """
        Create a progress bar specifically for image capture.

        Args:
            total_images (int): Total images to capture.
            class_name (str): Name of the class.

        Returns:
            Progress: Rich Progress object.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Capturing {class_name}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )

logger = SignLanguageLogger()


def get_logger(name: str = "sign_language") -> SignLanguageLogger:
    """
    Get a logger instance.

    Args:
        name (str): Logger name.

    Returns:
        SignLanguageLogger: Logger instance.
    """
    return SignLanguageLogger(name)
