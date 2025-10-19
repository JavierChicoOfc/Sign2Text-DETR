import argparse
import importlib
import sys
from pathlib import Path

SCRIPTS_MAP = {
    "train": "features.inference.train",
    "test": "features.inference.test",
    "realtime": "features.inference.realtime",
    "record_images": "record_images.record_images",
}


def import_module_by_map(key: str):
    """
    Import target module trying both with and without the 'src.' prefix.
    """
    rel_path = SCRIPTS_MAP[key]
    candidates = [f"src.{rel_path}", rel_path]
    last_err = None
    for mod in candidates:
        try:
            return importlib.import_module(mod)
        except ModuleNotFoundError as e:
            last_err = e
            continue
    raise last_err


def main():
    parser = argparse.ArgumentParser(description="Main entry point for project scripts.")
    parser.add_argument(
        "script",
        type=str,
        choices=["train", "test", "realtime", "record_images"],
        help="Script to execute.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Mode for the test script ('train' or 'test').",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for test inference.")
    parser.add_argument(
        "--num-classes", type=int, default=11, help="Number of classes for inference."
    )

    parser.add_argument(
        "--source", type=str, default=None, help="Recording source (e.g., 'webcam' or path)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for recorded images."
    )
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for recording.")
    parser.add_argument("--duration", type=int, default=None, help="Duration in seconds.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        module = import_module_by_map(args.script)

        if args.script == "train":
            if not hasattr(module, "train"):
                raise AttributeError("The 'train.py' module must define a 'train()' function.")
            module.train()

        elif args.script == "test":
            if not hasattr(module, "run_test_inference"):
                raise AttributeError(
                    "The 'test.py' module must define a 'run_test_inference()' function."
                )
            module.run_test_inference(
                mode=args.mode,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
            )

        elif args.script == "realtime":
            if not hasattr(module, "realtime"):
                raise AttributeError(
                    "The 'realtime.py' module must define a 'realtime()' function."
                )
            module.realtime()

        elif args.script == "record_images":
            if not hasattr(module, "record_images"):
                raise AttributeError(
                    "The 'record_images.py' module must define a 'record_images()' function."
                )
            module.record_images(
                source=args.source,
                output_dir=args.output_dir,
                fps=args.fps,
                duration=args.duration,
            )

    except Exception as e:
        print(f"⚠️ Error executing '{args.script}': {e}")


if __name__ == "__main__":
    main()
