"""
CLI driver for the CUDA image convolution sample.

Examples:
  Build only:
    python scripts/cli.py --build

  Build and run with Gaussian blur:
    python scripts/cli.py --build --image data/input.png --filter gaussian --show
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BINARY_PATH = PROJECT_ROOT / "bin" / "convolution"
OUTPUT_PATH = PROJECT_ROOT / "output.png"

FILTER_MAP = {
    "identity": 0,
    "gaussian": 1,
    "edge": 2,
    "sharpen": 3,
}


def run_build() -> None:
    """Invoke the Makefile to build the CUDA binary."""
    print("Building project with make ...")
    subprocess.run(["make"], cwd=PROJECT_ROOT, check=True)


def run_convolution(image: Path, filter_name: str) -> None:
    """Execute the compiled CUDA binary with the expected menu input."""
    if filter_name not in FILTER_MAP:
        raise ValueError(f"Unsupported filter '{filter_name}'")

    input_stream = f"1\n{image}\n2\n{FILTER_MAP[filter_name]}\n3\n4\n"
    print(f"Running convolution on {image} with filter '{filter_name}' ...")

    completed = subprocess.run(
        [str(BINARY_PATH)],
        input=input_stream,
        text=True,
        capture_output=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    if completed.returncode != 0:
        print("Binary stderr:\n" + completed.stderr)
        print("Binary stdout:\n" + completed.stdout)
        raise RuntimeError("CUDA binary failed. See logs above.")

    if "SUCCESS" not in completed.stdout:
        print("Binary stdout:\n" + completed.stdout)
        raise RuntimeError("CUDA binary did not report success.")

    print(completed.stdout.strip())


def show_results(input_image: Path, output_image: Path) -> None:
    """Display the input and output images side-by-side."""
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("matplotlib is required for --show") from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mpimg.imread(input_image))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mpimg.imread(output_image))
    axes[1].set_title("CUDA Result")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA image convolution driver")
    parser.add_argument("--build", action="store_true", help="Run make before executing")
    parser.add_argument("--image", type=Path, help="Path to input image")
    parser.add_argument(
        "--filter",
        choices=FILTER_MAP.keys(),
        default="gaussian",
        help="Convolution filter to apply",
    )
    parser.add_argument("--show", action="store_true", help="Display input/output images")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.build or not BINARY_PATH.exists():
        run_build()

    if args.image is None:
        print("No --image provided. Build step complete." if args.build else "No image to process.")
        return 0

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    run_convolution(args.image, args.filter)

    if args.show:
        if not OUTPUT_PATH.exists():
            raise FileNotFoundError(f"Expected output image at {OUTPUT_PATH}")
        show_results(args.image, OUTPUT_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
