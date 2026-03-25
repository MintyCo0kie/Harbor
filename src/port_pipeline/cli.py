from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .models import ImageSpec
from .pipeline import PortDetectionPipeline


def parse_size(size: str) -> tuple[int, int]:
    if "x" not in size:
        raise argparse.ArgumentTypeError("size must be in WIDTHxHEIGHT format")
    width_str, height_str = size.lower().split("x", maxsplit=1)
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must contain integer width and height") from exc
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Port detection pipeline")
    parser.add_argument("--place", help="Place name to geocode")
    parser.add_argument("--lat", type=float, help="Known center latitude")
    parser.add_argument("--lng", type=float, help="Known center longitude")
    parser.add_argument("--zoom", type=int, default=17, help="Map zoom level")
    parser.add_argument("--size", default="640x640", help="Image size, e.g. 640x640")
    parser.add_argument("--scale", type=int, default=1, choices=(1, 2), help="Static map scale")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--image-path", type=Path, help="Use an existing local image instead of downloading one")
    parser.add_argument(
        "--detector-command",
        help="External detector command template. Must contain {image} and {output}.",
    )
    parser.add_argument(
        "--detector-format",
        default="json",
        choices=("json", "dota_task1_harbor"),
        help="Detector output format.",
    )
    parser.add_argument(
        "--detector-score-threshold",
        type=float,
        default=0.0,
        help="Minimum detector score to keep.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    width, height = parse_size(args.size)

    pipeline = PortDetectionPipeline(Settings.from_env())
    result = pipeline.run(
        output_dir=args.output_dir,
        image_spec=ImageSpec(width=width, height=height, zoom=args.zoom, scale=args.scale),
        place=args.place,
        lat=args.lat,
        lng=args.lng,
        image_path=args.image_path,
        detector_command=args.detector_command,
        detector_format=args.detector_format,
        detector_score_threshold=args.detector_score_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
