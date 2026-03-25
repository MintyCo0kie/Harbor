from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import ImageSpec
from .tiled_workflow import run_tiled_port_detection


def parse_size(size: str) -> tuple[int, int]:
    if "x" not in size:
        raise argparse.ArgumentTypeError("size must be in WIDTHxHEIGHT format")
    width_str, height_str = size.lower().split("x", maxsplit=1)
    return int(width_str), int(height_str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiled port detection workflow")
    parser.add_argument("--place")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lng", type=float)
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--size", default="640x640")
    parser.add_argument("--scale", type=int, default=1, choices=(1, 2))
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_tiled"))
    parser.add_argument("--detector-command", required=True)
    parser.add_argument("--detector-format", default="dota_task1_harbor", choices=("json", "dota_task1_harbor"))
    parser.add_argument("--detector-score-threshold", type=float, default=0.1)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=200)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--boundary-alpha", type=float, default=0.25)
    parser.add_argument("--boundary-method", default="alpha_shape", choices=("alpha_shape", "convex_hull"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    width, height = parse_size(args.size)

    result = run_tiled_port_detection(
        output_dir=args.output_dir,
        place=args.place,
        lat=args.lat,
        lng=args.lng,
        image_path=args.image_path,
        image_spec=ImageSpec(width=width, height=height, zoom=args.zoom, scale=args.scale),
        detector_command=args.detector_command,
        detector_format=args.detector_format,
        detector_score_threshold=args.detector_score_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        nms_iou_threshold=args.nms_iou_threshold,
        boundary_alpha=args.boundary_alpha,
        boundary_method=args.boundary_method,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
