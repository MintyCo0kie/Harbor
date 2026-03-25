from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .geo import meters_per_pixel, zoom_for_width_meters
from .models import ImageSpec
from .pipeline import PortDetectionPipeline
from .scan_workflow import scan_port

# import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'


LABEL_ALIASES = {
    "container crane": "container-crane",
    "container_crane": "container-crane",
    "storage tank": "storage-tank",
    "storage_tank": "storage-tank",
    "large vehicle": "large-vehicle",
    "large_vehicle": "large-vehicle",
    "small vehicle": "small-vehicle",
    "small_vehicle": "small-vehicle",
}


def normalize_label(label: str) -> str:
    token = " ".join(label.strip().lower().split())
    token = LABEL_ALIASES.get(token, token)
    return token.replace("_", "-")


def parse_size(size: str) -> tuple[int, int]:
    if "x" not in size:
        raise argparse.ArgumentTypeError("size must be in WIDTHxHEIGHT format")
    width_str, height_str = size.lower().split("x", maxsplit=1)
    return int(width_str), int(height_str)


def parse_directions(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def parse_labels(raw: str) -> set[str]:
    return {normalize_label(item) for item in raw.split(",") if item.strip()}


def parse_label_score_thresholds(raw: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(
                "--boundary-label-min-scores must be like 'harbor:0.4,ship:0.7'"
            )
        label, value = token.split(":", maxsplit=1)
        label = normalize_label(label)
        if not label:
            raise argparse.ArgumentTypeError("Label in --boundary-label-min-scores cannot be empty")
        try:
            thresholds[label] = float(value.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid score threshold for label '{label}': {value}"
            ) from exc
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Directional harbor scan workflow")
    parser.add_argument("--place", help="Place name to geocode")
    parser.add_argument("--lat", type=float, help="Known center latitude")
    parser.add_argument("--lng", type=float, help="Known center longitude")
    parser.add_argument("--zoom", type=int, default=None, help="Map zoom level")
    parser.add_argument("--meters", type=float, default=None, help="Target image width in meters (approx)")
    parser.add_argument("--size", default="640x640", help="Image size, e.g. 640x640")
    parser.add_argument("--scale", type=int, default=1, choices=(1, 2), help="Static map scale")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_scan"))
    parser.add_argument("--detector-command")
    parser.add_argument("--mmrotate-config", type=Path, help="MMRotate config file")
    parser.add_argument("--mmrotate-checkpoint", type=Path, help="MMRotate checkpoint file")
    parser.add_argument("--mmrotate-root", type=Path, help="MMRotate repository root")
    parser.add_argument("--mmrotate-device", default="cuda:0", help="MMRotate device")
    parser.add_argument("--mmrotate-score-thr", type=float, default=0.3, help="MMRotate score threshold")
    parser.add_argument(
        "--mmrotate-labels",
        default="harbor",
        help="Comma-separated MMRotate classes to export (useful defaults for port boundary iteration)",
    )
    parser.add_argument(
        "--detector-format",
        default="json",
        choices=("json", "dota_task1_harbor"),
        help="Detector output format.",
    )
    parser.add_argument("--detector-score-threshold", type=float, default=0.3)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-no-new", type=int, default=3)
    parser.add_argument("--directions", default="left,up,down,right")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--boundary-alpha", type=float, default=0.25)
    parser.add_argument("--boundary-method", default="alpha_shape", choices=("alpha_shape", "convex_hull"))
    parser.add_argument(
        "--boundary-labels",
        default="harbor",
        help="Comma-separated labels used to build the final port boundary",
    )
    parser.add_argument(
        "--boundary-min-score",
        type=float,
        default=0.0,
        help="Global minimum score for detections used in boundary building",
    )
    parser.add_argument(
        "--boundary-label-min-scores",
        type=parse_label_score_thresholds,
        default={},
        help="Per-label minimum scores, e.g. harbor:0.4,ship:0.7",
    )
    parser.add_argument(
        "--step-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep seconds after each tile detection",
    )
    parser.add_argument(
        "--preview-zoom-out-levels",
        type=int,
        default=0,
        help="Zoom out N levels from the auto-fitted preview zoom for boundary preview image",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress updates")
    return parser


def _quote(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _build_mmrotate_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        _quote(str(Path("tools") / "mmrotate_harbor_detector.py")),
        "--image",
        "{image}",
        "--output",
        "{output}",
        "--config",
        _quote(str(args.mmrotate_config)),
        "--checkpoint",
        _quote(str(args.mmrotate_checkpoint)),
        "--device",
        _quote(str(args.mmrotate_device)),
        "--score-thr",
        str(args.mmrotate_score_thr),
        "--labels",
        _quote(str(args.mmrotate_labels)),
    ]
    if args.mmrotate_root:
        parts.extend(["--mmrotate-root", _quote(str(args.mmrotate_root))])
    return " ".join(parts)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    width, height = parse_size(args.size)

    detector_command = args.detector_command
    if not detector_command:
        if not args.mmrotate_config or not args.mmrotate_checkpoint:
            parser.error("--detector-command or both --mmrotate-config and --mmrotate-checkpoint are required")
        detector_command = _build_mmrotate_command(args)

    zoom = args.zoom
    place = args.place
    lat = args.lat
    lng = args.lng
    if args.meters is not None and zoom is None:
        pipeline = PortDetectionPipeline(Settings.from_env())
        center = pipeline.resolve_center(place=place, lat=lat, lng=lng)
        zoom = zoom_for_width_meters(
            center.lat,
            width_px=width,
            meters=args.meters,
            scale=args.scale,
        )
        place = None
        lat = center.lat
        lng = center.lng
        if args.verbose:
            approx_width = width * meters_per_pixel(center.lat, zoom)
            print(
                f"Auto zoom from --meters={args.meters}: zoom={zoom} "
                f"(approx width={approx_width:.1f}m at lat={center.lat:.6f})"
            )
    elif args.meters is not None and zoom is not None and args.verbose:
        print(
            f"Using explicit --zoom={zoom}; --meters={args.meters} is not used for zoom calculation."
        )
    if zoom is None:
        zoom = 17
    if args.verbose:
        print(f"Final scan image spec: {width}x{height}, zoom={zoom}, scale={args.scale}")

    result = scan_port(
        output_dir=args.output_dir,
        place=place,
        lat=lat,
        lng=lng,
        image_spec=ImageSpec(width=width, height=height, zoom=zoom, scale=args.scale),
        detector_command=detector_command,
        detector_format=args.detector_format,
        detector_score_threshold=args.detector_score_threshold,
        overlap_ratio=args.overlap_ratio,
        max_no_new=args.max_no_new,
        directions=parse_directions(args.directions),
        max_steps=args.max_steps,
        boundary_alpha=args.boundary_alpha,
        boundary_method=args.boundary_method,
        boundary_labels=parse_labels(args.boundary_labels),
        boundary_min_score=args.boundary_min_score,
        boundary_label_min_scores=args.boundary_label_min_scores,
        step_sleep_seconds=args.step_sleep_seconds,
        preview_zoom_out_levels=args.preview_zoom_out_levels,
        verbose=args.verbose,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
