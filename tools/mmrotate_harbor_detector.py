from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


LABEL_ALIASES = {
    "container crane": "container-crane",
    "container_crane": "container-crane",
    "storage tank": "storage-tank",
    "storage_tank": "storage-tank",
    "baseball diamond": "baseball-diamond",
    "baseball_diamond": "baseball-diamond",
    "ground track field": "ground-track-field",
    "ground_track_field": "ground-track-field",
    "large vehicle": "large-vehicle",
    "large_vehicle": "large-vehicle",
    "small vehicle": "small-vehicle",
    "small_vehicle": "small-vehicle",
    "soccer ball field": "soccer-ball-field",
    "soccer_ball_field": "soccer-ball-field",
    "swimming pool": "swimming-pool",
    "swimming_pool": "swimming-pool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMRotate harbor detector wrapper")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--config", required=True, help="MMRotate config file path")
    parser.add_argument("--checkpoint", required=True, help="MMRotate checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument(
        "--labels",
        default="harbor",
        help="Comma-separated class labels to export, e.g. harbor,ship,storage-tank,crane",
    )
    parser.add_argument("--mmrotate-root", default=None, help="Path to mmrotate repo root")
    return parser.parse_args()


def resolve_mmrotate_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    # tools/ -> harbor/ -> code/
    return Path(__file__).resolve().parents[2] / "mmrotate"


def normalize_label(label: str) -> str:
    token = " ".join(label.strip().lower().split())
    token = LABEL_ALIASES.get(token, token)
    return token.replace("_", "-")


def main() -> None:
    args = parse_args()
    mmrotate_root = resolve_mmrotate_root(args.mmrotate_root)
    sys.path.insert(0, str(mmrotate_root))

    try:
        import mmrotate  # noqa: F401
        from mmdet.apis import inference_detector, init_detector
        from mmrotate.core import obb2poly_np
        from mmrotate.datasets import DOTADataset
    except ImportError as exc:
        raise SystemExit(f"Failed to import mmrotate dependencies: {exc}")

    model = init_detector(args.config, args.checkpoint, device=args.device)
    results = inference_detector(model, args.image)

    classes = list(DOTADataset.CLASSES)
    requested_labels = [normalize_label(item) for item in args.labels.split(",") if item.strip()]
    if not requested_labels:
        requested_labels = ["harbor"]

    available_labels = {normalize_label(label): index for index, label in enumerate(classes)}
    if "all" in requested_labels:
        selected_labels = list(available_labels.keys())
    else:
        selected_labels = [label for label in requested_labels if label in available_labels]
    if not selected_labels:
        raise SystemExit(
            "None of the requested labels exist in DOTA classes. "
            f"requested={requested_labels}, available={classes}"
        )

    detections: list[dict] = []
    for label in selected_labels:
        class_index = available_labels[label]
        class_bboxes = results[class_index]
        if class_bboxes is None or len(class_bboxes) == 0:
            continue
        class_bboxes = np.asarray(class_bboxes)
        scores = class_bboxes[:, 5]
        keep = scores >= args.score_thr
        filtered = class_bboxes[keep]
        if filtered.size == 0:
            continue

        polys = obb2poly_np(filtered, version="le90")
        polys = polys.reshape(-1, 9)
        for poly_row in polys:
            score = float(poly_row[8])
            points = poly_row[:8].reshape(4, 2)
            detections.append(
                {
                    "label": label,
                    "score": score,
                    "polygon_px": [[float(x), float(y)] for x, y in points],
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"detections": detections}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
