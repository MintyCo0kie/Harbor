from __future__ import annotations

import json
from pathlib import Path

from .boundary import build_boundary
from .detector import build_detector
from .models import ImageSpec
from .pipeline import PortDetectionPipeline
from .postprocess import (
    TARGET_PORT_CLASSES,
    extract_boundary_points,
    filter_detections_by_label,
    nms_on_detections,
    polygon_px_to_latlng,
    remap_tile_detection_to_global,
)
from .tiling import save_tiles


def run_tiled_port_detection(
    *,
    output_dir: Path,
    place: str | None,
    lat: float | None,
    lng: float | None,
    image_path: Path | None,
    image_spec: ImageSpec,
    detector_command: str,
    detector_format: str,
    detector_score_threshold: float,
    tile_size: int,
    tile_overlap: int,
    nms_iou_threshold: float,
    boundary_alpha: float,
    boundary_method: str = "alpha_shape",
) -> dict:
    pipeline = PortDetectionPipeline.from_runtime()
    center = pipeline.resolve_center(place=place, lat=lat, lng=lng)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = output_dir / "satellite.png"
    pipeline.prepare_image(
        center=center,
        image_spec=image_spec,
        image_path=image_path,
        destination_path=source_image_path,
    )

    tiles_dir = output_dir / "tiles"
    tile_outputs_dir = output_dir / "tile_outputs"
    tile_outputs_dir.mkdir(parents=True, exist_ok=True)
    tiles = save_tiles(source_image_path, tiles_dir, tile_size=tile_size, overlap=tile_overlap)
    detector = build_detector(
        detector_format=detector_format,
        command_template=detector_command,
        score_threshold=detector_score_threshold,
    )

    global_detections: list[dict] = []
    for tile in tiles:
        tile_output_path = (
            tile_outputs_dir / f"{tile.tile_id}.json"
            if detector_format == "json"
            else tile_outputs_dir / f"{tile.tile_id}_Task1_harbor.txt"
        )
        raw_detections = detector.detect(
            image_path=Path(tile.image_path),
            output_path=tile_output_path,
        )
        for detection in raw_detections:
            global_detections.append(remap_tile_detection_to_global(detection, tile))

    filtered_detections = filter_detections_by_label(global_detections, TARGET_PORT_CLASSES)
    merged_detections = nms_on_detections(filtered_detections, iou_threshold=nms_iou_threshold)

    boundary_points = extract_boundary_points(merged_detections, include_vertices=True, include_centers=True)
    boundary_polygon_px = build_boundary(boundary_points, alpha=boundary_alpha, method=boundary_method)
    boundary_polygon_latlng = polygon_px_to_latlng(boundary_polygon_px, center=center, image_spec=image_spec)

    result = {
        "center": {"lat": center.lat, "lng": center.lng},
        "image": {
            "path": str(source_image_path),
            "width": image_spec.width,
            "height": image_spec.height,
            "zoom": image_spec.zoom,
            "scale": image_spec.scale,
        },
        "tiles": [tile.to_dict() for tile in tiles],
        "detections": merged_detections,
        "boundary": {
            "method": boundary_method,
            "alpha": boundary_alpha,
            "polygon_px": boundary_polygon_px,
            "polygon_latlng": boundary_polygon_latlng,
        },
    }
    (output_dir / "result_tiled.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
