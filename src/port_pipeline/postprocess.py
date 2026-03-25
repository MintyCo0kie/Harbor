from __future__ import annotations

import math

from .models import Coordinate, ImageSpec, Tile
from .geo import image_pixel_to_latlng


TARGET_PORT_CLASSES = {"harbor", "ship", "storage-tank", "crane"}


def remap_tile_detection_to_global(detection: dict, tile: Tile) -> dict:
    polygon_px = [
        [float(point[0]) + tile.offset_x, float(point[1]) + tile.offset_y]
        for point in detection["polygon_px"]
    ]
    return {
        "label": str(detection["label"]),
        "score": float(detection["score"]),
        "polygon_px": polygon_px,
        "tile_id": tile.tile_id,
    }


def filter_detections_by_label(detections: list[dict], allowed_labels: set[str]) -> list[dict]:
    normalized = {label.lower() for label in allowed_labels}
    return [d for d in detections if str(d["label"]).lower() in normalized]


def nms_on_detections(detections: list[dict], iou_threshold: float) -> list[dict]:
    kept: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for detection in detections:
        grouped.setdefault(str(detection["label"]).lower(), []).append(detection)

    for _, group in grouped.items():
        sorted_group = sorted(group, key=lambda item: float(item["score"]), reverse=True)
        while sorted_group:
            best = sorted_group.pop(0)
            kept.append(best)
            survivors: list[dict] = []
            for candidate in sorted_group:
                if polygon_iou(best["polygon_px"], candidate["polygon_px"]) < iou_threshold:
                    survivors.append(candidate)
            sorted_group = survivors
    return kept


def polygon_iou(poly_a: list[list[float]], poly_b: list[list[float]]) -> float:
    min_ax, min_ay, max_ax, max_ay = polygon_bounds(poly_a)
    min_bx, min_by, max_bx, max_by = polygon_bounds(poly_b)

    inter_w = max(0.0, min(max_ax, max_bx) - max(min_ax, min_bx))
    inter_h = max(0.0, min(max_ay, max_by) - max(min_ay, min_by))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, max_ax - min_ax) * max(0.0, max_ay - min_ay)
    area_b = max(0.0, max_bx - min_bx) * max(0.0, max_by - min_by)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def polygon_bounds(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_centroid(polygon: list[list[float]]) -> tuple[float, float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def polygon_area(polygon: list[list[float]]) -> float:
    area = 0.0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def extract_boundary_points(
    detections: list[dict],
    include_vertices: bool = True,
    include_centers: bool = True,
    polygon_key: str = "polygon_px",
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for detection in detections:
        polygon = detection.get(polygon_key)
        if not polygon:
            continue
        if include_vertices:
            points.extend((float(x), float(y)) for x, y in polygon)
        if include_centers:
            points.append(polygon_centroid(polygon))
    return points


def polygon_px_to_latlng(
    polygon_px: list[list[float]],
    center: Coordinate,
    image_spec: ImageSpec,
) -> list[dict]:
    converted = []
    for pixel_x, pixel_y in polygon_px:
        coordinate = image_pixel_to_latlng(pixel_x, pixel_y, center, image_spec)
        converted.append({"lat": coordinate.lat, "lng": coordinate.lng})
    return converted


def radial_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.sqrt(dx * dx + dy * dy)
