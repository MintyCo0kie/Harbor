from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .boundary import build_boundary
from .config import Settings
from .detector import build_detector
from .geo import image_pixel_to_world_pixel, latlng_to_world_pixel, world_pixel_to_latlng
from .models import Coordinate, ImageSpec
from .pipeline import PortDetectionPipeline
from .postprocess import extract_boundary_points, polygon_iou


DEFAULT_BOUNDARY_LABELS = {
    "harbor",
}


@dataclass(frozen=True)
class ScanTile:
    tile_id: str
    center: Coordinate
    image_path: str
    output_path: str
    detections: list[dict]
    had_new_harbor: bool
    boundary_signal: float

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "center": {"lat": self.center.lat, "lng": self.center.lng},
            "image_path": self.image_path,
            "output_path": self.output_path,
            "detections": self.detections,
            "had_new_harbor": self.had_new_harbor,
            "boundary_signal": self.boundary_signal,
        }


def scan_port(
    *,
    output_dir: Path,
    place: str | None,
    lat: float | None,
    lng: float | None,
    image_spec: ImageSpec,
    detector_command: str,
    detector_format: str,
    detector_score_threshold: float,
    overlap_ratio: float,
    max_no_new: int,
    bridge_empty_layers: int = 3,
    distance_penalty: float = 0.2,
    directions: list[str],
    max_steps: int,
    boundary_alpha: float,
    boundary_method: str = "alpha_shape",
    boundary_labels: set[str] | None = None,
    boundary_min_score: float = 0.0,
    boundary_label_min_scores: dict[str, float] | None = None,
    step_sleep_seconds: float = 0.5,
    preview_zoom_out_levels: int = 0,
    verbose: bool = False,
) -> dict:
    settings = Settings.from_env()
    pipeline = PortDetectionPipeline(settings)
    center = pipeline.resolve_center(place=place, lat=lat, lng=lng)
    if verbose:
        print(f"Resolved center: lat={center.lat:.6f}, lng={center.lng:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tile_outputs_dir = output_dir / "tile_outputs"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_outputs_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(
        detector_format=detector_format,
        command_template=detector_command,
        score_threshold=detector_score_threshold,
    )
    active_boundary_labels = {
        label.strip().lower() for label in (boundary_labels or DEFAULT_BOUNDARY_LABELS) if label.strip()
    }
    if not active_boundary_labels:
        active_boundary_labels = set(DEFAULT_BOUNDARY_LABELS)
    per_label_thresholds = {
        key.strip().lower(): float(value)
        for key, value in (boundary_label_min_scores or {}).items()
        if key.strip()
    }

    def is_boundary_detection(detection: dict) -> bool:
        label = str(detection.get("label", "")).lower()
        if label not in active_boundary_labels:
            return False
        score = float(detection.get("score", 0.0))
        score_thr = per_label_thresholds.get(label, boundary_min_score)
        return score >= score_thr

    def boundary_signal(tile_detections: list[dict]) -> float:
        return sum(float(item.get("score", 0.0)) for item in tile_detections if is_boundary_detection(item))

    if verbose:
        print("Detector initialized.")
        print(f"Boundary labels: {sorted(active_boundary_labels)}")
        print(f"Boundary minimum score: {boundary_min_score}")
        if per_label_thresholds:
            print(f"Boundary per-label score thresholds: {per_label_thresholds}")

    visited: set[tuple[int, int]] = set()
    all_detections: list[dict] = []
    tiles: list[ScanTile] = []

    step_px_x = int(round(image_spec.width * (1.0 - overlap_ratio)))
    step_px_y = int(round(image_spec.height * (1.0 - overlap_ratio)))

    direction_vectors = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
    }
    active_directions = [direction for direction in directions if direction in direction_vectors]

    def has_new_harbor(tile_detections: list[dict]) -> bool:
        harbor_polys = [d for d in tile_detections if is_boundary_detection(d)]
        if not harbor_polys:
            return False
        existing = [d for d in all_detections if is_boundary_detection(d)]
        if not existing:
            return True
        for candidate in harbor_polys:
            if all(
                polygon_iou(candidate["polygon_world_px"], prior["polygon_world_px"]) < 0.1
                for prior in existing
            ):
                return True
        return False

    def run_single_tile(tile_id: str, tile_center: Coordinate) -> ScanTile:
        if verbose:
            print(f"Scanning tile {tile_id} at lat={tile_center.lat:.6f}, lng={tile_center.lng:.6f}")
        image_path = tiles_dir / f"{tile_id}.png"
        output_path = (
            tile_outputs_dir / f"{tile_id}.json"
            if detector_format == "json"
            else tile_outputs_dir / f"{tile_id}_Task1_harbor.txt"
        )

        pipeline.prepare_image(
            center=tile_center,
            image_spec=image_spec,
            image_path=None,
            destination_path=image_path,
        )

        raw_detections = detector.detect(image_path=image_path, output_path=output_path)
        if verbose:
            print(f"Tile {tile_id} detections: {len(raw_detections)}")
        if step_sleep_seconds > 0:
            time.sleep(step_sleep_seconds)

        formatted: list[dict] = []
        for detection in raw_detections:
            polygon_px = detection["polygon_px"]
            polygon_world = [
                list(image_pixel_to_world_pixel(x, y, tile_center, image_spec))
                for x, y in polygon_px
            ]
            polygon_latlng = [
                {
                    "lat": world_pixel_to_latlng(px, py, image_spec.zoom).lat,
                    "lng": world_pixel_to_latlng(px, py, image_spec.zoom).lng,
                }
                for px, py in polygon_world
            ]
            formatted.append(
                {
                    "label": str(detection["label"]),
                    "score": float(detection["score"]),
                    "polygon_px": [[float(x), float(y)] for x, y in polygon_px],
                    "polygon_world_px": polygon_world,
                    "polygon_latlng": polygon_latlng,
                }
            )

        had_new = has_new_harbor(formatted)
        signal = boundary_signal(formatted)
        if formatted:
            all_detections.extend(formatted)
        return ScanTile(
            tile_id=tile_id,
            center=tile_center,
            image_path=str(image_path),
            output_path=str(output_path),
            detections=formatted,
            had_new_harbor=had_new,
            boundary_signal=signal,
        )

    def mark_visited(cell: tuple[int, int]) -> bool:
        if cell in visited:
            return False
        visited.add(cell)
        return True

    def grid_to_center(gx: int, gy: int, center_world: tuple[float, float]) -> Coordinate:
        world_x = center_world[0] + gx * step_px_x
        world_y = center_world[1] + gy * step_px_y
        return world_pixel_to_latlng(world_x, world_y, image_spec.zoom)

    def grid_tile_id(gx: int, gy: int) -> str:
        x_prefix = "p" if gx >= 0 else "m"
        y_prefix = "p" if gy >= 0 else "m"
        return f"grid_x{x_prefix}{abs(gx):03d}_y{y_prefix}{abs(gy):03d}"

    center_world = latlng_to_world_pixel(center.lat, center.lng, image_spec.zoom)
    center_cell = (0, 0)
    if mark_visited(center_cell):
        if verbose:
            print("Starting center tile scan.")
        tiles.append(run_single_tile("center_000", center))

    # Heap item: (-priority, gx, gy, ttl, inherited_signal)
    frontier_heap: list[tuple[float, int, int, int, float]] = []
    pending_best: dict[tuple[int, int], float] = {}
    pending_ttl: dict[tuple[int, int], int] = {}
    anchor_cells: set[tuple[int, int]] = {center_cell}
    core_cells: set[tuple[int, int]] = set()

    def nearest_anchor_hops(gx: int, gy: int) -> int:
        # Use discovered evidence anchors instead of fixed origin so off-center starts are less biased.
        return min(abs(gx - ax) + abs(gy - ay) for ax, ay in anchor_cells)

    def enqueue_cell(gx: int, gy: int, ttl: int, signal: float) -> None:
        cell = (gx, gy)
        if cell in visited:
            return
        ttl = max(0, int(ttl))
        hop_distance = nearest_anchor_hops(gx, gy)
        priority = float(signal) - float(distance_penalty) * hop_distance + 0.05 * ttl
        previous = pending_best.get(cell)
        if previous is not None and previous >= priority:
            return
        pending_best[cell] = priority
        pending_ttl[cell] = max(pending_ttl.get(cell, 0), ttl)
        heapq.heappush(frontier_heap, (-priority, gx, gy, ttl, float(signal)))

    def enqueue_neighbors(cell: tuple[int, int], ttl: int, signal: float) -> None:
        gx, gy = cell
        for direction in active_directions:
            dx, dy = direction_vectors[direction]
            enqueue_cell(gx + dx, gy + dy, ttl, signal)

    def min_core_hops(cell: tuple[int, int]) -> int:
        gx, gy = cell
        return min(abs(gx - cx) + abs(gy - cy) for cx, cy in core_cells)

    def has_component_compatible_frontier() -> bool:
        if not pending_best:
            return False
        if not core_cells:
            return True
        for cell, priority in pending_best.items():
            ttl = pending_ttl.get(cell, 0)
            # A pending cell is compatible if it can still bridge back to the discovered port component.
            if min_core_hops(cell) <= ttl + 1 and priority > -0.5:
                return True
        return False

    def frontier_pending_stats() -> tuple[int, int]:
        total_pending = len(pending_best)
        if total_pending == 0:
            return 0, 0
        if not core_cells:
            return total_pending, total_pending

        compatible_pending = 0
        for cell, priority in pending_best.items():
            ttl = pending_ttl.get(cell, 0)
            if min_core_hops(cell) <= ttl + 1 and priority > -0.5:
                compatible_pending += 1
        return total_pending, compatible_pending

    seed_ttl = max(0, bridge_empty_layers)
    seed_signal = max(tiles[-1].boundary_signal if tiles else 0.0, 0.1)
    enqueue_neighbors(center_cell, seed_ttl, seed_signal)

    no_new_steps = 0
    expanded_steps = 0

    while frontier_heap and expanded_steps < max_steps:
        neg_priority, gx, gy, parent_ttl, inherited_signal = heapq.heappop(frontier_heap)
        cell = (gx, gy)

        expected = pending_best.get(cell)
        if expected is None:
            continue
        if abs((-neg_priority) - expected) > 1e-9:
            continue

        pending_best.pop(cell, None)
        pending_ttl.pop(cell, None)

        if not mark_visited(cell):
            continue

        expanded_steps += 1
        if verbose:
            print(
                f"Scanning step {expanded_steps}: cell=({gx},{gy}) "
                f"priority={-neg_priority:.3f} ttl={parent_ttl}"
            )

        next_center = grid_to_center(gx, gy, center_world)
        tile = run_single_tile(grid_tile_id(gx, gy), next_center)
        tiles.append(tile)

        if tile.boundary_signal > 0.0:
            core_cells.add(cell)

        if tile.had_new_harbor:
            no_new_steps = 0
            child_ttl = max(0, bridge_empty_layers)
            child_signal = max(tile.boundary_signal, inherited_signal, 0.1)
        else:
            no_new_steps += 1
            child_ttl = max(0, parent_ttl - 1)
            child_signal = max(0.0, inherited_signal * 0.6)
            if verbose:
                print(f"No new harbor in current step. Consecutive no-new count: {no_new_steps}")
            if no_new_steps >= max_no_new:
                if not has_component_compatible_frontier():
                    if verbose:
                        total_pending, compatible_pending = frontier_pending_stats()
                        print(
                            f"Stopping scan after {no_new_steps} consecutive no-new steps "
                            "and no frontier remains compatible with the discovered port component. "
                            f"pending_total={total_pending}, "
                            f"pending_compatible={compatible_pending}"
                        )
                    break
                if verbose:
                    total_pending, compatible_pending = frontier_pending_stats()
                    print(
                        f"Reached max_no_new={max_no_new}, but keeping scan alive "
                        "because frontier still has component-compatible candidates. "
                        f"pending_total={total_pending}, "
                        f"pending_compatible={compatible_pending}"
                    )

        if tile.had_new_harbor or child_ttl > 0:
            if tile.boundary_signal > 0.0:
                anchor_cells.add(cell)
            enqueue_neighbors(cell, child_ttl, child_signal)

    if not frontier_heap and verbose:
        print("No more candidate cells in priority frontier. Stopping.")

    boundary_points = extract_boundary_points(
        [d for d in all_detections if is_boundary_detection(d)],
        include_vertices=True,
        include_centers=True,
        polygon_key="polygon_world_px",
    )
    boundary_polygon_world = build_boundary(boundary_points, alpha=boundary_alpha, method=boundary_method)
    if verbose:
        print(f"Boundary points: {len(boundary_points)}")
        print(f"Boundary polygon vertices: {len(boundary_polygon_world)}")
    boundary_polygon_latlng = [
        {"lat": world_pixel_to_latlng(x, y, image_spec.zoom).lat, "lng": world_pixel_to_latlng(x, y, image_spec.zoom).lng}
        for x, y in boundary_polygon_world
    ]

    preview_path = output_dir / "boundary_polygon.png"
    preview_center, preview_spec = draw_boundary_preview(
        pipeline=pipeline,
        polygon_world=boundary_polygon_world,
        scan_zoom=image_spec.zoom,
        fallback_center=center,
        base_image_spec=image_spec,
        output_path=preview_path,
        preview_zoom_out_levels=preview_zoom_out_levels,
        verbose=verbose,
    )

    result = {
        "center": {"lat": center.lat, "lng": center.lng},
        "image": {
            "width": image_spec.width,
            "height": image_spec.height,
            "zoom": image_spec.zoom,
            "scale": image_spec.scale,
        },
        "scan": {
            "overlap_ratio": overlap_ratio,
            "max_no_new": max_no_new,
            "bridge_empty_layers": bridge_empty_layers,
            "distance_penalty": distance_penalty,
            "directions": directions,
            "max_steps": max_steps,
            "strategy": "priority_frontier_with_bridge_budget",
            "boundary_labels": sorted(active_boundary_labels),
            "boundary_min_score": boundary_min_score,
            "boundary_label_min_scores": per_label_thresholds,
            "step_sleep_seconds": step_sleep_seconds,
            "preview_zoom_out_levels": preview_zoom_out_levels,
        },
        "tiles": [tile.to_dict() for tile in tiles],
        "detections": all_detections,
        "boundary": {
            "method": boundary_method,
            "alpha": boundary_alpha,
            "polygon_world_px": boundary_polygon_world,
            "polygon_latlng": boundary_polygon_latlng,
            "preview_image": str(preview_path),
            "preview_center": {"lat": preview_center.lat, "lng": preview_center.lng},
            "preview_zoom": preview_spec.zoom,
        },
    }
    (output_dir / "result_scan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def draw_boundary_preview(
    pipeline: PortDetectionPipeline,
    polygon_world: list[list[float]],
    scan_zoom: int,
    fallback_center: Coordinate,
    base_image_spec: ImageSpec,
    output_path: Path,
    preview_zoom_out_levels: int = 0,
    verbose: bool = False,
) -> tuple[Coordinate, ImageSpec]:
    preview_center, preview_spec = fit_preview_view(
        polygon_world=polygon_world,
        scan_zoom=scan_zoom,
        fallback_center=fallback_center,
        base_image_spec=base_image_spec,
        preview_zoom_out_levels=preview_zoom_out_levels,
    )
    if verbose:
        print(
            "Preview map request: "
            f"lat={preview_center.lat:.6f}, lng={preview_center.lng:.6f}, "
            f"zoom={preview_spec.zoom}, size={preview_spec.width}x{preview_spec.height}, scale={preview_spec.scale}"
        )

    base_map_path = output_path.with_name("boundary_base_map.png")
    pipeline.prepare_image(
        center=preview_center,
        image_spec=preview_spec,
        image_path=None,
        destination_path=base_map_path,
    )

    with Image.open(base_map_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    actual_width, actual_height = canvas.size
    expected_width = preview_spec.width * preview_spec.scale
    expected_height = preview_spec.height * preview_spec.scale
    if verbose and (actual_width != expected_width or actual_height != expected_height):
        print(
            "Preview base map size differs from requested size: "
            f"actual={actual_width}x{actual_height}, "
            f"requested={expected_width}x{expected_height}. "
            "Using actual size for polygon projection."
        )

    if polygon_world:
        center_world = latlng_to_world_pixel(preview_center.lat, preview_center.lng, preview_spec.zoom)
        half_width = actual_width / 2.0
        half_height = actual_height / 2.0
        zoom_factor = 2 ** (preview_spec.zoom - scan_zoom)
        local_points = []
        for point in polygon_world:
            world_x = float(point[0]) * zoom_factor
            world_y = float(point[1]) * zoom_factor
            pixel_x = world_x - center_world[0] + half_width
            pixel_y = world_y - center_world[1] + half_height
            local_points.append((pixel_x, pixel_y))

        if len(local_points) >= 2:
            draw.line(local_points + [local_points[0]], fill=(255, 0, 0), width=4)

    canvas.save(output_path)
    return preview_center, preview_spec


def fit_preview_view(
    polygon_world: list[list[float]],
    scan_zoom: int,
    fallback_center: Coordinate,
    base_image_spec: ImageSpec,
    preview_zoom_out_levels: int = 0,
) -> tuple[Coordinate, ImageSpec]:
    if not polygon_world:
        return fallback_center, base_image_spec

    points = [(float(point[0]), float(point[1])) for point in polygon_world]

    width_px = base_image_spec.width * base_image_spec.scale
    height_px = base_image_spec.height * base_image_spec.scale
    padding_px = 40
    fit_width = max(32, width_px - 2 * padding_px)
    fit_height = max(32, height_px - 2 * padding_px)

    chosen_zoom = 0
    chosen_center = fallback_center
    max_preview_zoom = min(base_image_spec.zoom, 18)

    for zoom in range(max_preview_zoom, -1, -1):
        zoom_factor = 2 ** (zoom - scan_zoom)
        world_points = [(x * zoom_factor, y * zoom_factor) for x, y in points]
        xs = [item[0] for item in world_points]
        ys = [item[1] for item in world_points]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        mid_x = (min(xs) + max(xs)) / 2.0
        mid_y = (min(ys) + max(ys)) / 2.0
        chosen_center = world_pixel_to_latlng(mid_x, mid_y, zoom)
        chosen_zoom = zoom
        if span_x <= fit_width and span_y <= fit_height:
            break

    if preview_zoom_out_levels > 0:
        chosen_zoom = max(0, chosen_zoom - int(preview_zoom_out_levels))
        zoom_factor = 2 ** (chosen_zoom - scan_zoom)
        world_points = [(x * zoom_factor, y * zoom_factor) for x, y in points]
        xs = [item[0] for item in world_points]
        ys = [item[1] for item in world_points]
        mid_x = (min(xs) + max(xs)) / 2.0
        mid_y = (min(ys) + max(ys)) / 2.0
        chosen_center = world_pixel_to_latlng(mid_x, mid_y, chosen_zoom)

    return chosen_center, ImageSpec(
        width=base_image_spec.width,
        height=base_image_spec.height,
        zoom=chosen_zoom,
        scale=base_image_spec.scale,
    )
