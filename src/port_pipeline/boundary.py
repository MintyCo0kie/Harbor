from __future__ import annotations

from .postprocess import radial_distance


def build_boundary(points: list[tuple[float, float]], alpha: float = 0.25, method: str = "alpha_shape") -> list[list[float]]:
    if len(points) < 3:
        return [[float(x), float(y)] for x, y in points]

    unique_points = _deduplicate(points)
    if len(unique_points) < 3:
        return [[float(x), float(y)] for x, y in unique_points]

    if method == "alpha_shape":
        polygon = _try_concave_hull(unique_points, alpha=alpha)
        if polygon:
            return polygon

    return [[float(x), float(y)] for x, y in convex_hull(unique_points)]


def _try_concave_hull(points: list[tuple[float, float]], alpha: float) -> list[list[float]] | None:
    try:
        from shapely import concave_hull
        from shapely.geometry import MultiPoint
    except ImportError:
        return None

    ratio = min(max(alpha, 0.0), 1.0)
    geometry = concave_hull(MultiPoint(points), ratio=ratio, allow_holes=False)
    if geometry.is_empty:
        return None
    if geometry.geom_type != "Polygon":
        geometry = geometry.convex_hull
        if geometry.is_empty or geometry.geom_type != "Polygon":
            return None

    coords = list(geometry.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    sorted_points = sorted(set(points))
    if len(sorted_points) <= 1:
        return sorted_points

    lower: list[tuple[float, float]] = []
    for point in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def cluster_points(points: list[tuple[float, float]], radius: float) -> list[list[tuple[float, float]]]:
    clusters: list[list[tuple[float, float]]] = []
    for point in points:
        placed = False
        for cluster in clusters:
            if any(radial_distance(point, existing) <= radius for existing in cluster):
                cluster.append(point)
                placed = True
                break
        if not placed:
            clusters.append([point])
    return clusters


def _deduplicate(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        normalized = (round(float(point[0]), 6), round(float(point[1]), 6))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((float(point[0]), float(point[1])))
    return deduped


def _cross(origin: tuple[float, float], point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
