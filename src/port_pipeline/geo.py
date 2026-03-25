from __future__ import annotations

import math

from .models import Coordinate, ImageSpec


TILE_SIZE = 256
EARTH_RADIUS_M = 6378137.0


def latlng_to_world_pixel(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (2**zoom)
    x = (lng + 180.0) / 360.0 * scale
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y


def world_pixel_to_latlng(x: float, y: float, zoom: int) -> Coordinate:
    scale = TILE_SIZE * (2**zoom)
    lng = x / scale * 360.0 - 180.0
    n = math.pi - (2.0 * math.pi * y / scale)
    lat = math.degrees(math.atan(math.sinh(n)))
    return Coordinate(lat=lat, lng=lng)


def image_pixel_to_latlng(
    pixel_x: float,
    pixel_y: float,
    center: Coordinate,
    image_spec: ImageSpec,
) -> Coordinate:
    center_x, center_y = latlng_to_world_pixel(center.lat, center.lng, image_spec.zoom)
    half_width = (image_spec.width * image_spec.scale) / 2.0
    half_height = (image_spec.height * image_spec.scale) / 2.0

    world_x = center_x + (pixel_x * image_spec.scale - half_width)
    world_y = center_y + (pixel_y * image_spec.scale - half_height)
    return world_pixel_to_latlng(world_x, world_y, image_spec.zoom)


def image_pixel_to_world_pixel(
    pixel_x: float,
    pixel_y: float,
    center: Coordinate,
    image_spec: ImageSpec,
) -> tuple[float, float]:
    center_x, center_y = latlng_to_world_pixel(center.lat, center.lng, image_spec.zoom)
    half_width = (image_spec.width * image_spec.scale) / 2.0
    half_height = (image_spec.height * image_spec.scale) / 2.0

    world_x = center_x + (pixel_x * image_spec.scale - half_width)
    world_y = center_y + (pixel_y * image_spec.scale - half_height)
    return world_x, world_y


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Approx meters-per-pixel at the given latitude for Web Mercator."""
    return (math.cos(math.radians(lat)) * 2.0 * math.pi * EARTH_RADIUS_M) / (TILE_SIZE * (2**zoom))


def zoom_for_width_meters(
    lat: float,
    width_px: int,
    meters: float,
    scale: int = 1,
    min_zoom: int = 0,
    max_zoom: int = 22,
) -> int:
    """Solve zoom so the image width roughly covers the requested meters."""
    if width_px <= 0:
        raise ValueError("width_px must be positive")
    if meters <= 0:
        raise ValueError("meters must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")

    numerator = math.cos(math.radians(lat)) * 2.0 * math.pi * EARTH_RADIUS_M * width_px * scale
    denominator = TILE_SIZE * meters
    if denominator <= 0:
        raise ValueError("meters must be positive")

    zoom = math.log2(numerator / denominator)
    zoom_int = int(round(zoom))
    return max(min_zoom, min(max_zoom, zoom_int))
