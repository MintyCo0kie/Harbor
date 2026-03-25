from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Coordinate:
    lat: float
    lng: float


@dataclass(frozen=True)
class ImageSpec:
    width: int
    height: int
    zoom: int
    scale: int = 1


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    polygon_px: list[tuple[float, float]]
    polygon_latlng: list[Coordinate]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["polygon_latlng"] = [asdict(point) for point in self.polygon_latlng]
        return data


@dataclass(frozen=True)
class Tile:
    tile_id: str
    image_path: str
    offset_x: int
    offset_y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)
