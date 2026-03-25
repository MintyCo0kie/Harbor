from __future__ import annotations

import json
from pathlib import Path

from .models import Coordinate


BUILTIN_PLACES: dict[str, Coordinate] = {
    "singapore": Coordinate(lat=1.2644, lng=103.8223),
    "singapore port": Coordinate(lat=1.2644, lng=103.8223),
    "port of singapore": Coordinate(lat=1.2644, lng=103.8223),
    "shanghai": Coordinate(lat=31.2304, lng=121.4737),
    "port of shanghai": Coordinate(lat=31.2304, lng=121.4737),
    "rotterdam": Coordinate(lat=51.9475, lng=4.1389),
    "port of rotterdam": Coordinate(lat=51.9475, lng=4.1389),
}


class PlaceLookup:
    def __init__(self, places_file: Path) -> None:
        self.places_file = places_file

    def resolve(self, place: str) -> Coordinate | None:
        normalized = self._normalize(place)
        if normalized in BUILTIN_PLACES:
            return BUILTIN_PLACES[normalized]

        file_places = self._load_file_places()
        return file_places.get(normalized)

    def _load_file_places(self) -> dict[str, Coordinate]:
        if not self.places_file.exists():
            return {}

        payload = json.loads(self.places_file.read_text(encoding="utf-8"))
        places: dict[str, Coordinate] = {}
        for key, value in payload.items():
            places[self._normalize(key)] = Coordinate(
                lat=float(value["lat"]),
                lng=float(value["lng"]),
            )
        return places

    @staticmethod
    def _normalize(place: str) -> str:
        return " ".join(place.strip().lower().split())
