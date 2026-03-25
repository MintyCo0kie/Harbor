from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    maps_api_key: str | None
    places_file: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file(Path(".env"))
        return cls(
            maps_api_key=os.getenv("MAPS_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY"),
            places_file=Path(os.getenv("PLACES_FILE", "data/places.json")),
        )
