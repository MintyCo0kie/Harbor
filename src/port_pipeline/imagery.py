from __future__ import annotations

from pathlib import Path

from .models import Coordinate, ImageSpec


class ImageryError(RuntimeError):
    pass


class RequestImageryProvider:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_satellite_image(
        self,
        center: Coordinate,
        image_spec: ImageSpec,
        output_path: Path,
    ) -> Path:
        import requests
        response = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap",
            params={
                "center": f"{center.lat},{center.lng}",
                "zoom": image_spec.zoom,
                "size": f"{image_spec.width}x{image_spec.height}",
                "scale": image_spec.scale,
                "maptype": "satellite",
                "format": "png",
                "key": self.api_key,
            },
            timeout=60,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ImageryError(
                f"Static Maps request did not return an image. Content-Type: {content_type or 'unknown'}"
            )

        output_path.write_bytes(response.content)
        return output_path
