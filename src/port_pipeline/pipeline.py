from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import Settings
from .detector import build_detector
from .geo import image_pixel_to_latlng
from .geocoding import RequestGeocoder
from .imagery import RequestImageryProvider
from .models import Coordinate, Detection, ImageSpec
from .place_lookup import PlaceLookup


class PipelineError(RuntimeError):
    pass


class PortDetectionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @classmethod
    def from_runtime(cls) -> "PortDetectionPipeline":
        return cls(Settings.from_env())

    def resolve_center(self, place: str | None, lat: float | None, lng: float | None) -> Coordinate:
        if lat is not None and lng is not None:
            return Coordinate(lat=lat, lng=lng)
        if not place:
            raise PipelineError("You must provide either --place or both --lat and --lng.")
        local_match = PlaceLookup(self.settings.places_file).resolve(place)
        if local_match is not None:
            return local_match
        if not self.settings.maps_api_key:
            raise PipelineError(
                "Place not found in the local registry, and MAPS_API_KEY is not configured for API geocoding."
            )
        geocoder = RequestGeocoder(self.settings.maps_api_key)
        return geocoder.geocode(place)

    def run(
        self,
        output_dir: Path,
        image_spec: ImageSpec,
        place: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
        image_path: Path | None = None,
        detector_command: str | None = None,
        detector_format: str = "json",
        detector_score_threshold: float = 0.0,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        center = self.resolve_center(place=place, lat=lat, lng=lng)

        normalized_image_path = output_dir / "satellite.png"
        detector_output_path = (
            output_dir / "detector_output.json"
            if detector_format == "json"
            else output_dir / "Task1_harbor.txt"
        )
        result_path = output_dir / "result.json"

        self.prepare_image(
            center=center,
            image_spec=image_spec,
            image_path=image_path,
            destination_path=normalized_image_path,
        )

        detector = build_detector(
            detector_format=detector_format,
            command_template=detector_command,
            score_threshold=detector_score_threshold,
        )
        raw_detections = detector.detect(
            image_path=normalized_image_path,
            output_path=detector_output_path,
        )

        detections: list[Detection] = []
        for item in raw_detections:
            polygon_px = [tuple(point) for point in item["polygon_px"]]
            polygon_latlng = [
                image_pixel_to_latlng(
                    pixel_x=point[0],
                    pixel_y=point[1],
                    center=center,
                    image_spec=image_spec,
                )
                for point in polygon_px
            ]
            detections.append(
                Detection(
                    label=item["label"],
                    score=float(item["score"]),
                    polygon_px=polygon_px,
                    polygon_latlng=polygon_latlng,
                )
            )

        result = {
            "center": {"lat": center.lat, "lng": center.lng},
            "image": {
                "path": str(normalized_image_path),
                "width": image_spec.width,
                "height": image_spec.height,
                "zoom": image_spec.zoom,
                "scale": image_spec.scale,
            },
            "detections": [detection.to_dict() for detection in detections],
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def prepare_image(
        self,
        *,
        center: Coordinate,
        image_spec: ImageSpec,
        image_path: Path | None,
        destination_path: Path,
    ) -> Path:
        if image_path is not None:
            if not image_path.exists():
                raise PipelineError(f"Input image not found: {image_path}")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, destination_path)
            return destination_path

        if not self.settings.maps_api_key:
            raise PipelineError(
                "MAPS_API_KEY is required to fetch satellite imagery when --image-path is not provided."
            )
        imagery = RequestImageryProvider(self.settings.maps_api_key)
        imagery.fetch_satellite_image(
            center=center,
            image_spec=image_spec,
            output_path=destination_path,
        )
        return destination_path
