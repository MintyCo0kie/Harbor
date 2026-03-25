from __future__ import annotations

from .models import Coordinate


class GeocodingError(RuntimeError):
    pass


class RequestGeocoder:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def geocode(self, place: str) -> Coordinate:
        import requests

        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": place,
                "key": self.api_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        status = payload.get("status")
        if status != "OK":
            error_message = payload.get("error_message")
            if error_message:
                raise GeocodingError(f"Google geocoding failed: {status} - {error_message}")
            raise GeocodingError(f"Google geocoding failed: {status}")

        results = payload.get("results", [])
        if not results:
            raise GeocodingError(f"No geocoding result for: {place}")

        location = results[0]["geometry"]["location"]
        return Coordinate(lat=float(location["lat"]), lng=float(location["lng"]))
