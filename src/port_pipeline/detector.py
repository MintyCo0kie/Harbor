from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path


class DetectorError(RuntimeError):
    pass


def run_detector_command(command_template: str, image_path: Path, output_path: Path) -> None:
    command = command_template.format(
        image=str(image_path),
        output=str(output_path),
        image_id=image_path.stem,
    )

    def _strip_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            return value[1:-1]
        return value

    args = [_strip_quotes(arg) for arg in shlex.split(command, posix=False)]

    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DetectorError(
            "Detector command failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


class ExternalJsonDetector:
    def __init__(self, command_template: str | None = None) -> None:
        self.command_template = command_template

    def detect(self, image_path: Path, output_path: Path) -> list[dict]:
        if not self.command_template:
            return []

        run_detector_command(self.command_template, image_path=image_path, output_path=output_path)

        if not output_path.exists():
            raise DetectorError(f"Detector output not found: {output_path}")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        detections = payload.get("detections")
        if not isinstance(detections, list):
            raise DetectorError("Detector output JSON must contain a 'detections' list.")
        return detections


class DotATask1HarborDetector:
    def __init__(self, command_template: str | None = None, score_threshold: float = 0.0) -> None:
        self.command_template = command_template
        self.score_threshold = score_threshold

    def detect(self, image_path: Path, output_path: Path) -> list[dict]:
        if not self.command_template:
            if not output_path.exists():
                return []
        else:
            run_detector_command(self.command_template, image_path=image_path, output_path=output_path)

        task1_path = self._resolve_task1_file(output_path)
        if not task1_path.exists():
            raise DetectorError(f"DOTA Task1 harbor output not found: {task1_path}")

        detections: list[dict] = []
        image_id = image_path.stem

        for raw_line in task1_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 10:
                raise DetectorError(
                    "Each DOTA Task1 line must contain 10 fields: "
                    "imgname score x1 y1 x2 y2 x3 y3 x4 y4"
                )

            line_image_id = Path(parts[0]).stem
            if line_image_id != image_id:
                continue

            score = float(parts[1])
            if score < self.score_threshold:
                continue

            coords = [float(value) for value in parts[2:]]
            polygon_px = [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]],
            ]
            detections.append(
                {
                    "label": "harbor",
                    "score": score,
                    "polygon_px": polygon_px,
                }
            )

        return detections

    @staticmethod
    def _resolve_task1_file(output_path: Path) -> Path:
        if output_path.is_dir():
            return output_path / "Task1_harbor.txt"
        if output_path.name.lower() == "task1_harbor.txt":
            return output_path
        return output_path


def build_detector(
    detector_format: str,
    command_template: str | None,
    score_threshold: float,
):
    if detector_format == "json":
        return ExternalJsonDetector(command_template=command_template)
    if detector_format == "dota_task1_harbor":
        return DotATask1HarborDetector(
            command_template=command_template,
            score_threshold=score_threshold,
        )
    raise DetectorError(f"Unsupported detector format: {detector_format}")
