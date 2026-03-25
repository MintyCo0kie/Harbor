# Port Detection Pipeline

This project provides a minimal pipeline for:

1. Converting a place name to a center coordinate
2. Fetching a satellite image, or using a local image directly
3. Running an external DOTA-style detector
4. Converting detected polygon pixels into latitude/longitude
5. Running a tiled DOTA-style harbor workflow for larger scenes

## Current status

The pipeline now supports Google geocoding and Google Static Maps when an API key is configured. That means you can use:

- `--place` with built-in or local place lookup
- `--lat` and `--lng` with a known center point
- `--image-path` with a local image
- `--detector-command` with your own detector
- `--detector-format json` for normalized JSON detector output
- `--detector-format dota_task1_harbor` for DOTA Task1 harbor text output

This is enough to run:

- place name to center coordinate through local lookup
- local image
- detector output in pixel coordinates
- pixel polygon to lat/lng conversion

## Install

```bash
pip install -r requirements.txt
```

## Config

To use Google APIs, set either:

```bash
set MAPS_API_KEY=your_api_key
```

or:

```bash
set GOOGLE_MAPS_API_KEY=your_api_key
```

PowerShell:

```powershell
$env:MAPS_API_KEY="your_api_key"
```

or:

```powershell
$env:GOOGLE_MAPS_API_KEY="your_api_key"
```

Optional local place file:

```bash
set PLACES_FILE=data/places.json
```

PowerShell:

```powershell
$env:PLACES_FILE="data/places.json"
```

## Run

Using a place name:

```bash
python -m src.port_pipeline.cli --place "singapore" --zoom 17 --size 640x640 --image-path data/google_earth_export.png --detector-command "python tools/mock_detector.py --image {image} --output {output}" --output-dir outputs
```

Known coordinates:

```bash
python -m src.port_pipeline.cli --lat 31.2304 --lng 121.4737 --zoom 17 --size 640x640 --output-dir outputs
```

Known coordinates with a local image:

```bash
python -m src.port_pipeline.cli --lat 31.2304 --lng 121.4737 --zoom 17 --size 640x640 --image-path data/google_earth_export.png --output-dir outputs
```

With an external detector:

```bash
python -m src.port_pipeline.cli --lat 31.2304 --lng 121.4737 --zoom 17 --size 640x640 --image-path data/google_earth_export.png --detector-command "python tools/mock_detector.py --image {image} --output {output}" --output-dir outputs
```

With a DOTA Task1 harbor detector:

```bash
python -m src.port_pipeline.cli --place "singapore" --zoom 17 --size 640x640 --detector-format dota_task1_harbor --detector-command "python tools/mock_dota_detector.py --image {image} --image-id {image_id} --output {output}" --output-dir outputs_dota
```

## Tiled harbor workflow

For larger satellite scenes, use the tiled workflow:

```bash
python -m src.port_pipeline.tiled_cli --place "singapore" --size 640x640 --image-path sample.png --detector-format dota_task1_harbor --detector-command "python tools/mock_dota_detector.py --image {image} --image-id {image_id} --output {output}" --tile-size 256 --tile-overlap 64 --nms-iou-threshold 0.3 --boundary-method alpha_shape --boundary-alpha 0.25 --output-dir outputs_tiled
```

This workflow does:

- split the large scene into overlapping tiles
- run the detector on each tile
- keep only `harbor`, `ship`, `storage-tank`, `crane`
- map tile-local detections back to global image coordinates
- apply NMS to remove duplicates from overlapping tiles
- rebuild one outer harbor boundary from the detection points

If `shapely` is installed, `alpha_shape` tries a concave boundary first. Otherwise it falls back to a convex hull.

## Directional scan workflow (port sweep)

This workflow matches the step-by-step scan you described: start from a center point, scan left/up/down/right on a
fixed-resolution grid, stop after N steps without new harbor detections, then build the smallest boundary polygon
around all detected harbor structures.

Example with MMRotate (JSON output):

```bash
python -m src.port_pipeline.scan_cli --place "singapore" --size 640x640 --meters 700 --scale 1 --overlap-ratio 0.2 --max-no-new 2 --max-steps 50 --detector-format json --mmrotate-config mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py --mmrotate-checkpoint mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --mmrotate-device cuda:0 --mmrotate-score-thr 0.3 --output-dir outputs_scan
```

You can still pass a custom detector via `--detector-command` if you prefer:

```bash
python -m src.port_pipeline.scan_cli --place "singapore" --size 640x640 --meters 700 --detector-command "python tools/mock_detector.py --image {image} --output {output}" --output-dir outputs_scan
```

The scan workflow writes `result_scan.json`, plus per-tile images and detector outputs under `outputs_scan/`.

## Detector output format

The external detector is expected to write a JSON file like this:

```json
{
  "detections": [
    {
      "label": "harbor",
      "score": 0.93,
      "polygon_px": [[120, 220], [380, 210], [395, 420], [110, 430]]
    }
  ]
}
```

For DOTA Task1 harbor output, the parser expects the official oriented-box line format:

```text
imgname score x1 y1 x2 y2 x3 y3 x4 y4
```

Only lines whose `imgname` matches the current image stem are used.

## Google-backed run

Using Google geocoding plus Google satellite imagery:

```bash
python -m src.port_pipeline.cli --place "singapore" --zoom 17 --size 640x640 --detector-command "python tools/mock_detector.py --image {image} --output {output}" --output-dir outputs_google
```

## Local place registry

Built-in examples include:

- `singapore`
- `singapore port`
- `port of singapore`
- `port of shanghai`
- `port of rotterdam`

You can extend this by creating `data/places.json` with additional port names and coordinates.

## Output

The pipeline writes:

- `outputs/satellite.png`
- `outputs/result.json`

`result.json` contains:

- center lat/lng
- image metadata
- detection polygons in pixels
- detection polygons in lat/lng
