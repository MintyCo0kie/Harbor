# Harbor Project README

This README only includes environment setup, required Python version, API key setup, and the run command.

## 1. Python Version

Use Python 3.8 (for example, 3.8.20).

## 2. Environment Setup

Create and activate a Conda environment first, then install dependencies with pip:

```bash
conda create -n harbor python=3.8 -y
conda activate harbor
cd harbor

#windows
pip install -r requirements.txt

#Mac

# macOS pip requirements for this project
# Usage:
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-mac.txt



# mmcv-full may not have a ready-made wheel on some macOS combinations.
# Try this first:
mim install "mmcv-full==1.7.2"
# If it fails, fallback (limited ops support):
mim install  "mmcv==1.7.2"


```

## 3. Detection Model download
```bash
mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest ./mmrotate
```

## 3. API Key Setup (.env)

Create a `.env` file in the project root and add one of the following keys:

```env
MAPS_API_KEY=your_api_key
```

or

```env
GOOGLE_MAPS_API_KEY=your_api_key
```

## 4. Run Command

```bash
python -m src.port_pipeline.scan_cli --place "singapore" --size 960x960 --meters 2000 --detector-format json --mmrotate-config mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py --mmrotate-checkpoint mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --output-dir outputs_scan_2000 --verbose --max-no-new 4 --step-sleep-seconds 0.2
```
