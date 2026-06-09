# Building Count and Building Coverage Estimation

Minimal MVP for binary building footprint segmentation from satellite imagery.

## Features

- Model implemented: U-Net
- Dataset trained on: WHU Building Dataset
- Output tasks: building count and building coverage estimation

Pipeline:

```text
Satellite image -> U-Net -> Building mask -> Building count -> Building coverage
```

## Colab Setup

Install dependencies:

```python
!pip install segmentation-models-pytorch opencv-python matplotlib
```

For local use, install the same dependencies from the project folder:

```bash
pip install -r requirements.txt
```

Optional Google Drive mount:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set your dataset paths:

```python
IMAGE_DIR = "/content/path/to/images"
MASK_DIR = "/content/path/to/masks"
CHECKPOINT_PATH = "/content/best_unet_buildings.pth"
```

Images and masks must be in separate folders and share filename stems, for example:

```text
images/tile_001.tif
masks/tile_001.png
```

## Train

```bash
python train.py \
  --image-dir /content/path/to/images \
  --mask-dir /content/path/to/masks \
  --checkpoint-path /content/best_unet_buildings.pth \
  --epochs 10 \
  --batch-size 4
```

For a quick smoke test:

```bash
python train.py \
  --image-dir /content/path/to/images \
  --mask-dir /content/path/to/masks \
  --checkpoint-path /content/best_unet_buildings.pth \
  --epochs 1 \
  --max-samples 16
```

## Evaluate

```bash
python evaluate.py \
  --image-dir /content/path/to/images \
  --mask-dir /content/path/to/masks \
  --checkpoint-path /content/best_unet_buildings.pth \
  --show-examples
```

Evaluation prints validation loss, IoU, Dice, mean building count, and mean building coverage percentage.

## Verify Dataset Before Training

For the WHU Building Dataset, run the verification script before training:

```bash
python verify_dataset.py \
  --dataset-root "/path/to/Satellite dataset I (global cities)" \
  --sample-count 10 \
  --min-area 20
```

If auto-detection does not find the folders, pass them explicitly:

```bash
python verify_dataset.py \
  --dataset-root "/path/to/dataset" \
  --image-dir "/path/to/dataset/image" \
  --mask-dir "/path/to/dataset/label"
```

The verifier prints the folder tree, image/mask counts, matched pairs, missing files, dimension statistics, mask binary checks, coverage, connected-component building counts, visual overlays, and a final `Dataset Ready = True/False` summary.

## Notes

- Masks are expected to be raster building masks, not GeoJSON polygons.
- The model is a single binary U-Net from `segmentation-models-pytorch`.
- Connected component analysis is used only on predicted binary masks.
- No road, topology, clustering, reliability, or multi-model analysis is included.
