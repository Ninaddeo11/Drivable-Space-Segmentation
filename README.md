# Drivable Space Segmentation — MAHE Mobility Hackathon

Real-time pixel-wise segmentation of drivable vs non-drivable areas using a custom U-Net CNN trained completely from scratch on the nuScenes dataset.

## Results

| Metric | Value |
|--------|-------|
| Model | U-Net CNN (from scratch) |
| Parameters | 7,763,074 |
| Best Epoch | 82 / 100 |
| mIoU | **0.9489** |
| Inference Speed | 43+ FPS |
| Device | NVIDIA RTX 4060 (CUDA) |
| Dataset | nuScenes v1.0-mini |

## Problem Statement

Level 4 autonomous vehicles must identify "Free Space" — areas where the car can physically move — regardless of whether lane markings exist. This project performs pixel-wise semantic segmentation classifying every pixel as either **Drivable** (green) or **Non-Drivable** (red) in complex urban environments including edge cases like road-to-grass transitions, water puddles, and construction barriers.

## Project Structure

```
├── model.py          <- U-Net CNN built from scratch (no pretrained weights)
├── dataset.py        <- nuScenes data loader + binary mask generator
├── train.py          <- Full training pipeline (Dice + CE loss, AdamW, cosine LR)
├── prepare_data.py   <- Converts nuScenes v1.0-mini into training-ready data
├── gui_app.py        <- Desktop GUI with live webcam segmentation
├── server.py         <- Flask backend API
├── index.html        <- Website frontend
└── requirements.txt
```

## Model Architecture

Custom U-Net with encoder-decoder structure and skip connections:

```
Input (3, 256, 512) — RGB image
  Encoder Block 1 -> 32 channels  + MaxPool
  Encoder Block 2 -> 64 channels  + MaxPool
  Encoder Block 3 -> 128 channels + MaxPool
  Encoder Block 4 -> 256 channels + MaxPool
    Bottleneck    -> 512 channels
  Decoder Block 4 + skip -> 256 channels
  Decoder Block 3 + skip -> 128 channels
  Decoder Block 2 + skip -> 64 channels
  Decoder Block 1 + skip -> 32 channels
  Final Conv (1x1) -> 2 channels
Output (2, 256, 512) — per-pixel class logits
```

Each encoder/decoder block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU

**No pretrained weights used. Trained from scratch as required.**

## Setup and Installation

### Requirements
- Python 3.12
- NVIDIA GPU with CUDA 12.1+

### Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python Pillow flask flask-cors numpy
```

### Prepare nuScenes data
1. Download nuScenes v1.0-mini from https://www.nuscenes.org
2. Update NUSCENES_ROOT path in prepare_data.py
3. Run:
```bash
py -3.12 prepare_data.py
```

### Train the model
```bash
py -3.12 train.py
```

Training saves the best checkpoint automatically to checkpoints/best_model.pth.

### Run desktop GUI
```bash
py -3.12 gui_app.py
```
- Click Load Model and select checkpoints/best_model.pth
- Click Start Webcam for live segmentation
- Or click Load Image to test on a photo

### Run website
Terminal 1:
```bash
py -3.12 server.py
```
Terminal 2:
```bash
py -3.12 -m http.server 8080
```
Open http://localhost:8080 in your browser.

## Training Details

- **Loss function:** Dice Loss + Cross Entropy Loss (50/50 weighted)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing LR
- **Mixed precision:** FP16 training via torch.amp
- **Batch size:** 8
- **Epochs:** 100
- **Input resolution:** 256x512

## Data Augmentation

- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random crop and resize
- Gaussian blur
- Grayscale simulation for weather/night conditions
- Random vertical shift for camera pitch simulation

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| mIoU | Mean Intersection over Union — primary metric |
| FPS | Inference speed — real-time performance |
| Precision | Correctly predicted drivable pixels |
| Recall | Drivable pixels correctly found |
| F1 Score | Harmonic mean of precision and recall |

## Team

MAHE Mobility Hackathon — Problem Statement 2: Real-time Drivable Space Segmentation