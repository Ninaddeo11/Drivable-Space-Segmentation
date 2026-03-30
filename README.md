# Drivable Space Segmentation — MAHE Hackathon

Real-time pixel-wise segmentation of drivable vs non-drivable areas using a
**custom U-Net CNN trained from scratch** on the nuScenes dataset.

---

## Project structure

```
project/
├── model.py          ← U-Net CNN (built from scratch, no pretrained weights)
├── dataset.py        ← nuScenes data loader + binary mask generator
├── train.py          ← Full training pipeline
├── gui_app.py        ← Desktop GUI with webcam feed
├── server.py         ← Flask backend (powers the website)
├── index.html        ← Website frontend
├── requirements.txt
├── data/
│   ├── images/       ← Put your nuScenes .jpg images here
│   └── masks/        ← Put your nuScenes annotation .png masks here
└── checkpoints/      ← Best model saved here automatically
```

---

## Step 1 — Install Python and dependencies

Make sure you have Python 3.9+ installed.

Open a terminal (Command Prompt on Windows) and run:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU support (MUCH faster training), install PyTorch with CUDA:
Visit https://pytorch.org/get-started/locally/ and select your CUDA version.
Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2 — Prepare the nuScenes dataset

1. Download nuScenes from https://www.nuscenes.org/nuscenes (free academic license)
2. Extract images and their semantic segmentation masks
3. Place images in `data/images/` and masks in `data/masks/`
   - Image files: `data/images/abc123.jpg`
   - Mask files:  `data/masks/abc123.png`  (same filename, grayscale with class IDs)

The drivable class ID in nuScenes is **24** (driveable_surface).
You can adjust this in `dataset.py` → `DRIVABLE_CLASS_IDS` set.

---

## Step 3 — Verify your setup

```bash
python model.py
```
You should see:
```
Input  shape: torch.Size([2, 3, 256, 512])
Output shape: torch.Size([2, 2, 256, 512])
Total parameters: 7,xxx,xxx
```

```bash
python dataset.py
```
You should see:
```
[Dataset] train: NNN samples
Image shape : torch.Size([3, 256, 512])
Mask  shape : torch.Size([256, 512])
Mask values : tensor([0, 1])
```

---

## Step 4 — Train the model

```bash
python train.py
```

Training runs for 100 epochs and prints progress like:
```
Epoch  1/100 | Train Loss: 0.6234 | Val Loss: 0.5891 | Val mIoU: 0.4123
  ✓ New best model saved (mIoU=0.4123)
Epoch  2/100 | ...
```

The best model is automatically saved to `checkpoints/best_model.pth`.

Training time: ~2–4 hours on a mid-range NVIDIA GPU for 100 epochs.

---

## Step 5 — Run the desktop GUI

```bash
python gui_app.py
```

- Click "Load Model" → select `checkpoints/best_model.pth`
- Click "Start Webcam" to see live segmentation
- Or click "Load Image" to segment a single photo
- Green = drivable, Red = non-drivable

---

## Step 6 — Run the website

Open TWO terminals:

**Terminal 1 — start the backend server:**
```bash
python server.py
```
You should see:
```
[Server] Model loaded on CUDA
[Server] Running on http://localhost:5000
```

**Terminal 2 — open the website:**
Simply open `index.html` in your browser (double-click the file).

OR serve it with Python:
```bash
python -m http.server 8080
```
Then open http://localhost:8080 in your browser.

---

## Model architecture summary

```
Input (3, 256, 512)
  └── Encoder block 1 → 32 channels
  └── Encoder block 2 → 64 channels
  └── Encoder block 3 → 128 channels
  └── Encoder block 4 → 256 channels
       └── Bottleneck  → 512 channels
       └── Decoder block 4 + skip → 256 channels
       └── Decoder block 3 + skip → 128 channels
       └── Decoder block 2 + skip → 64 channels
       └── Decoder block 1 + skip → 32 channels
            └── Final conv (1×1) → 2 channels
Output (2, 256, 512) — per-pixel class logits
```

Each encoder/decoder block uses: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU

---

## Evaluation metrics

| Metric    | Description                                         |
|-----------|-----------------------------------------------------|
| mIoU      | Mean Intersection over Union (primary metric)       |
| FPS       | Frames per second at inference (speed)              |
| Precision | Of predicted drivable pixels, how many are correct  |
| Recall    | Of actual drivable pixels, how many were found      |
| F1        | Harmonic mean of precision and recall               |

---

## Troubleshooting

**CUDA out of memory**: Reduce `batch_size` in `train.py` CONFIG to 4 or 2.

**Webcam not found**: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `gui_app.py`.

**Website cannot reach server**: Make sure `python server.py` is running before opening `index.html`.

**Low mIoU**: Try training for more epochs, or check that your mask class IDs match `DRIVABLE_CLASS_IDS` in `dataset.py`.
