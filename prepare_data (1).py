# prepare_data.py — IMPROVED VERSION
# Better drivable mask generation using multiple techniques combined
# Run: py -3.12 prepare_data.py

import os
import json
import shutil
import numpy as np
from PIL import Image
import cv2

NUSCENES_ROOT = r"C:\Users\Ninad\Desktop\v1.0-mini"
VERSION       = "v1.0-mini"
OUTPUT_IMAGES = "data/images"
OUTPUT_MASKS  = "data/masks"

def load_json(filename):
    path = os.path.join(NUSCENES_ROOT, VERSION, filename)
    with open(path, 'r') as f:
        return json.load(f)

print("Loading nuScenes JSON files...")
samples     = load_json("sample.json")
sample_data = load_json("sample_data.json")

sample_data_by_sample = {}
for sd in sample_data:
    sid = sd["sample_token"]
    sample_data_by_sample.setdefault(sid, []).append(sd)

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_MASKS,  exist_ok=True)
print(f"Found {len(samples)} samples. Processing CAM_FRONT...")


def generate_improved_mask(image_path):
    """
    Improved mask generation using 4 combined techniques:
    1. HSV colour detection for asphalt/road colours
    2. LAB colour space for better colour separation
    3. Adaptive trapezoid ROI based on image content
    4. Morphological refinement for clean edges
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # ── Technique 1: HSV road colour detection ──
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Roads: low saturation, mid-high value (grey/asphalt)
    mask_hsv = cv2.inRange(hsv,
                           np.array([0,   0,  40]),
                           np.array([180, 55, 210]))

    # ── Technique 2: LAB colour space ──
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Roads are neutral colour — low a and b channel variation
    mask_lab = cv2.inRange(lab,
                           np.array([40,  110, 110]),
                           np.array([220, 145, 145]))

    # ── Combine colour masks ──
    color_combined = cv2.bitwise_or(mask_hsv, mask_lab)

    # ── Technique 3: Adaptive trapezoid ROI ──
    # Wider at bottom, narrower at horizon (~45% height)
    roi = np.zeros((h, w), dtype=np.uint8)
    horizon = int(h * 0.45)   # horizon line
    trap = np.array([[
        (int(w * 0.02), h),              # bottom-left
        (int(w * 0.98), h),              # bottom-right
        (int(w * 0.60), horizon),        # top-right
        (int(w * 0.40), horizon),        # top-left
    ]], dtype=np.int32)
    cv2.fillPoly(roi, trap, 255)

    # Apply ROI
    combined = cv2.bitwise_and(color_combined, roi)

    # ── Technique 4: Morphological refinement ──
    # Close small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # Expand to fill road
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    combined = cv2.dilate(combined, kernel_dilate, iterations=2)

    # Re-apply ROI after dilation
    combined = cv2.bitwise_and(combined, roi)

    # ── Fill largest connected component only ──
    # This removes stray detections outside the main road
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8)

    if num_labels > 1:
        # Find largest component (excluding background=0)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(combined)
        clean[labels == largest] = 255

        # Also keep components touching bottom edge (always road)
        for i in range(1, num_labels):
            comp_mask = (labels == i).astype(np.uint8) * 255
            if comp_mask[h-5:h, :].any():
                clean = cv2.bitwise_or(clean, comp_mask)

        combined = clean

    # Final ROI clamp
    combined = cv2.bitwise_and(combined, roi)

    return Image.fromarray(combined)


processed = 0
skipped   = 0

for sample in samples:
    cam = None
    for sd in sample_data_by_sample.get(sample["token"], []):
        if "CAM_FRONT" in sd.get("filename", "") and sd.get("is_key_frame", False):
            cam = sd
            break
    if cam is None:
        skipped += 1
        continue

    src = os.path.join(NUSCENES_ROOT, cam["filename"])
    if not os.path.exists(src):
        skipped += 1
        continue

    out_img  = os.path.join(OUTPUT_IMAGES, cam["token"] + ".jpg")
    out_mask = os.path.join(OUTPUT_MASKS,  cam["token"] + ".png")

    shutil.copy2(src, out_img)
    mask = generate_improved_mask(src)
    if mask:
        mask.save(out_mask)
        processed += 1

    if processed % 20 == 0 and processed > 0:
        print(f"  {processed} done...")

print(f"\nDone! {processed} images, {skipped} skipped.")

# ── Save verification image ──
img_files = sorted(os.listdir(OUTPUT_IMAGES))[:1]
if img_files:
    si = os.path.join(OUTPUT_IMAGES, img_files[0])
    sm = os.path.join(OUTPUT_MASKS,  img_files[0].replace(".jpg", ".png"))
    if os.path.exists(sm):
        ia = np.array(Image.open(si).convert("RGB"))
        ma = np.array(Image.open(sm).convert("L").resize(
             (ia.shape[1], ia.shape[0]), Image.NEAREST))
        ov = ia.copy()
        ov[ma > 127]  = (ov[ma > 127]  * 0.5 + np.array([0,   200, 80])  * 0.5).astype(np.uint8)
        ov[ma <= 127] = (ov[ma <= 127] * 0.5 + np.array([200, 30,  30])  * 0.5).astype(np.uint8)
        Image.fromarray(ov).save("data/sample_verification.jpg")
        print("Verification saved → data/sample_verification.jpg")
        print("Open it to check mask quality before training!")

print("\n" + "="*50)
print("Next step: py -3.12 train.py")
print("="*50)
