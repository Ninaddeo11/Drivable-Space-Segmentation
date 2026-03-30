# server.py
# Flask backend — serves the trained model via a REST API
# The website (index.html) sends images here and gets back segmented results
# Run: python server.py

import io
import base64
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import UNet


# ═══════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════
MODEL_PATH = 'checkpoints/best_model.pth'
IMG_SIZE   = (256, 512)
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Overlay colours (RGB)
DRIVABLE_COLOR     = np.array([0, 220, 90],  dtype=np.uint8)
NON_DRIVABLE_COLOR = np.array([220, 30, 30], dtype=np.uint8)
OVERLAY_ALPHA      = 0.45

app = Flask(__name__)
CORS(app)  # allow requests from the website running on a different port

model = None  # loaded on startup


# ═══════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════
def load_model():
    global model
    m = UNet(in_channels=3, num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(checkpoint['model_state'])
    m.to(DEVICE)
    m.eval()
    print(f"[Server] Model loaded on {DEVICE.upper()}")
    model = m


PREPROCESS = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_inference(pil_image):
    """
    pil_image : PIL.Image (RGB)
    Returns   : base64-encoded PNG of the segmented overlay
    """
    w_orig, h_orig = pil_image.size

    tensor = PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        mask   = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize mask back to original
    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Build overlay
    img_np  = np.array(pil_image)
    overlay = img_np.copy()
    overlay[mask_resized == 1] = DRIVABLE_COLOR
    overlay[mask_resized == 0] = NON_DRIVABLE_COLOR

    blended = (img_np * (1 - OVERLAY_ALPHA) + overlay * OVERLAY_ALPHA).astype(np.uint8)

    # Encode to base64 PNG
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Drivable percentage
    drivable_pct = float((mask_resized == 1).mean() * 100)

    return b64, drivable_pct


# ═══════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Simple ping endpoint so the website can check if the server is alive."""
    return jsonify({'status': 'ok', 'device': DEVICE, 'model_loaded': model is not None})


@app.route('/segment', methods=['POST'])
def segment():
    """
    Accepts a base64-encoded image in JSON body:
        { "image": "data:image/jpeg;base64,/9j/4AAQ..." }
    Returns:
        { "result": "data:image/png;base64,...", "drivable_pct": 62.3 }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Strip data URL prefix if present
        img_str = data['image']
        if ',' in img_str:
            img_str = img_str.split(',', 1)[1]

        img_bytes = base64.b64decode(img_str)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        result_b64, drivable_pct = run_inference(pil_image)

        return jsonify({
            'result'      : f'data:image/png;base64,{result_b64}',
            'drivable_pct': round(drivable_pct, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    load_model()
    print("[Server] Running on http://localhost:5000")
    print("[Server] Open index.html in your browser to use the web UI")
    app.run(host='0.0.0.0', port=5000, debug=False)
