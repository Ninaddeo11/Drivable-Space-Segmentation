# gui_app.py
# Desktop GUI for real-time drivable space segmentation
# Run: py -3.12 gui_app.py

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageTk

from model import UNet

# ═══════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════
MODEL_PATH         = 'checkpoints/best_model.pth'
IMG_SIZE           = (256, 512)
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
DRIVABLE_COLOR     = (0, 255, 100)   # green (BGR)
NON_DRIVABLE_COLOR = (0, 0, 200)     # red   (BGR)
OVERLAY_ALPHA      = 0.5


# ═══════════════════════════════════════════════
#  MODEL LOADER
# ═══════════════════════════════════════════════
def load_model(path):
    model = UNet(in_channels=3, num_classes=2)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    miou = checkpoint.get('miou', 0)
    print(f"Model loaded — mIoU={miou:.4f} — device={DEVICE}")
    return model, miou


# ═══════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════
PREPROCESS = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

def predict_frame(model, frame_bgr):
    h_orig, w_orig = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    tensor    = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        mask   = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    overlay = frame_bgr.copy()
    overlay[mask_resized == 1] = DRIVABLE_COLOR
    overlay[mask_resized == 0] = NON_DRIVABLE_COLOR

    blended = cv2.addWeighted(frame_bgr, 1 - OVERLAY_ALPHA,
                              overlay,   OVERLAY_ALPHA, 0)

    drivable_pct = float((mask_resized == 1).mean() * 100)
    return blended, mask_resized, drivable_pct


# ═══════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════
class SegmentationApp:
    def __init__(self, root):
        self.root         = root
        self.root.title("Drivable Space Segmentation — MAHE Hackathon")
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(True, True)

        self.model        = None
        self.cap          = None
        self.running      = False
        self.show_overlay = tk.BooleanVar(value=True)
        self.fps_list     = []

        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root,
                 text="Drivable Space Segmentation",
                 font=('Helvetica', 18, 'bold'),
                 fg='#00d4aa', bg='#1a1a2e').pack(pady=(16, 2))

        tk.Label(self.root,
                 text="Real-time pixel-wise segmentation  |  CNN from scratch  |  MAHE Hackathon",
                 font=('Helvetica', 9), fg='#666688', bg='#1a1a2e').pack(pady=(0, 10))

        self.video_label = tk.Label(self.root, bg='#0d0d1a')
        self.video_label.pack(padx=16, pady=4)

        stats = tk.Frame(self.root, bg='#16213e')
        stats.pack(fill='x', padx=16, pady=4)

        self.fps_lbl    = tk.Label(stats, text="FPS: --",
                                   font=('Consolas', 11), fg='#00d4aa', bg='#16213e')
        self.drive_lbl  = tk.Label(stats, text="Drivable: --%",
                                   font=('Consolas', 11), fg='#00ff64', bg='#16213e')
        self.device_lbl = tk.Label(stats, text=f"Device: {DEVICE.upper()}",
                                   font=('Consolas', 11), fg='#aaaaaa', bg='#16213e')
        self.status_lbl = tk.Label(stats, text="No model loaded",
                                   font=('Consolas', 11), fg='#ff6b6b', bg='#16213e')

        self.fps_lbl.pack(side='left', padx=12, pady=6)
        self.drive_lbl.pack(side='left', padx=12)
        self.device_lbl.pack(side='left', padx=12)
        self.status_lbl.pack(side='right', padx=12)

        ctrl = tk.Frame(self.root, bg='#1a1a2e')
        ctrl.pack(pady=10)

        s = {'font': ('Helvetica', 11), 'relief': 'flat',
             'padx': 14, 'pady': 8, 'cursor': 'hand2'}

        self.load_btn = tk.Button(ctrl, text="Load Model",
                                  bg='#0f3460', fg='white',
                                  command=self.load_model_dialog, **s)
        self.load_btn.grid(row=0, column=0, padx=6)

        self.cam_btn = tk.Button(ctrl, text="Start Webcam",
                                 bg='#16213e', fg='white',
                                 command=self.toggle_webcam,
                                 state='disabled', **s)
        self.cam_btn.grid(row=0, column=1, padx=6)

        self.img_btn = tk.Button(ctrl, text="Load Image",
                                 bg='#16213e', fg='white',
                                 command=self.load_image,
                                 state='disabled', **s)
        self.img_btn.grid(row=0, column=2, padx=6)

        tk.Checkbutton(ctrl, text="Show overlay",
                       variable=self.show_overlay,
                       font=('Helvetica', 11), fg='white',
                       bg='#1a1a2e', activeforeground='white',
                       activebackground='#1a1a2e',
                       selectcolor='#0f3460').grid(row=0, column=3, padx=10)

        leg = tk.Frame(self.root, bg='#1a1a2e')
        leg.pack(pady=(0, 14))
        tk.Label(leg, text="  ", bg='#00ff64', width=2).pack(side='left', padx=4)
        tk.Label(leg, text="Drivable", fg='white',
                 bg='#1a1a2e', font=('Helvetica', 10)).pack(side='left', padx=(0, 14))
        tk.Label(leg, text="  ", bg='#c80000', width=2).pack(side='left', padx=4)
        tk.Label(leg, text="Non-Drivable", fg='white',
                 bg='#1a1a2e', font=('Helvetica', 10)).pack(side='left')

    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
            initialdir="checkpoints"
        )
        if not path:
            return
        try:
            self.model, miou = load_model(path)
            self.status_lbl.config(
                text=f"Model loaded  mIoU={miou:.4f}", fg='#00d4aa')
            self.cam_btn.config(state='normal')
            self.img_btn.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")

    def toggle_webcam(self):
        if self.running:
            self.running = False
            self.cam_btn.config(text="Start Webcam")
            if self.cap:
                self.cap.release()
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            self.running = True
            self.cam_btn.config(text="Stop Webcam")
            threading.Thread(target=self._webcam_loop, daemon=True).start()

    def _webcam_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            t0 = time.time()

            if self.show_overlay.get() and self.model:
                display, _, dpct = predict_frame(self.model, frame)
            else:
                display = frame
                dpct    = 0.0

            elapsed = time.time() - t0
            self.fps_list.append(1.0 / max(elapsed, 1e-6))
            if len(self.fps_list) > 30:
                self.fps_list.pop(0)
            fps = np.mean(self.fps_list)

            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(display_rgb)
            pil_img.thumbnail((820, 520))
            photo = ImageTk.PhotoImage(pil_img)

            self.video_label.after(0, self._update_frame, photo, fps, dpct)

        self.running = False

    def _update_frame(self, photo, fps, dpct):
        self.video_label.config(image=photo)
        self.video_label.image = photo
        self.fps_lbl.config(text=f"FPS: {fps:.1f}")
        self.drive_lbl.config(text=f"Drivable: {dpct:.1f}%")

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
            initialdir="."
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        if self.show_overlay.get() and self.model:
            display, _, dpct = predict_frame(self.model, frame)
            self.drive_lbl.config(text=f"Drivable: {dpct:.1f}%")
        else:
            display = frame

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_rgb)
        pil_img.thumbnail((820, 520))
        photo = ImageTk.PhotoImage(pil_img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app  = SegmentationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
