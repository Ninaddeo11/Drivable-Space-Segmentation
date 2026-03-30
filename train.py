# train.py
# Complete training pipeline for drivable space segmentation
# Run: py -3.12 train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model   import UNet
from dataset import DrivableDataset


# ═══════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════
CONFIG = {
    'data_dir'     : 'data',
    'img_size'     : (256, 512),
    'batch_size'   : 8,
    'num_epochs'   : 100,
    'learning_rate': 1e-3,
    'num_classes'  : 2,
    'save_path'    : 'checkpoints/best_model.pth',
    'device'       : 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ═══════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = torch.zeros_like(preds)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union        = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice        = DiceLoss()
        self.ce          = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        return self.dice_weight * self.dice(preds, targets) + \
               (1 - self.dice_weight) * self.ce(preds, targets)


# ═══════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════

def compute_miou(preds, targets, num_classes=2):
    pred_classes = preds.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls   = (pred_classes == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).float().sum()
        union        = (pred_cls | target_cls).float().sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    return np.mean(ious)


# ═══════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════

def train():
    device = CONFIG['device']
    print(f"Using device: {device}")

    # ── Data ──
    train_dataset = DrivableDataset(
        CONFIG['data_dir'], split='train',
        img_size=CONFIG['img_size'], augment=True
    )
    val_dataset = DrivableDataset(
        CONFIG['data_dir'], split='val',
        img_size=CONFIG['img_size'], augment=False
    )

    # num_workers=0 fixes Windows multiprocessing crash
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=0, pin_memory=True
    )

    # ── Model ──
    model = UNet(in_channels=3, num_classes=CONFIG['num_classes']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser and loss ──
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=CONFIG['num_epochs'])
    criterion = CombinedLoss(dice_weight=0.5)

    # ── Mixed precision (updated syntax, no FutureWarning) ──
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    os.makedirs('checkpoints', exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, CONFIG['num_epochs'] + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                preds = model(images)
                loss  = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                preds  = model(images)
                loss   = criterion(preds, masks)
                val_loss += loss.item()
                val_miou += compute_miou(preds, masks, CONFIG['num_classes'])

        avg_val_loss = val_loss / len(val_loader)
        avg_val_miou = val_miou / len(val_loader)

        scheduler.step()

        print(f"\nEpoch {epoch:3d}/{CONFIG['num_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val mIoU: {avg_val_miou:.4f}")

        # ── Save best model ──
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'miou'       : best_miou,
                'config'     : CONFIG,
            }, CONFIG['save_path'])
            print(f"  ✓ New best model saved (mIoU={best_miou:.4f})")

    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    print(f"Model saved to: {CONFIG['save_path']}")


if __name__ == '__main__':
    train()
