# dataset.py — IMPROVED VERSION
# Stronger data augmentation for better generalisation

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import random


class DrivableDataset(Dataset):
    """
    Loads images + binary masks (0=non-drivable, 255=drivable)
    with strong augmentation for better model generalisation.
    """

    def __init__(self, data_dir, split='train', img_size=(256, 512), augment=True):
        self.img_dir  = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.img_size = img_size
        self.augment  = augment and (split == 'train')

        all_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

        split_idx = int(len(all_files) * 0.8)
        if split == 'train':
            self.filenames = all_files[:split_idx]
        else:
            self.filenames = all_files[split_idx:]

        print(f"[Dataset] {split}: {len(self.filenames)} samples")

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.filenames)

    def _to_binary_mask(self, mask_array):
        """Convert 0/255 mask to binary 0/1."""
        return (mask_array > 127).astype(np.int64)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        stem  = os.path.splitext(fname)[0]

        img_path  = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, stem + '.png')

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        # Resize
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        mask  = mask.resize( (self.img_size[1], self.img_size[0]), Image.NEAREST)

        if self.augment:
            # ── 1. Random horizontal flip ──
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # ── 2. Stronger colour jitter ──
            image = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.1
            )(image)

            # ── 3. Random crop and resize ──
            if random.random() > 0.4:
                crop_h = int(self.img_size[0] * random.uniform(0.75, 0.95))
                crop_w = int(self.img_size[1] * random.uniform(0.75, 0.95))
                i, j, h, w = T.RandomCrop.get_params(
                    image, output_size=(crop_h, crop_w)
                )
                image = TF.crop(image, i, j, h, w)
                mask  = TF.crop(mask,  i, j, h, w)
                image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
                mask  = mask.resize( (self.img_size[1], self.img_size[0]), Image.NEAREST)

            # ── 4. Random Gaussian blur ──
            if random.random() > 0.6:
                image = image.filter(__import__('PIL.ImageFilter',
                         fromlist=['GaussianBlur']).GaussianBlur(
                         radius=random.uniform(0.5, 1.5)))

            # ── 5. Random grayscale (simulates night/bad weather) ──
            if random.random() > 0.85:
                image = TF.to_grayscale(image, num_output_channels=3)

            # ── 6. Random vertical shift (simulates camera pitch) ──
            if random.random() > 0.7:
                shift = random.randint(-20, 20)
                image = TF.affine(image, angle=0, translate=(0, shift),
                                  scale=1.0, shear=0)
                mask  = TF.affine(mask,  angle=0, translate=(0, shift),
                                  scale=1.0, shear=0)

        # Convert to tensors
        image   = TF.to_tensor(image)
        image   = self.normalize(image)

        mask_np = np.array(mask, dtype=np.int64)
        mask_np = self._to_binary_mask(mask_np)
        mask_t  = torch.from_numpy(mask_np)

        return image, mask_t


if __name__ == '__main__':
    ds = DrivableDataset('data', split='train')
    img, mask = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Mask  shape : {mask.shape}")
    print(f"Mask values : {mask.unique()}")
