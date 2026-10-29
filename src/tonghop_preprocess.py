"""Tonghop 3.1: load the scene photo and preprocess it."""
import os

import cv2
import numpy as np

from utils import load_image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tonghop_scene")


def resize_max_dim(img, max_dim=640):
    """Shrink so the longer side is at most max_dim (keeps thresholds comparable across photos)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def load_scene(data_dir=DATA_DIR, max_dim=640):
    """First image in data_dir, else scikit-image's `camera` sample photo."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(exts)) if os.path.isdir(data_dir) else []
    if files:
        path = os.path.join(data_dir, files[0])
        print(f"[Tonghop] Using custom scene image: {path}")
        img = load_image(path)
    else:
        from skimage import data, img_as_ubyte
        print("[Tonghop] No scene photo found -- using the scikit-image 'camera' sample")
        img = np.stack([img_as_ubyte(data.camera())] * 3, axis=-1)
    return resize_max_dim(img, max_dim)


def normalize_lighting(img_rgb, clip_limit=2.0, tile_grid_size=8):
    """CLAHE on the Lab lightness channel, so hue and saturation are untouched."""
    l, a, b = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def denoise(img_rgb, d=7, sigma_color=50, sigma_space=50):
    """Bilateral filter: smooths flat areas, keeps edges."""
    return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)


def preprocess(img_rgb):
    normalized = normalize_lighting(img_rgb)
    denoised = denoise(normalized)
    return {"original": img_rgb, "normalized": normalized, "denoised": denoised,
            "gray": cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)}
