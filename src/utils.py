"""Image loading and output helpers. Color arrays use RGB order."""
import os

import cv2
import numpy as np
from skimage import data, img_as_ubyte


def load_sample_images():
    """Load the four scikit-image examples used in this project."""
    astronaut = img_as_ubyte(data.astronaut())
    checkerboard = img_as_ubyte(data.checkerboard())
    coins = img_as_ubyte(data.coins())
    camera = img_as_ubyte(data.camera())
    return {
        "astronaut": astronaut,
        "checkerboard": checkerboard,
        "coins": coins,
        "camera": camera,
    }


def to_gray3(gray):
    """Repeat grayscale values across RGB channels for display."""
    return np.stack([gray, gray, gray], axis=-1)


def add_gaussian_noise(img, sigma=25, seed=0):
    """Add reproducible Gaussian noise and keep values in uint8 range."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_image(path, img):
    """Convert RGB to BGR only when writing with OpenCV."""
    ensure_dir(os.path.dirname(path))
    if img.ndim == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)


def save_grid(path, images, titles, cmap=None, suptitle=None, ncols=None):
    """Save a labeled image grid as PNG."""
    import matplotlib.pyplot as plt

    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        c = (cmap or "gray") if img.ndim == 2 else None
        ax.imshow(img, cmap=c, vmin=0 if img.ndim == 2 else None, vmax=255 if img.ndim == 2 else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
