"""Part 1: grayscale conversion and RGB channel operations."""
import os

import cv2
import numpy as np

from utils import load_sample_images, to_gray3, save_image, save_grid, ensure_dir

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part1")


def rgb_to_gray_manual(img_rgb):
    """Apply BT.601 luminance weights to RGB channels."""
    r, g, b = img_rgb[..., 0].astype(np.float32), img_rgb[..., 1].astype(np.float32), img_rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def gray_to_rgb_replicated(gray):
    """Repeat gray values; lost color information cannot be recovered."""
    return to_gray3(gray)


def split_channels(img_rgb):
    r = img_rgb[..., 0]
    g = img_rgb[..., 1]
    b = img_rgb[..., 2]
    return r, g, b


def reconstruct_from_channels(r, g, b):
    return np.stack([r, g, b], axis=-1)


def make_swapped_variants(img_rgb):
    r, g, b = split_channels(img_rgb)
    zero = np.zeros_like(r)
    return {
        "R-G swapped (G,R,B)": np.stack([g, r, b], axis=-1),
        "R-B swapped (B,G,R)": np.stack([b, g, r], axis=-1),
        "Red channel only": np.stack([r, zero, zero], axis=-1),
        "Green channel only": np.stack([zero, g, zero], axis=-1),
        "Blue channel only": np.stack([zero, zero, b], axis=-1),
        "Missing Blue (R,G,0)": np.stack([r, g, zero], axis=-1),
    }


def run():
    ensure_dir(OUT_DIR)
    images = load_sample_images()
    img = images["astronaut"]

    gray_manual = rgb_to_gray_manual(img)
    gray_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    max_diff = int(np.max(np.abs(gray_manual.astype(int) - gray_cv2.astype(int))))
    print(f"[Part1] Max |manual_gray - cv2_gray| = {max_diff} (expect <= 1, rounding only)")

    gray_back_to_rgb = gray_to_rgb_replicated(gray_manual)

    save_grid(
        os.path.join(OUT_DIR, "01_rgb_gray_roundtrip.png"),
        [img, to_gray3(gray_manual), gray_back_to_rgb],
        ["Original RGB", "Grayscale (manual formula)", "Gray replicated back to RGB\n(color info NOT recovered)"],
        suptitle="Part 1.1 - RGB <-> Grayscale",
    )

    r, g, b = split_channels(img)
    save_grid(
        os.path.join(OUT_DIR, "02_channel_split.png"),
        [img, to_gray3(r), to_gray3(g), to_gray3(b)],
        ["Original RGB", "R channel (as gray)", "G channel (as gray)", "B channel (as gray)"],
        suptitle="Part 1.2 - Splitting Color Channels",
        ncols=4,
    )

    reconstructed = reconstruct_from_channels(r, g, b)
    assert np.array_equal(reconstructed, img), "Reconstruction from R,G,B must be lossless"
    save_grid(
        os.path.join(OUT_DIR, "03_reconstruction.png"),
        [img, reconstructed],
        ["Original", "Reconstructed from R+G+B\n(pixel-identical)"],
        suptitle="Part 1.3 - Lossless Reconstruction",
    )

    variants = make_swapped_variants(img)
    save_grid(
        os.path.join(OUT_DIR, "04_channel_variants.png"),
        [img] + list(variants.values()),
        ["Original"] + list(variants.keys()),
        suptitle="Part 1.4 - New Images via Channel Swap / Combination",
        ncols=4,
    )

    save_image(os.path.join(OUT_DIR, "gray_manual.png"), gray_manual)
    print(f"[Part1] Figures written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    run()
