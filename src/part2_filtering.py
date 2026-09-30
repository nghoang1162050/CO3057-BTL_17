"""Part 2: low-pass smoothing and high-pass edge filters."""
import os

import cv2
import numpy as np

from utils import load_sample_images, add_gaussian_noise, save_grid, ensure_dir

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part2")

def mean_filter(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(img, -1, kernel)


def gaussian_filter(img, ksize=5, sigma=1.5):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def laplacian_filter(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return cv2.convertScaleAbs(lap)


def sobel_filter(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag = np.clip(mag / mag.max() * 255, 0, 255) if mag.max() > 0 else mag
    return mag.astype(np.uint8)


def custom_sharpen_filter(img):
    """Use a sharpen kernel that boosts the center against its neighbors."""
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def noise_std(img):
    """Measure the residual after blurring; this also includes real edges."""
    gray = to_gray(img).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    return float(np.std(gray - blurred))


def run():
    ensure_dir(OUT_DIR)
    images = load_sample_images()

    test_cases = {
        "checkerboard (sharp, low-detail)": images["checkerboard"],
        "coins (textured, grayscale)": images["coins"],
        "astronaut+noise (color, noisy)": add_gaussian_noise(images["astronaut"], sigma=25),
    }

    for name, img in test_cases.items():
        slug = name.split(" ")[0]
        gray = to_gray(img)

        mean5 = mean_filter(img, 5)
        gauss5 = gaussian_filter(img, 5, 1.5)
        lap = laplacian_filter(gray)
        sob = sobel_filter(gray)
        sharp = custom_sharpen_filter(img)

        def as3(x):
            return x if x.ndim == 3 else np.stack([x] * 3, axis=-1)

        save_grid(
            os.path.join(OUT_DIR, f"{slug}_lowpass.png"),
            [img, as3(mean5), as3(gauss5)],
            ["Original", "Mean filter 5x5\n(low-pass)", "Gaussian filter 5x5, sigma=1.5\n(low-pass)"],
            suptitle=f"Part 2 - Low-pass filtering: {name}",
        )
        save_grid(
            os.path.join(OUT_DIR, f"{slug}_highpass.png"),
            [img, as3(lap), as3(sob), as3(sharp)],
            ["Original", "Laplacian\n(high-pass, edges)", "Sobel magnitude\n(high-pass, gradients)", "Custom sharpen kernel\n(high-pass, detail boost)"],
            suptitle=f"Part 2 - High-pass filtering: {name}",
            ncols=4,
        )

        print(f"[Part2] {name}: noise proxy(std of residual) "
              f"original={noise_std(img):.2f}  after mean={noise_std(mean5):.2f}  "
              f"after gaussian={noise_std(gauss5):.2f}")

    print(f"[Part2] Figures written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    run()
