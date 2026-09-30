"""Part 3: basic, affine, and projective image transforms."""
import os

import cv2
import numpy as np

from utils import load_sample_images, to_gray3, save_grid, ensure_dir

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part3")


def translate(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def rotate(img, angle_deg, scale=1.0):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, fx, fy):
    h, w = img.shape[:2]
    M = np.float32([[fx, 0, 0], [0, fy, 0]])
    return cv2.warpAffine(img, M, (int(w * fx), int(h * fy)))


def affine_transform(img):
    """Use three point pairs to define an affine map."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[w * 0.0, h * 0.15], [w * 0.9, h * 0.05], [w * 0.15, h * 0.95]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (w, h)), M


def projective_transform(img):
    """Use four point pairs to make a perspective warp."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([[w * 0.28, h * 0.05], [w * 0.72, h * 0.05],
                       [w * 0.02, h * 0.98], [w * 0.98, h * 0.98]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h)), M


def run():
    ensure_dir(OUT_DIR)
    images = load_sample_images()
    img = to_gray3(images["checkerboard"])

    trans = translate(img, tx=40, ty=25)
    rot = rotate(img, angle_deg=25, scale=1.0)
    scl = scale_image(img, fx=0.7, fy=1.3)
    aff, aff_M = affine_transform(img)
    proj, proj_M = projective_transform(img)

    save_grid(
        os.path.join(OUT_DIR, "01_basic_transforms.png"),
        [img, trans, rot, scl],
        ["Original", "Translation (tx=40, ty=25)", "Rotation (25 deg)", "Scaling (fx=0.7, fy=1.3)"],
        suptitle="Part 3.1 - Translation / Rotation / Scaling",
        ncols=4,
    )

    save_grid(
        os.path.join(OUT_DIR, "02_affine_vs_projective.png"),
        [img, aff, proj],
        ["Original (parallel grid lines)",
         "Affine transform\n(parallel lines STAY parallel)",
         "Projective transform\n(parallel lines CONVERGE - perspective)"],
        suptitle="Part 3.2 - Affine vs Projective Transformation",
    )

    photo = images["astronaut"]
    aff_photo, _ = affine_transform(photo)
    proj_photo, _ = projective_transform(photo)
    save_grid(
        os.path.join(OUT_DIR, "03_affine_vs_projective_photo.png"),
        [photo, aff_photo, proj_photo],
        ["Original", "Affine", "Projective (homography)"],
        suptitle="Part 3.3 - Same Comparison on a Photo",
    )

    print("[Part3] Affine matrix (2x3):\n", aff_M)
    print("[Part3] Projective/homography matrix (3x3):\n", proj_M)
    print(f"[Part3] Figures written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    run()
