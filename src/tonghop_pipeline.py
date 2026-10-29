"""Tonghop: preprocessing (3.1) -> geometry (3.2) -> content (3.3) -> visualization (3.4)."""
import os
import time

import cv2
import numpy as np

from utils import ensure_dir, save_grid, to_gray3
from tonghop_preprocess import load_scene, preprocess, DATA_DIR
from tonghop_geometry import detect_edges, detect_lines, detect_corners, draw_geometry
from tonghop_content import spectral_residual_saliency, saliency_to_bbox, segment_grabcut, draw_content_overlay

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tonghop_scene")


def run(data_dir=DATA_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    scene = load_scene(data_dir)
    pre = preprocess(scene)

    t0 = time.time()
    edges = detect_edges(pre["gray"])
    lines = detect_lines(edges)
    corners = detect_corners(pre["gray"])
    t_geometry = time.time() - t0
    geometry_viz = draw_geometry(pre["denoised"], lines, corners)

    t0 = time.time()
    saliency = spectral_residual_saliency(pre["gray"])
    rect = saliency_to_bbox(saliency)
    mask = segment_grabcut(pre["denoised"], rect)
    t_content = time.time() - t0
    content_viz = draw_content_overlay(pre["denoised"], mask)

    saliency_viz = cv2.cvtColor(cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET),
                                cv2.COLOR_BGR2RGB)
    rect_viz = pre["denoised"].copy()
    x, y, w, h = rect
    cv2.rectangle(rect_viz, (x, y), (x + w, y + h), (255, 255, 0), 2)

    save_grid(os.path.join(out_dir, "00_preprocess.png"),
              [pre["original"], pre["normalized"], pre["denoised"], to_gray3(pre["gray"])],
              ["Original", "CLAHE lighting-normalized", "Bilateral denoised", "Grayscale"],
              suptitle="Tonghop 3.1 - Preprocessing")
    save_grid(os.path.join(out_dir, "01_geometry.png"), [pre["denoised"], to_gray3(edges), geometry_viz],
              ["Input", "Canny edges", f"{len(lines)} Hough lines + {len(corners)} Shi-Tomasi corners"],
              suptitle="Tonghop 3.2 - Geometric Structure Analysis")
    save_grid(os.path.join(out_dir, "02_content.png"), [pre["denoised"], saliency_viz, rect_viz, content_viz],
              ["Input", "Spectral-residual saliency", "Saliency -> GrabCut seed rect", "GrabCut segmentation"],
              suptitle="Tonghop 3.3 - Content Analysis (saliency -> segmentation)", ncols=4)
    save_grid(os.path.join(out_dir, "03_before_after.png"), [scene, geometry_viz, content_viz],
              ["Before (raw input)", "After: geometric structure", "After: content segmentation"],
              suptitle="Tonghop 3.4 - Before / After Visualization")

    std_before = float(cv2.cvtColor(pre["original"], cv2.COLOR_RGB2GRAY).std())
    std_after = float(cv2.cvtColor(pre["normalized"], cv2.COLOR_RGB2GRAY).std())
    print(f"[Tonghop] Preprocessing: grayscale contrast (std) {std_before:.2f} -> {std_after:.2f} after CLAHE")
    print(f"[Tonghop] Geometry: {len(lines)} lines, {len(corners)} corners in {t_geometry * 1000:.1f} ms")
    print(f"[Tonghop] Content: saliency+GrabCut in {t_content * 1000:.1f} ms, "
          f"foreground = {float((mask > 0).mean()) * 100:.1f}% of image")
    print(f"[Tonghop] Figures written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Capstone scene-analysis pipeline")
    ap.add_argument("--data", default=DATA_DIR, help="folder with the scene photo (else the 'camera' sample)")
    ap.add_argument("--out", default=OUT_DIR, help="folder for the figures")
    args = ap.parse_args()
    run(args.data, args.out)
