"""Project 2, Part 1: gradient-domain editing (Poisson seamless cloning).

For each unknown pixel p inside the mask with 4-neighbors N(p):

    4 f(p) - sum_q f(q) = sum_q (g(p) - g(q)) + sum_{q outside mask} B(q)

where g is the source and B the background. Pixels outside the mask are fixed
(Dirichlet boundary), so the pasted region takes the background's color while
keeping the source's gradients.
"""
import os
import time

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from utils import ensure_dir, save_grid, load_image, load_mask, to_gray3

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "p2_gradient_domain")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "p2_gradient_domain")
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def make_synthetic_pair(bg_size=(360, 500), src_size=(170, 170), seed=0):
    """Sunset-gradient background and a star on a flat, mismatched backdrop.
    The mask covers the whole circular patch, so a plain paste shows a halo."""
    h, w = bg_size
    top = np.array([255, 170, 90], dtype=np.float32)
    bottom = np.array([55, 30, 85], dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    background = np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()

    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=25)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    background = np.clip(background + noise[..., None] * 25 - 12, 0, 255).astype(np.uint8)

    sh, sw = src_size
    source = np.full((sh, sw, 3), (120, 200, 235), dtype=np.uint8)
    radii = np.where(np.arange(10) % 2 == 0, min(sh, sw) * 0.30, min(sh, sw) * 0.13)
    angles = np.deg2rad(-90) + np.arange(10) * np.pi / 5
    star = np.stack([sw // 2 + radii * np.cos(angles), sh // 2 + radii * np.sin(angles)], axis=1)
    cv2.fillPoly(source, [star.astype(np.int32)], (255, 221, 51))

    mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.circle(mask, (sw // 2, sh // 2), int(min(sh, sw) * 0.47), 255, -1)
    offset = (h // 2 - sh // 2 - 25, w // 2 - sw // 2)  # (y, x) of the patch's top-left corner
    return background, source, mask, offset


def load_pair(data_dir=DATA_DIR):
    """Load background/source/mask (.png or .jpg) and an optional offset.txt ("y,x")
    from data_dir; use the synthetic pair if the three images are not all there."""
    def find(stem):
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(data_dir, stem + ext)
            if os.path.exists(path):
                return path

    paths = [find(s) for s in ("background", "source", "mask")]
    if not all(paths):
        print("[P2 Part1] No complete custom image set -- using the synthetic demo pair")
        return make_synthetic_pair()

    background, source, mask = load_image(paths[0]), load_image(paths[1]), load_mask(paths[2])
    mask = ((mask > 127) * 255).astype(np.uint8)
    if source.shape[:2] != mask.shape[:2]:
        raise ValueError("source and mask must have the same size")
    offset_path = os.path.join(data_dir, "offset.txt")
    if os.path.exists(offset_path):
        y, x = (int(v) for v in open(offset_path).read().strip().split(","))
    else:
        y, x = (background.shape[0] - mask.shape[0]) // 2, (background.shape[1] - mask.shape[1]) // 2
    print(f"[P2 Part1] Using custom images from {data_dir}")
    return background, source, mask, (y, x)


def shift(a, dy, dx):
    """out[y, x] = a[y + dy, x + dx], zero outside the array."""
    h, w = a.shape
    return np.pad(a, 1)[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def build_laplacian(mask):
    """Sparse matrix A: 4 on the diagonal, -1 for each neighbor that is also unknown."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    index_map = -np.ones(mask.shape, dtype=np.int64)
    index_map[ys, xs] = np.arange(n)
    padded = np.pad(index_map, 1, constant_values=-1)

    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in DIRECTIONS:
        nb = padded[ys + 1 + dy, xs + 1 + dx]
        valid = nb >= 0
        rows.append(np.arange(n)[valid])
        cols.append(nb[valid])
        vals.append(np.full(valid.sum(), -1.0))
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def poisson_rhs(S, B, mask, mix_gradients):
    """Source Laplacian plus the known background value for neighbors outside the mask."""
    rhs = np.zeros_like(S)
    for dy, dx in DIRECTIONS:
        B_n, mask_n = shift(B, dy, dx), shift(mask, dy, dx)
        diff = S - shift(S, dy, dx)
        if mix_gradients:
            diff_b = B - B_n
            diff = np.where(np.abs(diff) >= np.abs(diff_b), diff, diff_b)
        rhs += diff + np.where(mask_n == 0, B_n, 0.0)
    return rhs


def check_placement(background, mask, offset):
    (y, x), (sh, sw), (bh, bw) = offset, mask.shape, background.shape[:2]
    if y < 0 or x < 0 or y + sh > bh or x + sw > bw:
        raise ValueError(f"Mask {(sh, sw)} at offset {offset} does not fit in background {(bh, bw)}")


def poisson_blend(background, source, mask, offset, mix_gradients=False):
    """Clone `source` (where mask > 0) into `background` at offset (y, x).
    mix_gradients=True keeps the stronger of the source and background gradients."""
    check_placement(background, mask, offset)
    background, source = background.astype(np.float64), source.astype(np.float64)
    by, bx = offset
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, mask.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, mask.shape[1])

    local_mask, local_src = mask[y0:y1, x0:x1], source[y0:y1, x0:x1]
    lh, lw = local_mask.shape
    gy, gx = by + y0, bx + x0
    local_bg = background[gy:gy + lh, gx:gx + lw]
    inside = local_mask > 0

    solver = splu(build_laplacian(local_mask).tocsc())  # one factorization for all channels
    result = local_bg.copy()
    for c in range(3):
        rhs = poisson_rhs(local_src[..., c], local_bg[..., c], local_mask, mix_gradients)
        result[..., c][inside] = np.clip(solver.solve(rhs[inside]), 0, 255)

    out = background.copy()
    out[gy:gy + lh, gx:gx + lw] = result
    return np.clip(out, 0, 255).astype(np.uint8)


def naive_paste(background, source, mask, offset):
    """Baseline: hard cut-and-paste of the masked pixels."""
    check_placement(background, mask, offset)
    by, bx = offset
    out = background.copy()
    region = out[by:by + mask.shape[0], bx:bx + mask.shape[1]]
    region[mask > 0] = source[mask > 0]
    return out


def feathered_paste(background, source, mask, offset, feather_px=12):
    """Baseline: alpha blend with a Gaussian-softened mask."""
    check_placement(background, mask, offset)
    by, bx = offset
    sh, sw = mask.shape
    alpha = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=feather_px) / 255.0, 0, 1)[..., None]
    out = background.astype(np.float64)
    out[by:by + sh, bx:bx + sw] = out[by:by + sh, bx:bx + sw] * (1 - alpha) + source.astype(np.float64) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def full_size_mask(shape_hw, mask, offset):
    """Place `mask` at `offset` inside an otherwise empty array of the background's size."""
    full = np.zeros(shape_hw, dtype=np.uint8)
    by, bx = offset
    full[by:by + mask.shape[0], bx:bx + mask.shape[1]] = mask
    return full


_full_size_mask = full_size_mask  # name used by the notebooks


def boundary_ring(mask, radius=3):
    kernel = np.ones((3, 3), np.uint8)
    return (cv2.dilate(mask, kernel, iterations=radius) > 0) & (cv2.erode(mask, kernel, iterations=radius) == 0)


def seam_score(img_rgb, ring):
    """Mean Sobel gradient magnitude in a thin ring around the mask boundary."""
    if not ring.any():
        return 0.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2)[ring].mean())


def run(data_dir=DATA_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    background, source, mask, offset = load_pair(data_dir)

    naive = naive_paste(background, source, mask, offset)
    feather = feathered_paste(background, source, mask, offset)
    t0 = time.time()
    poisson = poisson_blend(background, source, mask, offset)
    t_poisson = time.time() - t0
    mixed = poisson_blend(background, source, mask, offset, mix_gradients=True)

    by, bx = offset
    sh, sw = mask.shape
    ring = boundary_ring(full_size_mask(background.shape[:2], mask, offset))
    scores = {
        "background alone (reference)": seam_score(background, ring),
        "naive cut-paste": seam_score(naive, ring),
        "feathered alpha blend": seam_score(feather, ring),
        "Poisson (import gradients)": seam_score(poisson, ring),
        "Poisson (mixed gradients)": seam_score(mixed, ring),
    }

    save_grid(os.path.join(out_dir, "01_inputs.png"), [background, source, to_gray3(mask)],
              ["Background", "Source (object on mismatched backdrop)", "Mask"],
              suptitle="Project 2, Part 1 - Inputs")
    save_grid(os.path.join(out_dir, "02_comparison.png"), [naive, feather, poisson, mixed],
              ["Naive cut-paste\n(hard edge, halo)", "Feathered alpha blend\n(soft edge, halo still visible)",
               "Poisson blending\n(import gradients)", "Poisson blending\n(mixed gradients)"],
              suptitle="Project 2, Part 1 - Gradient Domain Editing vs. Direct Compositing", ncols=4)

    pad = 25
    y0, y1 = max(by - pad, 0), min(by + sh + pad, background.shape[0])
    x0, x1 = max(bx - pad, 0), min(bx + sw + pad, background.shape[1])
    save_grid(os.path.join(out_dir, "03_boundary_zoom.png"),
              [im[y0:y1, x0:x1] for im in (naive, feather, poisson)],
              ["Naive (zoom)", "Feathered (zoom)", "Poisson (zoom)"],
              suptitle="Project 2, Part 1 - Zoomed Boundary Comparison")

    print(f"[P2 Part1] Poisson solve: {int(mask.sum() / 255)} unknown pixels/channel in {t_poisson * 1000:.1f} ms")
    for name, score in scores.items():
        print(f"[P2 Part1] seam score ({name}): {score:.2f}")
    print(f"[P2 Part1] Figures written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gradient-domain (Poisson) image editing")
    ap.add_argument("--data", default=DATA_DIR, help="folder with background/source/mask (+ optional offset.txt)")
    ap.add_argument("--out", default=OUT_DIR, help="folder for the figures")
    args = ap.parse_args()
    run(args.data, args.out)
