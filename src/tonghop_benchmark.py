"""Segmentation benchmark with exact ground truth for the capstone (3.3).

A textured foreground (any photo) is cut out with a real silhouette
(`skimage.data.horse`) and pasted onto a different background, so the
true mask is known per pixel. Every method is run on several scenes and
several random seeds (GrabCut is stochastic through its k-means init),
and each unsupervised method is reported twice:

  * `blind`  - foreground label chosen by the pipeline's own rule
               (highest mean saliency), the realistic setting;
  * `oracle` - foreground label chosen with the ground truth (best IoU),
               an upper bound that separates the algorithm's quality from
               the quality of the label-selection rule.
"""
import cv2
import numpy as np
from skimage import data, img_as_ubyte

import slides_ch09_11 as s9
import tonghop_content as content
import tonghop_preprocess as prep

H, W = 360, 480


def _resize(a, shape_hw=(H, W)):
    return cv2.resize(img_as_ubyte(a), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)


def sky_background():
    yy = np.linspace(0, 1, H)[:, None, None]
    sky = np.clip(np.array([70, 110, 190]) * (1 - yy) + np.array([170, 200, 230]) * yy, 0, 255) * np.ones((1, W, 1))
    return np.clip(sky + np.random.default_rng(0).normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)


def silhouette_mask(scale=0.6, x0=40, bottom_margin=30):
    horse = ~data.horse()
    oh, ow = int(328 * scale), int(400 * scale)
    sil = cv2.resize(horse.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
    gt = np.zeros((H, W), np.uint8)
    gt[H - oh - bottom_margin:H - bottom_margin, x0:x0 + ow] = sil * 255
    return gt


def scene_catalog():
    """name -> (foreground photo, background photo, difficulty tag)."""
    moon = cv2.cvtColor(img_as_ubyte(data.moon()), cv2.COLOR_GRAY2RGB)
    return {
        "astronaut/sky": (data.astronaut(), sky_background(), "high contrast"),
        "rocket/moon": (data.rocket(), moon, "high contrast"),
        "chelsea/sky": (data.chelsea(), sky_background(), "high contrast"),
        "astronaut/coffee": (data.astronaut(), data.coffee(), "medium"),
        "chelsea/coffee": (data.chelsea(), data.coffee(), "low contrast"),
        "coffee/chelsea": (data.coffee(), data.chelsea(), "low contrast"),
    }


def compose(fg, bg, gt):
    scene = _resize(bg).copy()
    m = gt > 0
    scene[m] = _resize(fg)[m]
    return scene


def _label_masks(labels):
    return [((labels == i) * 255).astype(np.uint8) for i in np.unique(labels)]


def _pick(masks, sal=None, gt=None):
    """Blind: mask with the highest mean saliency. Oracle: highest IoU with gt."""
    if gt is not None:
        return max(masks, key=lambda m: s9.iou(m, gt))
    return max(masks, key=lambda m: sal[m > 0].mean() if (m > 0).any() else -1)


def run_scene(scene, gt, seed, grabcut_iters=5):
    """Return {method: mask} for one scene/seed. Oracle variants carry an
    ' (oracle)' suffix."""
    cv2.setRNGSeed(seed)
    p = prep.preprocess(scene)
    sal = content.spectral_residual_saliency(p["gray"])
    rect = content.saliency_to_bbox(sal)
    g = p["gray"]
    out = {}

    otsu = [((g > s9.otsu_threshold(g)) * 255).astype(np.uint8), ((g <= s9.otsu_threshold(g)) * 255).astype(np.uint8)]
    ada = s9.adaptive_threshold(g, 51, 5)
    ada = [ada, 255 - ada]
    km = _label_masks(s9.kmeans_segment(p["denoised"], 3, seed=seed))
    kmp = _label_masks(s9.kmeans_segment(p["denoised"], 3, True, 1.0, seed=seed))
    for name, masks in [("Otsu", otsu), ("Ngưỡng thích nghi", ada), ("K-means (màu)", km), ("K-means (màu+vị trí)", kmp)]:
        out[name] = _pick(masks, sal=sal)
        out[name + " (oracle)"] = _pick(masks, gt=gt)

    fg = s9.erode((sal > np.percentile(sal, 95)).astype(np.uint8) * 255, s9.disk(1))
    bg = ((sal < np.percentile(sal, 40)) * 255).astype(np.uint8)
    out["Watershed (marker từ saliency)"] = s9.watershed_markers(g, fg, bg)

    seeds = {"GrabCut (hộp saliency)": rect,
             "GrabCut (hộp 80% giữa ảnh)": (int(.1 * g.shape[1]), int(.1 * g.shape[0]), int(.8 * g.shape[1]), int(.8 * g.shape[0]))}
    ys, xs = np.nonzero(gt)
    seeds["GrabCut (hộp lý tưởng)"] = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    for name, r in seeds.items():
        out[name] = content.segment_grabcut(p["denoised"], r, iterations=grabcut_iters)
    return out


def run_benchmark(seeds=(0, 1, 2), scale=0.6, grabcut_iters=5):
    """Returns (table, per_scene): table[method] = (mean IoU, std IoU over
    all scene x seed runs); per_scene[scene][method] = mean IoU over seeds."""
    gt = silhouette_mask(scale)
    scores = {}
    per_scene = {}
    for sname, (fg, bg, tag) in scene_catalog().items():
        scene = compose(fg, bg, gt)
        acc = {}
        for sd in seeds:
            for m, mask in run_scene(scene, gt, sd, grabcut_iters).items():
                acc.setdefault(m, []).append(s9.iou(mask, gt))
                scores.setdefault(m, []).append(acc[m][-1])
        per_scene[sname] = {m: float(np.mean(v)) for m, v in acc.items()}
    table = {m: (float(np.mean(v)), float(np.std(v))) for m, v in scores.items()}
    return table, per_scene, gt
