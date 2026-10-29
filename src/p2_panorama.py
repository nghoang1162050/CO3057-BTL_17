"""Project 2, Part 2: panorama stitching with SIFT and ORB.

Pipeline: features -> ratio-test matching -> RANSAC homography -> warp every
view into the first view's frame -> feather-blend the overlaps. The same
pipeline runs once per feature extractor so the two compare on identical input.
"""
import os
import time

import cv2
import numpy as np

from utils import ensure_dir, save_grid, save_image, load_image

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "p2_panorama")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "p2_panorama")


def load_views(data_dir=DATA_DIR):
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(files) < 3:
        raise FileNotFoundError(f"Put at least 3 overlapping photos of one scene in {data_dir}")
    print(f"[P2 Part2] Using {len(files)} images from {data_dir}")
    return [load_image(os.path.join(data_dir, f)) for f in files]


def detect_and_describe(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create() if method == "sift" else cv2.ORB_create(nfeatures=2000)
    return detector.detectAndCompute(gray, None)


def match_features(desc1, desc2, method, ratio=0.75):
    """Lowe's ratio test: keep a match only if it is clearly better than the runner-up."""
    matcher = cv2.BFMatcher(cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING)
    return [p[0] for p in matcher.knnMatch(desc1, desc2, k=2) if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def estimate_homography(kp1, kp2, matches, ransac_thresh=4.0):
    """Homography taking image-2 points into image-1 coordinates (RANSAC)."""
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)


def feather_blend(canvas, canvas_mask, warped, warped_mask):
    """In overlaps, weight each pixel by its distance to its own image's border."""
    dist_c = cv2.distanceTransform(canvas_mask, cv2.DIST_L2, 5)
    dist_w = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5)
    overlap = (canvas_mask > 0) & (warped_mask > 0)
    only_new = (warped_mask > 0) & (canvas_mask == 0)

    denom = dist_c + dist_w
    denom[denom == 0] = 1
    w_new = np.zeros(dist_c.shape, dtype=np.float32)
    w_new[overlap] = dist_w[overlap] / denom[overlap]
    w_new[only_new] = 1.0

    out = canvas.astype(np.float32) * (1 - w_new[..., None]) + warped.astype(np.float32) * w_new[..., None]
    out[only_new] = warped[only_new]
    canvas_only = (canvas_mask > 0) & (warped_mask == 0)
    out[canvas_only] = canvas[canvas_only]
    return np.clip(out, 0, 255).astype(np.uint8)


def stitch_sequence(images, method="sift", ratio=0.75, ransac_thresh=4.0, max_matches_drawn=60):
    keypoints, descriptors = zip(*(detect_and_describe(img, method) for img in images))

    H_ref = [np.eye(3)]  # view i -> view 0
    pair_stats, matches_list = [], []
    for i in range(1, len(images)):
        matches = match_features(descriptors[i - 1], descriptors[i], method, ratio)
        H, inlier_mask = estimate_homography(keypoints[i - 1], keypoints[i], matches, ransac_thresh)
        if H is None:
            raise RuntimeError(f"Not enough good matches between view {i - 1} and view {i} ({method}).")
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        pair_stats.append({"pair": (i - 1, i), "matches": len(matches), "inliers": inliers,
                           "inlier_ratio": inliers / len(matches) if matches else 0.0})
        matches_list.append(matches)
        H_ref.append(H_ref[-1] @ H)

    corners = np.concatenate([
        cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
        for (h, w), H in zip((im.shape[:2] for im in images), H_ref)])
    xmin, ymin = np.floor(corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(corners.max(axis=0).ravel()).astype(int)
    tx, ty = -min(xmin, 0), -min(ymin, 0)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    size = (int(xmax + tx), int(ymax + ty))

    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    for img, H in zip(images, H_ref):
        warped = cv2.warpPerspective(img, T @ H, size)
        warped_mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), T @ H, size)
        canvas = feather_blend(canvas, canvas_mask, warped, warped_mask)
        canvas_mask = cv2.bitwise_or(canvas_mask, warped_mask)

    return {
        "panorama": canvas,
        "keypoints_viz": [cv2.drawKeypoints(img, kp, None, color=(0, 255, 0)) for img, kp in zip(images, keypoints)],
        "matches_viz": [cv2.drawMatches(images[i - 1], keypoints[i - 1], images[i], keypoints[i],
                                        matches_list[i - 1][:max_matches_drawn], None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        for i in range(1, len(images))],
        "pair_stats": pair_stats,
        "keypoint_counts": [len(kp) for kp in keypoints],
    }


def opencv_reference_stitch(images):
    """OpenCV's built-in Stitcher, used only as an independent sanity check."""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch([cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images])
    return status, (cv2.cvtColor(pano, cv2.COLOR_BGR2RGB) if status == cv2.Stitcher_OK else None)


def run(data_dir=DATA_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    images = load_views(data_dir)
    n = len(images)
    save_grid(os.path.join(out_dir, "00_inputs.png"), images, [f"View {i + 1}" for i in range(n)],
              suptitle="Project 2, Part 2 - Input image set", ncols=n)

    results = {}
    for method in ("sift", "orb"):
        t0 = time.time()
        res = results[method] = stitch_sequence(images, method)
        elapsed = time.time() - t0

        save_grid(os.path.join(out_dir, f"01_keypoints_{method}.png"), res["keypoints_viz"],
                  [f"View {i + 1}: {c} keypoints" for i, c in enumerate(res["keypoint_counts"])],
                  suptitle=f"Project 2, Part 2 - Keypoints ({method.upper()})", ncols=n)
        for i, vis in enumerate(res["matches_viz"]):
            save_image(os.path.join(out_dir, f"02_matches_{method}_{i}-{i + 1}.png"), vis)
        save_image(os.path.join(out_dir, f"03_panorama_{method}.png"), res["panorama"])

        print(f"[P2 Part2][{method.upper()}] total time: {elapsed:.2f}s, keypoints/view: {res['keypoint_counts']}")
        for s in res["pair_stats"]:
            print(f"  pair {s['pair']}: {s['matches']} good matches, "
                  f"{s['inliers']} RANSAC inliers ({s['inlier_ratio'] * 100:.1f}%)")

    save_grid(os.path.join(out_dir, "04_sift_vs_orb.png"), [results["sift"]["panorama"], results["orb"]["panorama"]],
              ["SIFT pipeline", "ORB pipeline"], suptitle="Project 2, Part 2 - Panorama: SIFT vs ORB", ncols=2)

    try:
        status, pano = opencv_reference_stitch(images)
        if pano is not None:
            save_image(os.path.join(out_dir, "05_opencv_stitcher_reference.png"), pano)
            print("[P2 Part2] cv2.Stitcher reference: OK (saved for comparison)")
        else:
            print(f"[P2 Part2] cv2.Stitcher reference failed with status={status}")
    except Exception as e:
        print(f"[P2 Part2] cv2.Stitcher reference skipped: {e}")
    print(f"[P2 Part2] Figures written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Panorama stitching, SIFT vs ORB")
    ap.add_argument("--data", default=DATA_DIR, help="folder with at least 3 overlapping photos")
    ap.add_argument("--out", default=OUT_DIR, help="folder for the figures")
    args = ap.parse_args()
    run(args.data, args.out)
