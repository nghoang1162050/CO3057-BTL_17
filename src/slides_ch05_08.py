"""From-scratch building blocks that follow the lecture slides, used by
notebooks/CV_Project_2.ipynb.

  Chap 05 - compositing: alpha/pyramid blending, gradient domain (1-D),
            optimal seams by dynamic programming
  Chap 08 - features: Harris detector, ratio-distance matching, ROC
"""
import cv2
import numpy as np

from slides_ch02_04 import apply_homography, correlate2d, gaussian_kernel, SOBEL_X, SOBEL_Y


# --- Chap 05: blending -----------------------------------------------------

def alpha_blend(left, right, alpha):
    """I = alpha * I_left + (1 - alpha) * I_right; alpha in [0, 1], HxW."""
    a = alpha[..., None] if left.ndim == 3 else alpha
    return np.clip(a * left + (1 - a) * right, 0, 255).astype(np.uint8)


def horizontal_ramp(h, w, width):
    """alpha = 1 on the left, 0 on the right, with a linear transition of
    `width` pixels around the centre (width=0 gives a hard cut)."""
    x = np.arange(w, dtype=np.float32) - w / 2
    ramp = np.clip(0.5 - x / max(width, 1e-6), 0, 1) if width > 0 else (x < 0).astype(np.float32)
    return np.tile(ramp, (h, 1))


def gaussian_pyramid(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(img, levels):
    """Band-pass decomposition: L_i = G_i - up(G_{i+1}); the last entry is
    the low-pass residual."""
    g = gaussian_pyramid(img, levels)
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(levels - 1)]
    return lap + [g[-1]]


def pyramid_blend(a, b, mask, levels=5):
    """Blend each frequency band separately, with the mask blurred more at
    coarser levels: low frequencies mix over a wide band, high frequencies
    over a narrow one."""
    la, lb = laplacian_pyramid(a, levels), laplacian_pyramid(b, levels)
    gm = gaussian_pyramid(mask.astype(np.float32), levels)
    blended = []
    for xa, xb, m in zip(la, lb, gm):
        m = m[..., None] if xa.ndim == 3 else m
        blended.append(m * xa + (1 - m) * xb)
    out = blended[-1]
    for lvl in range(levels - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[lvl].shape[1], blended[lvl].shape[0])) + blended[lvl]
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_1d_gradient(target, source, start, end):
    """Gradient-domain blending in 1-D (slide 'Gradient Domain blending').
    Inside [start, end) solve  min sum (f[i+1]-f[i] - (s[i+1]-s[i]))^2
    with f pinned to `target` at both ends of the hole: keep the source's
    derivative, not its absolute values."""
    n = end - start
    f_left, f_right = target[start - 1], target[end]
    # Unknowns f[start..end-1]; one equation per finite difference touching the hole.
    A = np.zeros((n + 1, n))
    b = np.zeros(n + 1)
    ds = np.diff(source[start - 1:end + 1])
    for k in range(n + 1):
        if k < n:
            A[k, k] = 1
        if k > 0:
            A[k, k - 1] = -1
        b[k] = ds[k]
    b[0] += f_left
    b[n] -= f_right
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    out = target.astype(np.float64).copy()
    out[start:end] = sol
    return out


# --- Chap 05: optimal seam by dynamic programming ----------------------------

def min_error_seam(cost):
    """Vertical seam through an HxW cost map. M(i,j) = E(i,j) +
    min(M(i-1,j-1), M(i-1,j), M(i-1,j+1)); backtrack from the cheapest cell
    of the last row. Returns the column index per row."""
    h, w = cost.shape
    M = cost.astype(np.float64).copy()
    for i in range(1, h):
        left = np.r_[np.inf, M[i - 1, :-1]]
        right = np.r_[M[i - 1, 1:], np.inf]
        M[i] += np.minimum(np.minimum(left, M[i - 1]), right)
    seam = np.empty(h, dtype=int)
    seam[-1] = int(np.argmin(M[-1]))
    for i in range(h - 2, -1, -1):
        j = seam[i + 1]
        lo, hi = max(j - 1, 0), min(j + 2, w)
        seam[i] = lo + int(np.argmin(M[i, lo:hi]))
    return seam


def seam_composite(left, right, seam):
    """Take `left` to the left of the seam and `right` to its right."""
    h, w = left.shape[:2]
    take_left = np.arange(w)[None, :] < seam[:, None]
    take_left = take_left[..., None] if left.ndim == 3 else take_left
    return np.where(take_left, left, right)


# --- Chap 08: Harris corners -------------------------------------------------

def harris_response(gray, sigma=1.5, k=0.04):
    """H = sum_w [[Ix^2, IxIy], [IxIy, Iy^2]] (Gaussian window), then
    R = det(H) - k trace(H)^2, which avoids the eigenvalue square root.
    Also returns the smaller eigenvalue lambda_- (the 'min eigenvalue'
    score of the slides) for comparison."""
    f = gray.astype(np.float32)
    ix, iy = correlate2d(f, SOBEL_X / 8), correlate2d(f, SOBEL_Y / 8)
    win = gaussian_kernel(sigma)
    sxx, syy, sxy = correlate2d(ix * ix, win), correlate2d(iy * iy, win), correlate2d(ix * iy, win)
    det, tr = sxx * syy - sxy ** 2, sxx + syy
    lam_min = tr / 2 - np.sqrt(np.maximum((sxx - syy) ** 2 / 4 + sxy ** 2, 0))
    return det - k * tr ** 2, lam_min


def local_maxima(score, rel_thresh=0.01, window=7):
    """Keep pixels above `rel_thresh * max` that are the maximum of their
    window x window neighbourhood (non-maximum suppression). Returns (N,2) xy."""
    dil = cv2.dilate(score, np.ones((window, window), np.uint8))
    keep = (score >= dil) & (score > rel_thresh * score.max())
    ys, xs = np.nonzero(keep)
    return np.stack([xs, ys], axis=1)


# --- Chap 08: matching and evaluation ------------------------------------------

def descriptor_distances(desc1, desc2, metric="l2"):
    """All-pairs descriptor distances: Euclidean for float descriptors
    (SIFT), Hamming (number of differing bits) for binary ones (ORB)."""
    if metric == "hamming":
        a = np.unpackbits(desc1, axis=1).astype(np.float32)
        b = np.unpackbits(desc2, axis=1).astype(np.float32)
        return a.shape[1] - (a @ b.T + (1 - a) @ (1 - b).T)
    a, b = desc1.astype(np.float32), desc2.astype(np.float32)
    sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0))


def match_ratio_test(desc1, desc2, ratio=0.75, metric="l2"):
    """Nearest-neighbour matching with ratio distance d(f1,f2)/d(f1,f2')
    (f2' = second best). Returns (idx1, idx2, ratio_per_query, nn_idx)."""
    d = descriptor_distances(desc1, desc2, metric)
    order = np.argsort(d, axis=1)
    best, second = d[np.arange(len(d)), order[:, 0]], d[np.arange(len(d)), order[:, 1]]
    r = best / np.maximum(second, 1e-12)
    nn = order[:, 0]
    keep = np.nonzero(r < ratio)[0]
    return keep, nn[keep], r, nn


def match_roc(kp1, kp2, desc1, desc2, H_gt, tol=3.0, n_thresholds=60, metric="l2"):
    """ROC of a feature matcher, with ground truth supplied by the known
    homography H_gt (img1 -> img2). A nearest-neighbour match is *correct*
    if the matched keypoint lies within `tol` px of the true projection.
    Sweeping the ratio-distance threshold gives
        TPR = accepted correct / all correct,
        FPR = accepted wrong   / all wrong."""
    pts1 = np.float32([k.pt for k in kp1])
    pts2 = np.float32([k.pt for k in kp2])
    _, _, ratio, nn = match_ratio_test(desc1, desc2, metric=metric)
    err = np.linalg.norm(apply_homography(H_gt, pts1) - pts2[nn], axis=1)
    correct = err < tol
    n_pos, n_neg = max(int(correct.sum()), 1), max(int((~correct).sum()), 1)
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr = np.array([(correct & (ratio < t)).sum() / n_pos for t in thresholds])
    fpr = np.array([(~correct & (ratio < t)).sum() / n_neg for t in thresholds])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc, int(correct.sum()), len(correct)


# --- Ground truth for the synthetic panorama sets ------------------------------

def synthetic_view_homographies(scene_width, n_views=4, view_width=480, overlap=0.42,
                                jitter_frac=0.05, seed=0, height=360):
    """Exact homography H_i (view i -> view 0) for the views produced by
    `p2_panorama.synthesize_views`: view_i = J_i * crop(scene, x0_i), so
    H_i = J_0 T(-x0_0) T(x0_i) J_i^-1. Lets us measure alignment error
    against the truth instead of judging only by eye."""
    from p2_panorama import _jitter_homography
    step = int(view_width * (1 - overlap))
    x0 = [min(i * step, scene_width - view_width) for i in range(n_views)]
    J = [_jitter_homography(view_width, height, jitter_frac, seed + i) for i in range(n_views)]

    def T(tx):
        return np.array([[1, 0, tx], [0, 1, 0], [0, 0, 1]], np.float64)

    out = []
    for i in range(n_views):
        H = J[0] @ T(-x0[0]) @ T(x0[i]) @ np.linalg.inv(J[i])
        out.append(H / H[2, 2])
    return out


def corner_transfer_error(H_est, H_gt, width, height):
    """Mean distance (px) between the images of the 4 corners and the centre
    of a view under the estimated and the true homography."""
    pts = np.array([[0, 0], [width, 0], [width, height], [0, height], [width / 2, height / 2]], np.float64)
    return float(np.linalg.norm(apply_homography(H_est, pts) - apply_homography(H_gt, pts), axis=1).mean())


def harris_repeatability(detect, img, M, tol=3.0, margin=12):
    """Fraction of keypoints found in `img` that are re-found (within `tol`
    px) after the image is warped by the 3x3 matrix `M` -- the 'invariance'
    question of the slides. `detect(image) -> (N,2) xy array`."""
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    p1 = detect(img)
    p2 = detect(warped)
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    mapped = apply_homography(M, p1.astype(np.float64))
    ok = (mapped[:, 0] > margin) & (mapped[:, 0] < w - margin) & (mapped[:, 1] > margin) & (mapped[:, 1] < h - margin)
    if ok.sum() == 0:
        return 0.0
    d = np.linalg.norm(mapped[ok][:, None, :] - p2[None, :, :].astype(np.float64), axis=2).min(axis=1)
    return float((d < tol).mean())


# ------------------------------------------------------------ graph cut (Chap 05, slide 41-43)
def _pair_costs(A, B):
    """Matching cost of cutting between 4-neighbours p,q (Kwatra / Photomontage):
    |A(p)-B(p)|^2 + |A(q)-B(q)|^2. Returns (horizontal edges HxW-1, vertical edges H-1xW)."""
    d = ((A.astype(np.float64) - B.astype(np.float64)) ** 2).sum(axis=2)
    return d[:, :-1] + d[:, 1:], d[:-1, :] + d[1:, :]


def graphcut_labels(A, B, force_A, force_B, scale=1.0):
    """Min-cut labelling of every pixel as A (True) or B (False). Source = A, sink = B;
    pixels in `force_A` / `force_B` are tied to them with infinite capacity; neighbour edges carry
    the matching cost, so the minimum cut is the cheapest closed boundary separating the two
    forced regions (not restricted to a single vertical path like the DP seam)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow
    h, w = force_A.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    ch, cv = _pair_costs(A, B)
    q = lambda c: np.maximum(np.rint(c * scale / max(ch.max(), cv.max(), 1e-9) * 1000000), 1).astype(np.int32)   # scipy max-flow needs int32 capacities
    src, dst, cap = [], [], []
    for a, b, c in ((idx[:, :-1], idx[:, 1:], ch), (idx[:-1, :], idx[1:, :], cv)):
        a, b, c = a.ravel(), b.ravel(), q(c).ravel()
        src += [a, b]; dst += [b, a]; cap += [c, c]
    S, T, INF = n, n + 1, 2 * 10 ** 9
    fa, fb = idx[force_A], idx[force_B]
    src += [np.full(len(fa), S), fb]; dst += [fa, np.full(len(fb), T)]; cap += [np.full(len(fa), INF, np.int32), np.full(len(fb), INF, np.int32)]
    g = csr_matrix((np.concatenate(cap).astype(np.int32), (np.concatenate(src), np.concatenate(dst))), shape=(n + 2, n + 2))
    flow = maximum_flow(g, S, T)
    resid = g - flow.flow
    resid.data = np.maximum(resid.data, 0); resid.eliminate_zeros()
    from scipy.sparse.csgraph import breadth_first_order
    reach = breadth_first_order(resid, S, directed=True, return_predecessors=False)
    lab = np.zeros(n + 2, bool); lab[reach] = True
    return lab[:n].reshape(h, w), float(flow.flow_value) * max(ch.max(), cv.max()) / 1000000 / scale


def labels_energy(labels, A, B):
    """Sum of matching costs over 4-neighbour pairs with different labels (same energy the graph cut minimises)."""
    ch, cv = _pair_costs(A, B)
    return float(ch[labels[:, :-1] != labels[:, 1:]].sum() + cv[labels[:-1, :] != labels[1:, :]].sum())


def labels_composite(labels, A, B):
    return np.where(labels[..., None], A, B)
