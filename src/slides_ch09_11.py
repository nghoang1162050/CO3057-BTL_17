"""From-scratch building blocks that follow the lecture slides, used by
notebooks/CV_Project_Tonghop.ipynb.

  Chap 06-07 - binary morphology (erosion, dilation, opening, closing)
  Chap 09    - Hough transform for lines and circles
  Chap 10-11 - segmentation: Otsu / adaptive thresholding, region growing,
               k-means, marker-controlled watershed, overlap metrics
"""
import cv2
import numpy as np


# --- Chap 06-07: morphology --------------------------------------------------

def _offsets(se):
    """Offsets (dy, dx) of the foreground pixels of a structuring element,
    relative to its origin (centre)."""
    cy, cx = se.shape[0] // 2, se.shape[1] // 2
    ys, xs = np.nonzero(se)
    return [(int(y - cy), int(x - cx)) for y, x in zip(ys, xs)]


def _shifted(a, dy, dx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def erode(binary, se):
    """A (-) B = {z | B_z is contained in A}: a pixel survives only if the
    structuring element, centred on it, fits entirely inside the object.
    Outside the image counts as object, so the border itself is not eroded
    (same convention as OpenCV)."""
    a = binary > 0
    out = np.ones_like(a)
    for dy, dx in _offsets(se):
        out &= _shifted(a, -dy, -dx, True)
    return (out * 255).astype(np.uint8)


def dilate(binary, se):
    """A (+) B = {z | (B^)_z intersects A}: a pixel is set if the reflected
    structuring element, centred on it, touches the object."""
    a = binary > 0
    out = np.zeros_like(a)
    for dy, dx in _offsets(se):
        out |= _shifted(a, dy, dx, False)
    return (out * 255).astype(np.uint8)


def opening(binary, se):
    """(A (-) B) (+) B: removes thin protrusions and small specks, smooths
    the contour from inside."""
    return dilate(erode(binary, se), se)


def closing(binary, se):
    """(A (+) B) (-) B: fills small holes and narrow gaps, smooths the
    contour from outside."""
    return erode(dilate(binary, se), se)


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


# --- Chap 09: Hough transform ---------------------------------------------------

def hough_lines_accumulator(edges, theta_step_deg=1.0, rho_step=1.0):
    """Every edge pixel votes for all (rho, theta) with
    rho = x cos(theta) + y sin(theta)  (the sinusoid in Hough space).
    Returns (accumulator[rho_idx, theta_idx], rhos, thetas_rad)."""
    h, w = edges.shape
    thetas = np.deg2rad(np.arange(-90, 90, theta_step_deg))
    diag = np.hypot(h, w)
    rhos = np.arange(-diag, diag + rho_step, rho_step)
    ys, xs = np.nonzero(edges)
    acc = np.zeros((len(rhos), len(thetas)), np.int32)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for j in range(len(thetas)):                       # vectorised over edge points, one column per theta
        r_idx = np.round((xs * cos_t[j] + ys * sin_t[j] + diag) / rho_step).astype(int)
        acc[:, j] = np.bincount(r_idx, minlength=len(rhos))[:len(rhos)]
    return acc, rhos, thetas


def hough_chord_normalized(acc, shape_hw, theta_step_deg=1.0, rho_step=1.0, min_chord=60):
    """Votes divided by the number of image pixels the line (rho, theta)
    crosses (its chord length), i.e. the *density* of edge pixels along the
    line. Raw Hough votes grow with chord length, so in a dense edge map
    (texture) the longest chords -- the image diagonals -- win regardless of
    structure; dividing removes that bias. Lines whose chord is shorter than
    `min_chord` px are zeroed because their density is too noisy."""
    full = np.ones(shape_hw, np.uint8)
    chord, _, _ = hough_lines_accumulator(full, theta_step_deg, rho_step)
    out = np.where(chord >= min_chord, acc / np.maximum(chord, 1), 0.0)
    return out.astype(np.float32)


def hough_peaks(acc, n_peaks=10, rel_thresh=0.4, nms=9):
    """Strongest local maxima of the accumulator, with non-maximum
    suppression so adjacent cells count as one peak. Returns (row, col, votes)."""
    a = acc.astype(np.float32)
    dil = cv2.dilate(a, np.ones((nms, nms), np.uint8))
    cand = np.argwhere((a >= dil) & (a >= rel_thresh * a.max()))
    votes = a[cand[:, 0], cand[:, 1]]
    order = np.argsort(-votes)[:n_peaks]
    return [(int(cand[i, 0]), int(cand[i, 1]), int(votes[i])) for i in order]


def draw_hough_lines(img_rgb, peaks, rhos, thetas, color=(255, 60, 60)):
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for r_i, t_i, _ in peaks:
        rho, th = rhos[r_i], thetas[t_i]
        a, b = np.cos(th), np.sin(th)
        x0, y0 = a * rho, b * rho
        p1 = (int(x0 - 2000 * b), int(y0 + 2000 * a))
        p2 = (int(x0 + 2000 * b), int(y0 - 2000 * a))
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
    return out


def hough_circles(edges, gray, radii, n_peaks=8, nms_dist=None, rel_thresh=0.5):
    """Circle Hough transform with gradient information (slide 'Using
    Gradient Information'): each edge pixel casts a vote only along its own
    gradient direction, at distance r on both sides, instead of on a full
    circle -- one accumulator cell per radius per direction.
    Returns [(x, y, r, votes)], strongest first, after suppressing
    centres closer than `nms_dist`."""
    f = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    ys, xs = np.nonzero(edges)
    mag = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-9
    ux, uy = gx[ys, xs] / mag, gy[ys, xs] / mag
    h, w = gray.shape
    found = []
    for r in radii:
        acc = np.zeros((h, w), np.float32)
        for sign in (1, -1):
            cx = np.round(xs + sign * r * ux).astype(int)
            cy = np.round(ys + sign * r * uy).astype(int)
            ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            np.add.at(acc, (cy[ok], cx[ok]), 1)
        acc = cv2.GaussianBlur(acc, (0, 0), 1.5) * 9
        dil = cv2.dilate(acc, np.ones((2 * int(r * 0.5) + 1,) * 2, np.uint8))
        for y, x in np.argwhere((acc >= dil) & (acc > 0)):
            found.append((int(x), int(y), int(r), float(acc[y, x])))
    if not found:
        return []
    top = max(f_[3] for f_ in found)
    found = sorted([f_ for f_ in found if f_[3] >= rel_thresh * top], key=lambda t: -t[3])
    kept = []
    for x, y, r, v in found:
        d = nms_dist or r
        if all(np.hypot(x - kx, y - ky) > d for kx, ky, _, _ in kept):
            kept.append((x, y, r, v))
        if len(kept) == n_peaks:
            break
    return kept


# --- Chap 10-11: segmentation --------------------------------------------------

def otsu_threshold(gray):
    """Global threshold that maximizes the between-class variance of the
    histogram (equivalently minimizes the within-class variance)."""
    p = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / np.where(denom > 0, denom, 1), 0)
    return int(np.argmax(sigma_b))


def adaptive_threshold(gray, block=35, c=5):
    """Local threshold T(x,y) = mean of the block x block neighbourhood - c;
    copes with non-uniform illumination where one global T cannot."""
    mean = cv2.blur(gray.astype(np.float32), (block, block))
    return ((gray.astype(np.float32) > mean - c) * 255).astype(np.uint8)


def region_growing(gray, seed_xy, tol=12):
    """Pixel aggregation from one seed: append 4-connected neighbours whose
    gray level is within `tol` of the running region mean."""
    h, w = gray.shape
    sx, sy = seed_xy
    region = np.zeros((h, w), bool)
    region[sy, sx] = True
    total, count = float(gray[sy, sx]), 1
    frontier = [(sy, sx)]
    while frontier:
        y, x = frontier.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not region[ny, nx] and abs(float(gray[ny, nx]) - total / count) <= tol:
                region[ny, nx] = True
                total += float(gray[ny, nx])
                count += 1
                frontier.append((ny, nx))
    return (region * 255).astype(np.uint8)


def kmeans(x, k, iters=50, seed=0):
    """Lloyd's algorithm exactly as on the slide: random initial centroids,
    assign each point to the nearest centroid, recompute centroids, repeat
    until the assignment stops changing. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), k, replace=False)].astype(np.float64)
    labels = np.full(len(x), -1)
    for _ in range(iters):
        d = ((x[:, None, :] - centroids[None]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centroids[j] = x[labels == j].mean(axis=0)
    return labels, centroids


def kmeans_segment(img, k=4, use_position=False, position_weight=1.0, seed=0, max_dim=160):
    """Cluster pixels by colour (R,G,B) or colour + position (R,G,B,x,y).
    Features are rescaled to [0,1] first (axis-scaling slide) so no feature
    dominates. Runs at reduced size for speed, labels are resized back.
    Returns the label map at the input resolution."""
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    small = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    feats = small.reshape(-1, small.shape[-1] if small.ndim == 3 else 1).astype(np.float64) / 255.0
    if use_position:
        ys, xs = np.mgrid[0:sh, 0:sw]
        pos = np.stack([xs.ravel() / sw, ys.ravel() / sh], axis=1) * position_weight
        feats = np.hstack([feats, pos])
    labels, _ = kmeans(feats, k, seed=seed)
    return cv2.resize(labels.reshape(sh, sw).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def watershed_no_markers(gray, sigma=1.0):
    """Watershed of the gradient magnitude with one basin per regional
    minimum -- the slide's 'over-segmentation' failure mode."""
    from skimage.segmentation import watershed
    g = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma else gray
    grad = cv2.magnitude(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1))
    return watershed(grad)


def watershed_markers(gray, fg_mask, bg_mask):
    """Marker-controlled watershed on the gradient magnitude: only the given
    foreground / background markers may be regional minima, which removes
    the over-segmentation. Returns a 0/255 foreground mask."""
    from skimage.segmentation import watershed
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    markers = np.zeros(gray.shape, np.int32)
    markers[bg_mask > 0] = 1
    markers[fg_mask > 0] = 2
    labels = watershed(grad, markers)
    return ((labels == 2) * 255).astype(np.uint8)


def iou(mask_a, mask_b):
    a, b = mask_a > 0, mask_b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def dice(mask_a, mask_b):
    a, b = mask_a > 0, mask_b > 0
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 1.0
