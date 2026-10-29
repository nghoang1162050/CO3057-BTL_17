"""From-scratch building blocks that follow the lecture slides, used by
notebooks/CV_Project_1.ipynb.

  Chap 01 - image as a matrix / resolution          (`downsample_resolution`)
  Chap 02 - point processing, histograms, filtering (`negative` ... `subsample`)
  Chap 03 - derivative filters, gradients           (`dog_kernels`, `gradient`)
  Chap 04 - homogeneous transforms, warping         (`translation_matrix` ... `ransac_homography`)

Everything is plain numpy so each formula on the slides maps to a few lines;
OpenCV is only used where the notebook explicitly compares against it.
"""
import cv2
import numpy as np


# --- Chap 02: point processing ---------------------------------------------

def negative(gray):
    """s = L - 1 - r"""
    return (255 - gray).astype(np.uint8)


def log_transform(gray):
    """s = c * log(1 + r), c chosen so that 255 maps to 255: expands dark
    values, compresses bright ones."""
    c = 255.0 / np.log1p(255.0)
    return np.clip(c * np.log1p(gray.astype(np.float32)), 0, 255).astype(np.uint8)


def gamma_correct(gray, gamma):
    """Power law s = 255 * (r/255)^gamma. gamma < 1 brightens, > 1 darkens."""
    return np.clip(255.0 * (gray.astype(np.float32) / 255.0) ** gamma, 0, 255).astype(np.uint8)


def contrast_stretch(gray, low_pct=1, high_pct=99):
    """Linear stretch of the [p_low, p_high] intensity range onto [0, 255]."""
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    if hi <= lo:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def histogram(gray):
    return np.bincount(gray.ravel(), minlength=256)


def cumulative_histogram(gray):
    """Normalized CDF of the intensities, in [0, 1]."""
    cdf = np.cumsum(histogram(gray)).astype(np.float64)
    return cdf / cdf[-1]


def hist_equalize(gray):
    """s = T(r) = round(255 * (CDF(r) - CDF_min) / (1 - CDF_min)): spreads
    the intensities so the histogram becomes (approximately) flat. The
    CDF_min term (same convention as cv2.equalizeHist) makes the darkest
    occupied level map to exactly 0."""
    cdf = cumulative_histogram(gray)
    cdf_min = cdf[cdf > 0].min()
    lut = np.round(255 * (cdf - cdf_min) / max(1 - cdf_min, 1e-12)).clip(0, 255).astype(np.uint8)
    return lut[gray]


# --- Chap 01: resolution ---------------------------------------------------

def downsample_resolution(img, size):
    """Resolution reduction as in the Chap 01 slide: keep a size x size
    sampling of the image, shown back at the original size (nearest
    neighbour) so the pixel blocks are visible."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# --- Chap 02: linear filtering ---------------------------------------------

def correlate2d(img, kernel, border="reflect"):
    """Cross-correlation g[m,n] = sum_{k,l} h[k,l] f[m+k, n+l], same-size
    output. `border` follows the 'practical matters' slide:
    constant (clip/black), wrap (circular), edge (copy), reflect (symmetric)."""
    kernel = np.asarray(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    mode = {"constant": "constant", "wrap": "wrap", "edge": "edge", "reflect": "symmetric"}[border]
    f = img.astype(np.float32)
    pad_width = ((ph, kh - 1 - ph), (pw, kw - 1 - pw)) + ((0, 0),) * (f.ndim - 2)
    padded = np.pad(f, pad_width, mode=mode)
    out = np.zeros_like(f)
    h, w = f.shape[:2]
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def convolve2d(img, kernel, border="reflect"):
    """Convolution = cross-correlation with the kernel flipped in both axes."""
    return correlate2d(img, np.asarray(kernel)[::-1, ::-1], border)


def box_kernel(size):
    return np.ones((size, size), np.float32) / (size * size)


def gaussian_kernel(sigma, size=None):
    """Sampled 2-D Gaussian, normalized to sum 1. Default half-width is
    3 sigma (rule of thumb on the 'choosing kernel width' slide)."""
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    ax = np.arange(size) - size // 2
    g1 = np.exp(-ax ** 2 / (2.0 * sigma ** 2))
    k = np.outer(g1, g1)
    return (k / k.sum()).astype(np.float32)


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SOBEL_Y = SOBEL_X.T.copy()
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)


def subsample(img, factor, prefilter_sigma=None):
    """Keep every `factor`-th row/column. With `prefilter_sigma` the image
    is Gaussian-filtered first (anti-aliasing); without it, fine detail
    aliases into false low frequencies."""
    f = img.astype(np.float32)
    if prefilter_sigma:
        f = correlate2d(f, gaussian_kernel(prefilter_sigma))
    return np.clip(f[::factor, ::factor], 0, 255).astype(np.uint8)


# --- Chap 03: derivatives and gradients --------------------------------------

def dog_kernels(sigma):
    """Derivative-of-Gaussian kernels (x, y): by the derivative theorem of
    convolution, d/dx (f * g) = f * (dg/dx), so smoothing + differentiating
    costs one convolution."""
    g = gaussian_kernel(sigma)
    dx = np.gradient(g, axis=1)
    dy = np.gradient(g, axis=0)
    return dx, dy


def gradient(gray, sigma=None):
    """Return (magnitude, orientation) of the image gradient. With `sigma`
    the derivative-of-Gaussian kernels are used, otherwise plain Sobel.
    Orientation theta = atan2(gy, gx) points to the steepest increase."""
    f = gray.astype(np.float32)
    if sigma:
        kx, ky = dog_kernels(sigma)
    else:
        kx, ky = SOBEL_X, SOBEL_Y
    gx, gy = correlate2d(f, kx), correlate2d(f, ky)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def match_template(gray, template, method="ncc"):
    """Template matching from the slides: 'ssd' returns 1 - normalized
    SSD (high = good), 'corr' the raw correlation, 'ncc' the zero-mean
    normalized cross-correlation (invariant to local brightness and
    contrast). Returns a response map of size (H-h+1, W-w+1)."""
    f, t = gray.astype(np.float32), template.astype(np.float32)
    if method == "ssd":
        r = cv2.matchTemplate(f, t, cv2.TM_SQDIFF)
        return 1.0 - r / r.max()
    if method == "corr":
        return cv2.matchTemplate(f, t, cv2.TM_CCORR)
    return cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)


# --- Chap 04: homogeneous 2-D transforms -------------------------------------

def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def rotation_matrix(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]], np.float64)


def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)


def shear_matrix(shx, shy):
    return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], np.float64)


def about_center(M, shape_hw):
    """Conjugate M with a translation so it acts around the image centre
    instead of the origin (matrix composition T(c) M T(-c))."""
    h, w = shape_hw
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return translation_matrix(cx, cy) @ M @ translation_matrix(-cx, -cy)


def apply_homography(H, pts):
    """Map (N, 2) points through a 3x3 matrix, dividing by w."""
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def warp_forward(img, M, out_shape_hw):
    """Forward warping: push every source pixel to T(x, y). Rounded to the
    nearest destination pixel, so holes appear wherever no pixel lands."""
    h, w = img.shape[:2]
    oh, ow = out_shape_hw
    ys, xs = np.mgrid[0:h, 0:w]
    dst = apply_homography(M, np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64))
    dx, dy = np.round(dst[:, 0]).astype(int), np.round(dst[:, 1]).astype(int)
    ok = (dx >= 0) & (dx < ow) & (dy >= 0) & (dy < oh)
    out = np.zeros((oh, ow) + img.shape[2:], img.dtype)
    out[dy[ok], dx[ok]] = img[ys.ravel()[ok], xs.ravel()[ok]]
    return out


def warp_inverse(img, M, out_shape_hw, interpolation="bilinear"):
    """Inverse warping: for every destination pixel look up T^-1(x', y') in
    the source and interpolate -- no holes, which is why it is preferred."""
    h, w = img.shape[:2]
    oh, ow = out_shape_hw
    ys, xs = np.mgrid[0:oh, 0:ow]
    src = apply_homography(np.linalg.inv(M), np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64))
    sx, sy = src[:, 0], src[:, 1]
    inside = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    f = img.astype(np.float32)
    out = np.zeros((oh * ow,) + img.shape[2:], np.float32)
    if interpolation == "nearest":
        xi, yi = np.round(sx[inside]).astype(int), np.round(sy[inside]).astype(int)
        out[inside] = f[yi, xi]
    else:
        x0, y0 = np.floor(sx[inside]).astype(int), np.floor(sy[inside]).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        ax, ay = sx[inside] - x0, sy[inside] - y0
        if f.ndim == 3:
            ax, ay = ax[:, None], ay[:, None]
        out[inside] = ((1 - ax) * (1 - ay) * f[y0, x0] + ax * (1 - ay) * f[y0, x1]
                       + (1 - ax) * ay * f[y1, x0] + ax * ay * f[y1, x1])
    return np.clip(out.reshape((oh, ow) + img.shape[2:]), 0, 255).astype(np.uint8)


def fit_affine(src, dst):
    """Least-squares 2x3 affine matrix from >= 3 correspondences
    ([x' y']^T = A [x y 1]^T): 6 unknowns, 2 equations per point."""
    A = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return np.vstack([sol.T, [0, 0, 1]])


def _normalize_points(pts):
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return apply_homography(T, pts), T


def fit_homography(src, dst):
    """Direct Linear Transform: each correspondence gives 2 linear
    equations in the 9 entries of H (8 DoF, defined up to scale), so 4
    points are the minimum; with more, the SVD gives the least-squares
    solution. Coordinates are normalized first for numerical stability."""
    ps, Ts = _normalize_points(np.asarray(src, np.float64))
    pd, Td = _normalize_points(np.asarray(dst, np.float64))
    rows = []
    for (x, y), (u, v) in zip(ps, pd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def ransac_homography(src, dst, thresh=3.0, iters=2000, seed=0):
    """RANSAC: fit H to random 4-point samples, keep the model with the
    most inliers (reprojection error < thresh px), then refit on all of
    its inliers. Returns (H, inlier_mask)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, 0
    for _ in range(iters):
        idx = rng.choice(len(src), 4, replace=False)
        try:
            H = fit_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        err = np.linalg.norm(apply_homography(H, src) - dst, axis=1)
        mask = err < thresh
        if mask.sum() > best_count:
            best_mask, best_count = mask, int(mask.sum())
    if best_mask is None or best_count < 4:
        return None, None
    return fit_homography(src[best_mask], dst[best_mask]), best_mask


# ---------------------------------------------------------------- morphing (Chap 04: warping and morphing)
def warp_triangles(img, src_pts, dst_pts, simplices, out_shape):
    """Piecewise-affine warp: every Delaunay triangle of the *destination* mesh is filled from the matching
    source triangle by the affine map fitted on its 3 vertices (inverse warping, so no holes)."""
    import cv2
    out = np.zeros(out_shape, np.float32)
    h, w = out_shape[:2]
    for tri in simplices:
        s = src_pts[tri].astype(np.float32); d = dst_pts[tri].astype(np.float32)
        M = cv2.getAffineTransform(d, s)                                   # dst -> src
        mask = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(mask, np.rint(d).astype(np.int32), 1, cv2.LINE_8)
        warped = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[mask > 0] = warped[mask > 0]
    return out


def morph(imgA, imgB, ptsA, ptsB, t):
    """Frame at time t in [0,1]: interpolate the mesh vertices ((1-t)*A + t*B), warp both images to the
    interpolated mesh with the SAME triangulation, then cross-dissolve."""
    from scipy.spatial import Delaunay
    pts = (1 - t) * ptsA + t * ptsB
    tri = Delaunay(pts).simplices
    wa = warp_triangles(imgA, ptsA, pts, tri, imgA.shape)
    wb = warp_triangles(imgB, ptsB, pts, tri, imgB.shape)
    return np.clip((1 - t) * wa + t * wb, 0, 255).astype(np.uint8)
