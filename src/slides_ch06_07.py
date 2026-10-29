"""From-scratch morphology that follows the Chap 06-07 slides beyond the basic erode/dilate/open/close in
slides_ch09_11: hit-or-miss, boundary extraction, hole filling, connected components, thinning, skeleton,
morphological reconstruction (opening by reconstruction, border clearing) and the grayscale operators
(gradient, top-hat / bottom-hat, granulometry).

Binary images are boolean arrays here and everything outside the image counts as background (the convention
the textbook algorithms assume), unlike slides_ch09_11.erode which follows OpenCV (outside = object).
"""
import numpy as np

import slides_ch09_11 as s9

SQ3 = np.ones((3, 3), bool)
CROSS3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def erode_b(a, se):
    out = np.ones_like(a, bool)
    for dy, dx in s9._offsets(se):
        out &= s9._shifted(a, -dy, -dx, False)
    return out


def dilate_b(a, se):
    out = np.zeros_like(a, bool)
    for dy, dx in s9._offsets(se):
        out |= s9._shifted(a, dy, dx, False)
    return out


def opening_b(a, se):
    return dilate_b(erode_b(a, se), se)


def hit_or_miss(a, b_fg, b_bg):
    """A (*) (B1, B2) = (A (-) B1) AND (A^c (-) B2): B1 must fit in the object, B2 in the background."""
    return erode_b(a, b_fg) & erode_b(~a, b_bg)


def boundary(a, se=SQ3):
    """beta(A) = A - (A (-) B)."""
    return a & ~erode_b(a, se)


def fill_holes_from_seeds(a, seeds, se=CROSS3):
    """Slide 30-31: X_k = (X_{k-1} (+) B) AND A^c, started from one point inside each hole; result = A OR X."""
    x = seeds & ~a
    while True:
        nx = dilate_b(x, se) & ~a
        if (nx == x).all():
            return a | x
        x = nx


def reconstruct_dilation(marker, mask, se=SQ3):
    """Geodesic dilation: repeat f <- (f (+) B) AND g until stable (slides 61-65)."""
    f = marker & mask
    while True:
        nf = dilate_b(f, se) & mask
        if (nf == f).all():
            return f
        f = nf


def _border(a):
    b = np.zeros_like(a, bool)
    b[[0, -1], :] = True
    b[:, [0, -1]] = True
    return b


def fill_holes(a):
    """Slide 70: marker = 1 - I on the image border, 0 elsewhere; H = [R_{I^c}(marker)]^c."""
    return ~reconstruct_dilation(_border(a) & ~a, ~a, CROSS3)


def clear_border(a):
    """Slide 73: remove objects touching the border = A - R_A(border marker)."""
    return a & ~reconstruct_dilation(_border(a) & a, a, SQ3)


def opening_by_reconstruction(a, se_erode, se_recon=SQ3):
    """Slide 68: erode to remove small stuff, then reconstruct the objects that survived with their true shape."""
    return reconstruct_dilation(erode_b(a, se_erode), a, se_recon)


def label_components(a, se=SQ3):
    """Slide 34-35: grow X_k = (X_{k-1} (+) B) AND A from a seed until stable; repeat for the next unlabelled pixel."""
    labels = np.zeros(a.shape, np.int32)
    n = 0
    remaining = a.copy()
    while remaining.any():
        y, x = np.argwhere(remaining)[0]
        seed = np.zeros_like(a, bool)
        seed[y, x] = True
        comp = reconstruct_dilation(seed, a, se)
        n += 1
        labels[comp] = n
        remaining &= ~comp
    return labels, n


def _thin_pairs():
    """(foreground, background) element pairs for thinning: the 8 rotations of the edge and corner patterns."""
    edge_fg = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], bool)
    edge_bg = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], bool)
    corner_fg = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], bool)
    corner_bg = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], bool)
    pairs = []
    for k in range(4):
        pairs.append((np.rot90(edge_fg, k), np.rot90(edge_bg, k)))
        pairs.append((np.rot90(corner_fg, k), np.rot90(corner_bg, k)))
    return pairs


def thinning(a, max_iter=100):
    """A (x) {B} = A - (A (*) B), cycling through the 8 element pairs until nothing changes (slides 42-44)."""
    x = a.copy()
    for _ in range(max_iter):
        before = x.copy()
        for fg, bg in _thin_pairs():
            x &= ~hit_or_miss(x, fg, bg)
        if (x == before).all():
            break
    return x


def skeleton_lantuejoul(a, se=CROSS3):
    """Slides 47-49: S(A) = union_k [(A (-) kB) - (A (-) kB) o B]. Returns (skeleton, per-k subsets)."""
    subsets, e = [], a.copy()
    while e.any():
        subsets.append(e & ~opening_b(e, se))
        e = erode_b(e, se)
    sk = np.zeros_like(a, bool)
    for s in subsets:
        sk |= s
    return sk, subsets


def reconstruct_from_skeleton(subsets, se=CROSS3):
    """Slide 52: A = union_k (S_k (+) kB)."""
    out = np.zeros_like(subsets[0], bool)
    for k, s in enumerate(subsets):
        d = s.copy()
        for _ in range(k):
            d = dilate_b(d, se)
        out |= d
    return out


# --- grayscale (slides 79-106) -------------------------------------------------------------
def gray_erode(f, se):
    out = np.full(f.shape, np.inf, np.float64)
    for dy, dx in s9._offsets(se):
        out = np.minimum(out, s9._shifted(f.astype(np.float64), -dy, -dx, np.inf))
    return out


def gray_dilate(f, se):
    out = np.full(f.shape, -np.inf, np.float64)
    for dy, dx in s9._offsets(se):
        out = np.maximum(out, s9._shifted(f.astype(np.float64), dy, dx, -np.inf))
    return out


def gray_opening(f, se):
    return gray_dilate(gray_erode(f, se), se)


def gray_closing(f, se):
    return gray_erode(gray_dilate(f, se), se)


def morph_gradient(f, se):
    return gray_dilate(f, se) - gray_erode(f, se)


def top_hat(f, se):
    return f - gray_opening(f, se)


def bottom_hat(f, se):
    return gray_closing(f, se) - f


def granulometry(f, radii):
    """Slide 96: surface area (sum of intensities) of the image opened with SEs of growing size; the drop between
    consecutive sizes is the amount of 'particles' of that size."""
    areas = [float(f.astype(np.float64).sum())] + [float(gray_opening(f, s9.disk(r)).sum()) for r in radii]
    return np.array(areas), -np.diff(areas)
