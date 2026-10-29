"""Tonghop 3.3: content analysis. Spectral-residual saliency picks the likely
subject, and its bounding box seeds GrabCut for the segmentation."""
import cv2
import numpy as np


def spectral_residual_saliency(gray, resize_to=64, smooth_sigma=3.0):
    """Hou & Zhang (2007): the log-amplitude spectrum minus its local average
    marks surprising structure; inverse FFT with the original phase gives the map."""
    h, w = gray.shape
    small = cv2.resize(gray, (resize_to, resize_to), interpolation=cv2.INTER_AREA).astype(np.float32)
    F = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(F) + 1e-8)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))

    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(F)))) ** 2
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=smooth_sigma)
    saliency = np.clip(cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC), 0, None)
    saliency -= saliency.min()
    return saliency / saliency.max() if saliency.max() > 0 else saliency


def saliency_to_bbox(saliency, percentile=75, margin=0.08, rel_area_thresh=0.25):
    """Padded box around the connected components above the saliency percentile whose
    area is at least rel_area_thresh of the largest. Taking all comparably sized
    components (not just the largest) covers objects whose saliency splits into
    several edge fragments, while small isolated specks are ignored."""
    h, w = saliency.shape
    binary = (saliency >= np.percentile(saliency, percentile)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.nonzero(areas >= areas.max() * rel_area_thresh)[0]
    left, top = stats[keep, cv2.CC_STAT_LEFT], stats[keep, cv2.CC_STAT_TOP]
    x0, y0 = int(left.min()), int(top.min())
    x1 = int((left + stats[keep, cv2.CC_STAT_WIDTH]).max())
    y1 = int((top + stats[keep, cv2.CC_STAT_HEIGHT]).max())
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x0 - mx, 0), max(y0 - my, 0)
    x1, y1 = min(x1 + mx, w - 1), min(y1 + my, h - 1)
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def segment_grabcut(img_rgb, rect, iterations=8):
    """GrabCut seeded by a rectangle; returns a 0/255 foreground mask."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def draw_content_overlay(img_rgb, mask, color=(255, 80, 200), alpha=0.45):
    tinted = (img_rgb.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
    overlay = img_rgb.copy()
    overlay[mask > 0] = tinted[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay
