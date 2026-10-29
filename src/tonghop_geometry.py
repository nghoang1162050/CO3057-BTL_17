"""Tonghop 3.2: geometric structure (edges, lines, corners)."""
import cv2
import numpy as np


def detect_edges(gray, low_thresh=80, high_thresh=180):
    """Canny edges; high thresholds leave fine texture out."""
    return cv2.Canny(gray, low_thresh, high_thresh)


def detect_lines(edges, min_line_length=70, max_line_gap=6, threshold=90, top_k=25):
    """Probabilistic Hough segments, keeping only the top_k longest (length filters out texture)."""
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    lines = lines.reshape(-1, 4)
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[np.argsort(-lengths)[:top_k]] if len(lines) > top_k else lines


def detect_corners(gray, max_corners=60, quality_level=0.05, min_distance=15):
    """Shi-Tomasi corners: well-spread points where intensity varies in every direction."""
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level,
                                      minDistance=min_distance)
    return np.empty((0, 2)) if corners is None else corners.reshape(-1, 2)


def draw_geometry(img_rgb, lines, corners, line_color=(255, 60, 60), corner_color=(60, 255, 60)):
    out = img_rgb.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 2, cv2.LINE_AA)
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), 4, corner_color, -1, cv2.LINE_AA)
    return out
