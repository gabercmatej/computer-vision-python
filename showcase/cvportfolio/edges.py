"""Algorithms adapted from assignment3/assigment3.py.

Original implementations: Matej Gaberc. Portfolio changes: see CHANGELOG.md.
"""
import numpy as np
import cv2


def gauss(sigma):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussdx(sigma):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = -(x / (np.sqrt(2 * np.pi) * sigma**3)) * np.exp(-(x**2) / (2 * sigma**2))
    kernel /= np.sum(np.abs(kernel))
    return kernel


def compute_derivatives(I, sigma):
    G = gauss(sigma).reshape(1, -1)
    D = gaussdx(sigma).reshape(1, -1)
    GT = G.T
    DT = D.T
    Ix = cv2.filter2D(I, -1, GT)
    Ix = cv2.filter2D(Ix, -1, D)
    Iy = cv2.filter2D(I, -1, G)
    Iy = cv2.filter2D(Iy, -1, DT)
    return Ix, Iy


def compute_second_derivatives(I, sigma):
    G = gauss(sigma).reshape(1, -1)
    D = gaussdx(sigma).reshape(1, -1)
    GT = G.T
    DT = D.T
    Ix, Iy = compute_derivatives(I, sigma)
    Ixx = cv2.filter2D(Ix, -1, GT)
    Ixx = cv2.filter2D(Ixx, -1, D)
    Iyy = cv2.filter2D(Iy, -1, G)
    Iyy = cv2.filter2D(Iyy, -1, DT)
    Ixy = cv2.filter2D(Ix, -1, G)
    Ixy = cv2.filter2D(Ixy, -1, DT)
    return Ixx, Iyy, Ixy


def gradient_magnitude(I, sigma):
    Ix, Iy = compute_derivatives(I, sigma)
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    angle = np.arctan2(Iy, Ix)
    return magnitude, angle


def non_max_suppression(Imag, Idir):
    I_nms = np.zeros(Imag.shape, dtype=Imag.dtype)
    height, width = Imag.shape
    angles = np.rad2deg(Idir) % 180
    for y in range(1, height-1):
        for x in range(1, width-1):
            mag = Imag[y, x]
            angle = angles[y, x]
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbor1 = Imag[y, x-1]
                neighbor2 = Imag[y, x+1]
            elif 22.5 <= angle < 67.5:
                neighbor1 = Imag[y-1, x-1]
                neighbor2 = Imag[y+1, x+1]
            elif 67.5 <= angle < 112.5:
                neighbor1 = Imag[y-1, x]
                neighbor2 = Imag[y+1, x]
            elif 112.5 <= angle < 157.5:
                neighbor1 = Imag[y-1, x+1]
                neighbor2 = Imag[y+1, x-1]
            if mag >= neighbor1 and mag >= neighbor2:
                I_nms[y, x] = mag
            else:
                I_nms[y, x] = 0
    return I_nms


def hysteresis(I_nms, tlow, thigh):
    """Keep 8-connected candidate components containing a strong edge."""
    if not 0 < tlow <= thigh:
        raise ValueError("Require 0 < low <= high")
    _, labels = cv2.connectedComponents((I_nms >= tlow).astype(np.uint8), connectivity=8)
    strong_labels = np.unique(labels[I_nms >= thigh])
    strong_labels = strong_labels[strong_labels != 0]
    return np.isin(labels, strong_labels).astype(np.uint8)


def nonmaxima_suppression_box(accumulator, k=9):
    suppressed = np.zeros_like(accumulator)
    ys, xs = np.nonzero(accumulator)
    for y, x in zip(ys, xs):
        y0 = max(0, y - k // 2)
        y1 = min(accumulator.shape[0], y + k // 2 + 1)
        x0 = max(0, x - k // 2)
        x1 = min(accumulator.shape[1], x + k // 2 + 1)
        window = accumulator[y0:y1, x0:x1]
        max_value = np.max(window)
        if accumulator[y, x] == max_value:
            if not np.any(suppressed[y0:y1, x0:x1]):
                suppressed[y, x] = accumulator[y, x]
    return suppressed


def hough_lines_fast(edge_image, bins_rho=180, bins_theta=180):
    height, width = edge_image.shape
    D = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    rhos = np.linspace(-D, D, bins_rho)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, bins_theta)
    accumulator = np.zeros((bins_rho, bins_theta), dtype=np.int32)
    ys, xs = np.nonzero(edge_image)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        r_idx = np.searchsorted(rhos, rho_vals) # index where the value would be in the array
        accumulator[r_idx, np.arange(bins_theta)] += 1
    return accumulator, rhos, thetas


def hough_find_lines(edge_image, bins_rho=180, bins_theta=180, n_lines=10, nms_size=11):
    accumulator, rhos, thetas = hough_lines_fast(edge_image, bins_rho, bins_theta)
    accumulator_nms = nonmaxima_suppression_box(accumulator, k=nms_size)
    r_idx, t_idx = np.nonzero(accumulator_nms)
    lines_with_votes = [(rhos[r], thetas[t], accumulator_nms[r, t]) for r, t in zip(r_idx, t_idx)]
    lines_with_votes.sort(key=lambda x: x[2], reverse=True)
    lines_with_votes = lines_with_votes[:n_lines]
    lines = [(rho, theta) for rho, theta, _ in lines_with_votes]
    return lines, accumulator, rhos, thetas
