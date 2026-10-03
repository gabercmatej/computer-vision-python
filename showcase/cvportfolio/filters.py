"""Algorithms adapted from assignment2/assigment2.py.

Original implementations: Matej Gaberc. Portfolio changes: see CHANGELOG.md.
"""
import numpy as np
import cv2


def simple_convolution_fix(I, k):
    N = len(k) // 2
    L = len(I)
    I_padded = np.zeros(L + 2 * N)
    I_padded[N:N + L] = I
    I_padded[:N] = I[N - 1::-1]  # left padding (mirror)
    I_padded[N + L:] = I[-1:-N - 1:-1] # right padding (mirror)
    k_flipped = k[::-1]
    result = np.zeros(L)
    for i in range(L):
        conv_sum = 0
        for j in range(len(k_flipped)):
            conv_sum += k_flipped[j] * I_padded[i + j]
        result[i] = conv_sum
    return result


def gauss(sigma):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussfilter(image, sigma):
    g = gauss(sigma)
    g_row = g.reshape(1, -1)
    img_rows_filtered = cv2.filter2D(image, -1, g_row)
    g_col = g.T
    img_filtered = cv2.filter2D(img_rows_filtered, -1, g_col)
    return img_filtered


def simple_median(I, w):
    N = w // 2
    # I_padded = np.zeros(len(I) + 2 * N)
    # I_padded[N:N + len(I)] = I
    # I_padded[:N] = I[N - 1::-1]  # left padding
    # I_padded[N + len(I):] = I[-1:-N - 1:-1]  # right padding
    I_padded = np.pad(I, N, mode='reflect')
    result = np.zeros(I.shape)
    for i in range(len(I)):
        window = I_padded[i:i + w]
        result[i] = np.median(window)
    return result


def medianfilter(image, window_size):
    # image_uint8 = (image * 255).astype(np.uint8)
    # filtered_uint8 = cv2.medianBlur(image_uint8, ksize=window_size)
    # return filtered_uint8.astype(np.float64) / 255.0
    pad = window_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            filtered[i, j] = np.median(window)
    return filtered


def myhist3(I_rgb, n_bins):
    H = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    R = I_rgb[:, :, 0].reshape(-1)
    G = I_rgb[:, :, 1].reshape(-1)
    B = I_rgb[:, :, 2].reshape(-1)
    interval = 1.0 / n_bins
    """
     for i, r_value in enumerate(R):
         R_idx = int(R[i] / interval)
         G_idx = int(G[i] / interval)
         B_idx = int(B[i] / interval)
         if R_idx == n_bins: R_idx -= 1
         if G_idx == n_bins: G_idx -= 1
         if B_idx == n_bins: B_idx -= 1
         H[R_idx, G_idx, B_idx] += 1
     """
    R_idx = np.floor(R / interval).astype(int)
    G_idx = np.floor(G / interval).astype(int)
    B_idx = np.floor(B / interval).astype(int)
    R_idx[R_idx == n_bins] = n_bins - 1
    G_idx[G_idx == n_bins] = n_bins - 1
    B_idx[B_idx == n_bins] = n_bins - 1
    np.add.at(H, (R_idx, G_idx, B_idx), 1)
    H /= len(R)
    return H


def compare_histograms(h1, h2, method):
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
    constant = 1e-10
    if method == "L2":
        return np.sqrt(np.sum((h1 - h2) ** 2))
    elif method == "Chi2":
        return 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + constant))
    elif method == "Intersection":
        return 1 - np.sum(np.minimum(h1, h2))
    elif method == "Hellinger":
        return np.sqrt(0.5 * np.sum((np.sqrt(h1) - np.sqrt(h2)) ** 2))
    else:
        return "Unknown method"
