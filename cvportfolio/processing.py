"""Algorithms adapted from coursework/assignment1/assigment1.py.

Original implementations: Matej Gaberc. Portfolio changes: see CHANGELOG.md.
"""
import numpy as np
import cv2


def myhist(I_g, n_bins):
    pixels = I_g.reshape(-1)
    H = np.zeros(n_bins)
    interval = 1.0 / n_bins
    for value in pixels:
        index = int(value/interval)
        if index == n_bins: #value = 1.0
            index = n_bins - 1
        H[index] += 1
    H = H / len(pixels)
    return H


def otsu_threshold(image):
    """Deterministic Otsu threshold on 256 bins for intensities in [0, 1]."""
    image = np.asarray(image, dtype=float)
    if image.size == 0 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected nonempty finite intensities in [0, 1]")
    quantized = np.rint(image * 255).astype(np.uint8)
    hist = np.bincount(quantized.ravel(), minlength=256).astype(float)
    probability = hist / hist.sum()
    weights = np.cumsum(probability)
    means = np.cumsum(probability * np.arange(256))
    denom = weights * (1-weights)
    score = np.divide((means[-1]*weights-means)**2, denom,
                      out=np.zeros(256), where=denom > 1e-15)
    return float(np.argmax(score)) / 255


def immask(image, mask):
    mask = mask.astype(np.uint8)
    expanded_mask = np.expand_dims(mask, axis=-1)
    masked_image = image * expanded_mask
    return masked_image
