"""Algorithms adapted from coursework/assignment6/exercise6.py.

Original implementations: Matej Gaberc. Portfolio changes: see CHANGELOG.md.
"""
import numpy as np
import cv2


def dual_pca(X, eps=1e-12):
    """Dual covariance PCA. Columns are samples; retain nonzero components."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError("PCA requires a finite matrix with at least two samples")
    mean = np.mean(X, axis=1, keepdims=True)
    Xd = X - mean
    N = X.shape[1]
    C_dual = (Xd.T @ Xd) / (N - 1)
    Uq, S, _ = np.linalg.svd(C_dual)
    keep = S > eps * max(float(S[0]), 1.0)
    S, Uq = S[keep], Uq[:, keep]
    U = (Xd @ Uq) / np.sqrt(S * (N - 1))
    # Fix arbitrary signs for stable exported eigenface plots.
    if U.shape[1]:
        signs = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])
        U *= signs
    return U, S, mean
