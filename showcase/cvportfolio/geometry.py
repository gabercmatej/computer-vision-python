"""Algorithms adapted from assignment5/assigment5.py.

Original implementations: Matej Gaberc. Portfolio changes: see CHANGELOG.md.
"""
import numpy as np
import cv2
from .course_utils.a5_utils import normalize_points


def ncc(p1, p2):
    p1 = p1 - np.mean(p1)
    p2 = p2 - np.mean(p2)
    normalization = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
    if normalization == 0:
        return 0
    return np.sum(p1 * p2) / normalization


def compute_disparity_ncc(imgL, imgR, patch_size=10, max_disp=80):
    h, w = imgL.shape
    half = patch_size // 2
    disparity = np.zeros((h, w))
    for y in range(half, h-half):
        for x in range(half, w-half):
            left_patch = imgL[y-half:y+half, x-half:x+half]
            best_ncc = -1
            best_disparity = 0
            for d in range(0, min(max_disp, x-half) + 1):
                xr = x - d
                right_patch = imgR[y-half:y+half, xr-half:xr+half]
                ncc_score = ncc(left_patch, right_patch)
                if ncc_score > best_ncc:
                    best_ncc = ncc_score
                    best_disparity = d
            disparity[y, x] = best_disparity
    return disparity


def build_matrix_A(pts1, pts2):
    N = pts1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        u1, v1 = pts1[i][:2]
        u2, v2 = pts2[i][:2]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
    return A


def estimate_fundamental_matrix(pts1, pts2):
    A = build_matrix_A(pts1, pts2)
    U, D, V = np.linalg.svd(A)
    F_full = V[-1].reshape(3,3)
    return F_full


def enforce_rank2(F_full):
    U, D, Vt = np.linalg.svd(F_full)
    D_new = np.array([[D[0], 0, 0],[0, D[1], 0],[0, 0, 0]]) # D[2] = 0
    return U @ D_new @ Vt


def fundamental_matrix(pts1, pts2):
    res1, T1 = normalize_points(pts1)
    res2, T2 = normalize_points(pts2)
    F_full = estimate_fundamental_matrix(res1, res2)
    F_rank2 = enforce_rank2(F_full)
    F = T2.T @ F_rank2 @ T1
    return F


def reprojection_errors(F, pts1, pts2):
    points1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    points2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    errors = []
    for p1, p2 in zip(points1, points2):
        l2 = F @ p1  # ax + by + c = 0
        l1 = F.T @ p2
        d1 = np.abs(l2[0] * p2[0] + l2[1] * p2[1] + l2[2]) / np.sqrt(l2[0] ** 2 + l2[1] ** 2) # |a*x0 + by*0 + c| / sqrt(a^2 + b^2)
        d2 = np.abs(l1[0] * p1[0] + l1[1] * p1[1] + l1[2]) / np.sqrt(l1[0] ** 2 + l1[1] ** 2)
        errors.append((d1 + d2) / 2)
    return np.array(errors)


def triangulate(pts1, pts2, P1, P2):
    N = pts1.shape[0]
    points_3D = np.zeros((N, 3))
    for i in range(N):
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1])
        A = np.zeros((4, 4))
        A[0] = x1[0] * P1[2] - P1[0]
        A[1] = x1[1] * P1[2] - P1[1]
        A[2] = x2[0] * P2[2] - P2[0]
        A[3] = x2[1] * P2[2] - P2[1]
        U, D, V = np.linalg.svd(A)
        X = V[-1]
        X = X / X[3] # normalize
        points_3D[i] = X[:3]
    return points_3D


def fundemental_ransac(points1, points2, matches, threshold=1.0, k=50):
    best_inliers = []
    best_F = None
    for i in range(k):
        idx = np.random.choice(len(matches), 8, replace=False)
        sample = matches[idx]
        p1 = points1[sample[:, 0]]
        p2 = points2[sample[:, 1]]
        F = fundamental_matrix(p1, p2)
        all_p1 = points1[matches[:, 0]]
        all_p2 = points2[matches[:, 1]]
        errors = reprojection_errors(F, all_p1, all_p2)
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F
    return best_F, matches[best_inliers]
