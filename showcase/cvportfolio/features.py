"""Algorithms adapted from assignment4/assigment4.py.

Original implementations: Matej Gaberc. Portfolio changes: see CHANGELOG.md.
"""
import numpy as np
import cv2
from .edges import gauss, gaussdx
from .course_utils.a4_utils import simple_descriptors, convolve


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


def hessian_points(I, sigma, thresh):
    Ixx, Iyy, Ixy = compute_second_derivatives(I, sigma)
    detH = Ixx*Iyy - Ixy**2
    if np.ptp(detH) <= np.finfo(float).eps:
        return np.zeros_like(detH), np.array([], dtype=int), np.array([], dtype=int)
    detH = (detH - np.min(detH)) / np.ptp(detH)
    detH_thresh = np.where(detH > thresh, detH, 0)
    I_nms = spatial_nonmax(detH_thresh)
    ys, xs = np.nonzero(I_nms > 0)
    return detH, xs, ys


def harris_points(I, sigma, thresh):
    sigma = 1.6 * sigma
    alpha = 0.06
    Ix, Iy = compute_derivatives(I, sigma)
    G = gauss(sigma).reshape(1, -1)
    GT = G.T
    Ix2 = Ix * Ix
    Sxx = convolve(Ix2, G, GT)
    Iy2 = Iy * Iy
    Syy = convolve(Iy2, G, GT)
    Ixy = Ix * Iy
    Sxy = convolve(Ixy, G, GT)
    detC2 = Sxx * Syy - Sxy * Sxy
    traceC2 = (Sxx + Syy) **2
    R = detC2 - alpha * traceC2
    if np.ptp(R) <= np.finfo(float).eps:
        return np.zeros_like(R), np.array([], dtype=int), np.array([], dtype=int)
    R = (R - np.min(R)) / np.ptp(R)
    R_thresh = np.where(R > thresh, R, 0)
    I_nms = spatial_nonmax(R_thresh)
    ys, xs = np.nonzero(I_nms > 0)
    return R, xs, ys


def hellinger_distance(d1, d2):
    return np.sqrt(0.5 * np.sum((np.sqrt(d1) - np.sqrt(d2))**2))


def find_correspondences(desc1, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        dists = np.array([hellinger_distance(d1, d2) for d2 in desc2])
        j = np.argmin(dists)  # index of most similar descriptor
        matches.append([i, j])
    return np.array(matches)


def find_matches(I1, I2, sigma=1, thresh=0.6):
    R1, xs1, ys1 = harris_points(I1, sigma, thresh)
    R2, xs2, ys2 = harris_points(I2, sigma, thresh)
    points1 = np.stack([xs1, ys1], axis=1)
    points2 = np.stack([xs2, ys2], axis=1)
    if not len(points1) or not len(points2):
        return np.empty((0, 2), dtype=int), points1, points2
    desc1 = simple_descriptors(I1, ys1, xs1, n_bins=16, window_size=20, sigma=1)
    desc2 = simple_descriptors(I2, ys2, xs2, n_bins=16, window_size=20, sigma=1)
    left_to_right = []
    for i, d1 in enumerate(desc1):
        dists = np.array([hellinger_distance(d1, d2) for d2 in desc2])
        j = np.argmin(dists)
        left_to_right.append(j)
    right_to_left = []
    for j, d2 in enumerate(desc2):
        dists = np.array([hellinger_distance(d2, d1) for d1 in desc1])
        i = np.argmin(dists)
        right_to_left.append(i)
    matches = []
    for i, j in enumerate(left_to_right):
        if right_to_left[j] == i:  # symmetric
            matches.append([i, j])
    return np.array(matches), points1, points2


def estimate_homography(points):
    """Normalized DLT for paired (x1, y1, x2, y2) coordinates."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Need at least four finite point pairs")
    def normalize(p):
        mean = p.mean(axis=0)
        distance = np.linalg.norm(p - mean, axis=1).mean()
        if distance < 1e-12:
            raise ValueError("Degenerate points")
        scale = np.sqrt(2) / distance
        T = np.array([[scale, 0, -scale*mean[0]], [0, scale, -scale*mean[1]], [0, 0, 1]])
        return (p-mean)*scale, T
    p1, T1 = normalize(points[:, :2]); p2, T2 = normalize(points[:, 2:])
    A = np.zeros((2*len(points), 9))
    for i, ((x1,y1),(x2,y2)) in enumerate(zip(p1,p2)):
        A[2*i] = [x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2]
        A[2*i+1] = [0,0,0,x1,y1,1,-y2*x1,-y2*y1,-y2]
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Point pairs do not determine a homography")
    _, _, VT = np.linalg.svd(A)
    H = np.linalg.inv(T2) @ VT[-1].reshape(3,3) @ T1
    scale = H[2,2] if abs(H[2,2]) > 1e-12 else np.linalg.norm(H)
    return H / scale


def reprojection_error(H, p1, p2):
    p1_h = np.hstack([p1, np.ones((p1.shape[0], 1))])  # homogeneous
    p2_proj_h = (H @ p1_h.T).T  # apply homography
    p2_proj = p2_proj_h[:, :2] / p2_proj_h[:, 2:3]  # normalize
    errors = np.linalg.norm(p2 - p2_proj, axis=1)  # Euclidean distance
    return errors


def ransac_homography(points1, points2, matches, threshold=3.0, k=1000, seed=42):
    """Seeded four-point RANSAC with a consensus refit."""
    matches = np.asarray(matches, dtype=int).reshape(-1,2)
    if len(matches) < 4 or threshold <= 0 or k < 1:
        raise ValueError("Need four matches, positive threshold and iterations")
    rng = np.random.default_rng(seed)
    a, b = points1[matches[:,0]], points2[matches[:,1]]
    best = np.zeros(len(matches), dtype=bool)
    best_error = np.inf
    for _ in range(k):
        selected = rng.choice(len(matches),4,replace=False)
        try:
            H = estimate_homography(np.c_[a[selected], b[selected]])
            errors = reprojection_error(H,a,b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        mask = errors < threshold
        score = errors[mask].mean() if mask.any() else np.inf
        if mask.sum() > best.sum() or (mask.sum() == best.sum() and score < best_error):
            best, best_error = mask, score
    if best.sum() < 4:
        raise ValueError("No valid homography consensus")
    H = estimate_homography(np.c_[a[best],b[best]])
    # Report the actual final model's inliers.
    final_mask = reprojection_error(H,a,b) < threshold
    return H, matches[final_mask]


def spatial_nonmax(response, size=3):
    """Keep positive local maxima in a two-dimensional neighborhood."""
    local_max = cv2.dilate(response, np.ones((size,size), dtype=np.uint8))
    selected = np.where((response == local_max) & (response > 0), response, 0)
    selected[[0,-1],:] = 0
    selected[:,[0,-1]] = 0
    return selected
