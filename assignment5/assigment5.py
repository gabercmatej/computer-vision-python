import cv2
from matplotlib import pyplot as plt
import numpy as np
from a5_utils import *

# EXERCISE 1
# 1b
f = 0.0025 # 2.5 mm
T = 0.12 # 12 cm
pz = np.linspace(0.2, 10, 500) # distances from 0.2 m to 10 m
d = (f * T) / pz
"""
plt.figure()
plt.plot(pz, d)
plt.title('Disparity based of object distance')
plt.show()
"""

# 1d
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
            for d in range(1, min(max_disp, x-half)):
                xr = x - d
                right_patch = imgR[y-half:y+half, xr-half:xr+half]
                ncc_score = ncc(left_patch, right_patch)
                if ncc_score > best_ncc:
                    best_ncc = ncc_score
                    best_disparity = d
            disparity[y, x] = best_disparity
    return disparity

imgL = cv2.imread("data/disparity/office_left.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("data/disparity/office_right.png", cv2.IMREAD_GRAYSCALE)
imgL = cv2.resize(imgL, None, fx=0.5, fy=0.5)
imgR = cv2.resize(imgR, None, fx=0.5, fy=0.5)
imgL = imgL.astype(np.float64)
imgR = imgR.astype(np.float64)
disp = compute_disparity_ncc(imgL, imgR, patch_size=10, max_disp=80)
disp_img = (disp / np.max(disp) * 255).astype(np.uint8)
cv2.imwrite("1d_disparity_ncc.png", disp_img)
"""
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(imgL, cmap='gray')
ax[1].imshow(disp, cmap='gray')
plt.show()
"""

# 1e
disp_LR = compute_disparity_ncc(imgL, imgR, patch_size=10, max_disp=80)
disp_RL = compute_disparity_ncc(imgR, imgL, patch_size=10, max_disp=80)
def left_right_consistency(disp_LR, disp_RL, threshold=1):
    h, w = disp_LR.shape
    disp_consistent = np.zeros_like(disp_LR)
    for y in range(h):
        for x in range(w):
            d = int(disp_LR[y, x])
            xr = x - d
            if xr >= 0 and xr < w:
                if abs(d - disp_RL[y, xr]) <= threshold:
                    disp_consistent[y, x] = d
    return disp_consistent

def warp_with_disparity(img, disp):
    h, w = img.shape
    warped = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            d = int(disp[y, x])
            if x - d >= 0:
                warped[y, x - d] = img[y, x]
    return warped

consistent_disparities = left_right_consistency(disp_LR, disp_RL)
disp_filtered = cv2.medianBlur(consistent_disparities.astype(np.uint8), 5) # smoothening
warped_L = warp_with_disparity(imgL, disp_filtered)
"""
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Right image")
plt.imshow(imgR, cmap='gray')
plt.subplot(1,3,2)
plt.title("Warped left image")
plt.imshow(warped_L, cmap='gray')
plt.subplot(1,3,3)
plt.title("Absolute difference")
plt.imshow(np.abs(warped_L - imgR), cmap='gray')
plt.show()
"""

# EXERCISE 2
# 2b
"""
def harris_points(I, sigma=1, thresh=0.6):
    alpha = 0.06
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    Sxx = cv2.GaussianBlur(Ix*Ix, (0,0), sigma)
    Syy = cv2.GaussianBlur(Iy*Iy, (0,0), sigma)
    Sxy = cv2.GaussianBlur(Ix*Iy, (0,0), sigma)
    detC = Sxx*Syy - Sxy**2
    traceC = Sxx + Syy
    R = detC - alpha * traceC**2
    R = (R - R.min()) / (R.max() - R.min())
    R_thresh = np.where(R > thresh, R, 0)
    ys, xs = np.nonzero(R_thresh)
    return np.stack([xs, ys], axis=1)

def simple_descriptors(I, ys, xs, window_size=20, n_bins=16):
    desc = []
    h, w = I.shape
    half = window_size // 2
    for x, y in zip(xs, ys):
        patch = I[max(y-half,0):min(y+half,h), max(x-half,0):min(x+half,w)]
        hist, _ = np.histogram(patch, bins=n_bins, range=(0,1), density=True)
        desc.append(hist)
    return np.array(desc)

def hellinger_distance(d1, d2):
    return np.sqrt(0.5 * np.sum((np.sqrt(d1)-np.sqrt(d2))**2))

def find_matches(I1, I2, sigma=1, thresh=0.6):
    points1 = harris_points(I1, sigma, thresh)
    points2 = harris_points(I2, sigma, thresh)
    desc1 = simple_descriptors(I1, points1[:,1], points1[:,0])
    desc2 = simple_descriptors(I2, points2[:,1], points2[:,0])
    left_to_right = [np.argmin([hellinger_distance(d1, d2) for d2 in desc2]) for d1 in desc1]
    right_to_left = [np.argmin([hellinger_distance(d2, d1) for d1 in desc1]) for d2 in desc2]
    matches = []
    for i, j in enumerate(left_to_right):
        if right_to_left[j] == i:
            matches.append([i, j])
    return np.array(matches), points1, points2
"""
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

img1 = cv2.imread("data/house/images/house.007.png")
img2 = cv2.imread("data/house/images/house.008.png")
I1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
I2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
h1, w1 = I1.shape
h2, w2 = I2.shape
# matches, points1, points2 = find_matches(I1, I2, sigma=1, thresh=0.6)
points1 = np.loadtxt("data/house/2D/house.007.corners")
points2 = np.loadtxt("data/house/2D/house.008.corners")
matches = read_matches("data/house/2D/house.nview-corners", 7, 8)
p1 = points1[matches[:, 0]]
p2 = points2[matches[:, 1]]
F = fundamental_matrix(p1, p2)
# print("Fundamental Matrix:", F)
matched_p1 = np.hstack([p1, np.ones((len(p1),1))])
matched_p2 = np.hstack([p2, np.ones((len(p2),1))])
"""
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(I1, cmap='gray')
for i, p2 in enumerate(matched_p2):
    l = F.T @ p2
    draw_epiline(l, h1, w1)
    x, y = matched_p1[i, 0], matched_p1[i, 1]
    plt.plot(x, y, '.', color='r', markersize=10)
plt.title("Epipolar lines in Image 1")
plt.subplot(1,2,2)
plt.imshow(I2, cmap='gray')
for i, p1 in enumerate(matched_p1):
    l = F @ p1
    draw_epiline(l, h2, w2)
    x, y = matched_p2[i, 0], matched_p2[i, 1]
    plt.plot(x, y, '.', color='r', markersize=10)
plt.title("Epipolar lines in Image 2")
plt.show()
"""

# 2c
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

point1 = np.array([[160, 463]])
point2 = np.array([[128, 437]])
single_error = np.mean(reprojection_errors(F, point1, point2))
# print("Reprojection error for single point pair:", single_error)
average_error = np.mean(reprojection_errors(F, p1, p2))
# print("Average reprojection error for all matches:", average_error)

# 2d

# 2e

# EXERCISE 3
# 3a
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

points1 = np.loadtxt("data/house/2D/house.007.corners")
points2 = np.loadtxt("data/house/2D/house.008.corners")
matches = read_matches("data/house/2D/house.nview-corners", 7, 8)
p1 = points1[matches[:, 0]]
p2 = points2[matches[:, 1]]
P1 = np.loadtxt("data/house/3D/house.007.P")
P2 = np.loadtxt("data/house/3D/house.008.P")
points_3D = triangulate(p1, p2, P1, P2)
T = np.array([[-1,0,0],[0,0,-1],[0,1,0]])
points_3D_modified = points_3D @ T.T
"""
fig = plt.figure(figsize=(18,6))
colors = [np.random.rand(3,) for i in range(p1.shape[0])]
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(I1, cmap='gray')
ax1.set_title("Image 1 with matches")
for i in range(p1.shape[0]):
    ax1.plot(p1[i,0], p1[i,1], '.', color=colors[i], markersize=5)
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(I2, cmap='gray')
ax2.set_title("Image 2 with matches")
for i in range(p2.shape[0]):
    ax2.plot(p2[i,0], p2[i,1], '.', color=colors[i], markersize=5)
ax3 = fig.add_subplot(1,3,3, projection='3d')
for i, (x, y, z) in enumerate(points_3D_modified):
    ax3.scatter(x, y, z, color=colors[i], s=30)
    ax3.text(x, y, z, str(i), color=colors[i], fontsize=8)
ax3.set_title("3D triangulation")
plt.tight_layout()
plt.show()
"""

# 3b
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

img1 = cv2.imread("data/house/images/house.007.png")
img2 = cv2.imread("data/house/images/house.008.png")
I1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
I2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
points1 = np.loadtxt("data/house/2D/house.007.corners")
points2 = np.loadtxt("data/house/2D/house.008.corners")
matches = read_matches("data/house/2D/house.nview-corners", 7, 8)
F_ransac, inlier_matches = fundemental_ransac(points1, points2, matches, threshold=1.0, k=50)
p1 = points1[inlier_matches[:, 0]]
p2 = points2[inlier_matches[:, 1]]
P1 = np.loadtxt("data/house/3D/house.007.P")
P2 = np.loadtxt("data/house/3D/house.008.P")
points_3D = triangulate(p1, p2, P1, P2)
T = np.array([
    [-1, 0, 0],
    [ 0, 0,-1],
    [ 0, 1, 0]
])
points_3D_mod = points_3D @ T.T
"""
fig = plt.figure(figsize=(18,6))
colors = [np.random.rand(3,) for i in range(len(p1))]
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(I1, cmap='gray')
ax1.set_title("Image 1 inlier matches")
for i in range(len(p1)):
    ax1.plot(p1[i,0], p1[i,1], '.', color=colors[i], markersize=5)
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(I2, cmap='gray')
ax2.set_title("Image 2 inlier matches")
for i in range(len(p2)):
    ax2.plot(p2[i,0], p2[i,1], '.', color=colors[i], markersize=5)
ax3 = fig.add_subplot(1,3,3, projection='3d')
ax3.set_title("3D triangulation RANSAC inliers")
for i, (x,y,z) in enumerate(points_3D_mod):
    ax3.scatter(x, y, z, color=colors[i], s=30)
    ax3.text(x, y, z, str(i), fontsize=8)
plt.tight_layout()
plt.show()
"""






