import cv2
from matplotlib import pyplot as plt
import numpy as np
from a4_utils import *

# EXERCISE 1
# 1a
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
                neighbor1 = Imag[y-1, x+1]
                neighbor2 = Imag[y+1, x-1]
            elif 67.5 <= angle < 112.5:
                neighbor1 = Imag[y-1, x]
                neighbor2 = Imag[y+1, x]
            elif 112.5 <= angle < 157.5:
                neighbor1 = Imag[y-1, x-1]
                neighbor2 = Imag[y+1, x+1]
            if mag >= neighbor1 and mag >= neighbor2:
                I_nms[y, x] = mag
            else:
                I_nms[y, x] = 0
    return I_nms

# code from assigment 3

def hessian_points(I, sigma, thresh):
    Ixx, Iyy, Ixy = compute_second_derivatives(I, sigma)
    detH = Ixx*Iyy - Ixy**2
    detH = (detH - np.min(detH)) / (np.max(detH) - np.min(detH)) # normalize
    detH_thresh = np.where(detH > thresh, detH, 0)
    I_nms = non_max_suppression(detH_thresh, np.zeros(detH.shape))
    ys, xs = np.nonzero(I_nms > 0)
    return detH, xs, ys

img = cv2.imread("data/graf/graf_a.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
I_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
I = I_g.astype(np.float64) / 255.0
detH1, xs1, ys1 = hessian_points(I, 0.3, 0.6)
detH2, xs2, ys2 = hessian_points(I, 0.7, 0.6)
detH3, xs3, ys3 = hessian_points(I, 3, 0.6)
"""
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.imshow(detH1, cmap="gray")
plt.subplot(2, 3, 2)
plt.imshow(detH2, cmap="gray")
plt.subplot(2, 3, 3)
plt.imshow(detH3, cmap="gray")
plt.subplot(2, 3, 4)
plt.imshow(img)
plt.plot(xs1, ys1, 'rx')
plt.subplot(2, 3, 5)
plt.imshow(img)
plt.plot(xs2, ys2, 'rx')
plt.subplot(2, 3, 6)
plt.imshow(img)
plt.plot(xs3, ys3, 'rx')
plt.show()
"""

# 1b
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
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    R_thresh = np.where(R > thresh, R, 0)
    I_nms = non_max_suppression(R_thresh, np.zeros(R.shape))
    ys, xs = np.nonzero(I_nms > 0)
    return R, xs, ys

R1, xs1, ys1 = harris_points(I, 0.5, 0.6)
R2, xs2, ys2 = harris_points(I, 0.7, 0.4)
R3, xs3, ys3 = harris_points(I, 3, 0.6)
"""
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.imshow(R1, cmap="gray")
plt.subplot(2, 3, 2)
plt.imshow(R2, cmap="gray")
plt.subplot(2, 3, 3)
plt.imshow(R3, cmap="gray")
plt.subplot(2, 3, 4)
plt.imshow(img)
plt.plot(xs1, ys1, 'rx')
plt.subplot(2, 3, 5)
plt.imshow(img)
plt.plot(xs2, ys2, 'rx')
plt.subplot(2, 3, 6)
plt.imshow(img)
plt.plot(xs3, ys3, 'rx')
plt.show()
"""

# EXERCISE 2
# 2a
def hellinger_distance(d1, d2):
    return np.sqrt(0.5 * np.sum((np.sqrt(d1) - np.sqrt(d2))**2))

def find_correspondences(desc1, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        dists = np.array([hellinger_distance(d1, d2) for d2 in desc2])
        j = np.argmin(dists)  # index of most similar descriptor
        matches.append([i, j])
    return np.array(matches)

img1 = cv2.imread("data/graf/graf_a_small.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("data/graf/graf_b_small.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
I1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
I2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
R1, xs1, ys1 = harris_points(I1, sigma=1, thresh=0.6)
R2, xs2, ys2 = harris_points(I2, sigma=1, thresh=0.6)
points1 = np.stack([xs1, ys1], axis=1)  # [x, y]
points2 = np.stack([xs2, ys2], axis=1)
desc1 = simple_descriptors(I1, ys1, xs1, n_bins = 16, window_size = 20, sigma = 1)
desc2 = simple_descriptors(I2, ys2, xs2, n_bins = 16, window_size = 20, sigma = 1)
matches = find_correspondences(desc1, desc2)
"""
display_matches(I1, I2, points1, points2, matches)
plt.show()
"""

# 2b
def find_matches(I1, I2, sigma=1, thresh=0.6):
    R1, xs1, ys1 = harris_points(I1, sigma, thresh)
    R2, xs2, ys2 = harris_points(I2, sigma, thresh)
    points1 = np.stack([xs1, ys1], axis=1)
    points2 = np.stack([xs2, ys2], axis=1)
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

img1 = cv2.imread("data/graf/graf_a_small.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("data/graf/graf_b_small.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
I1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
I2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
matches, points1, points2 = find_matches(I1, I2, sigma=1, thresh=0.6)
"""
display_matches(I1, I2, points1, points2, matches)
plt.show()
"""

# 2c

# 2d
def stabilize_video(video_path, output_path, sigma=1, thresh=0.6):
    capture = cv2.VideoCapture(video_path)
    ret, frame0 = capture.read() # first frame
    frame0 = cv2.resize(frame0, (640, 360))
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    matches0, points0, _ = find_matches(gray0, gray0, sigma, thresh) # kepoints in first frame
    # output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = capture.get(cv2.CAP_PROP_FPS)
    h, w = frame0.shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    out.write(frame0) # write frame
    frame_index = 1
    while True:
        ret, frame = capture.read() #read next frame
        if not ret: # while it can read h
            break
        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        matches, pts1, pts2 = find_matches(gray0, gray, sigma, thresh)  # matches between reference frame and new frame
        source_points = pts2[matches[:, 1]].astype(np.float32)
        destination_points = pts1[matches[:, 0]].astype(np.float32)
        # estimate affine transformation
        M, _ = cv2.estimateAffine2D(source_points, destination_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            M = np.eye(2, 3)
        stabilized = cv2.warpAffine(frame, M, (w, h))
        out.write(stabilized)
        frame_index += 1
    capture.release()
    out.release()

def stabilize_video(video_path, output_path, sigma=1, thresh=0.6):
    capture = cv2.VideoCapture(video_path)
    ret, frame0 = capture.read() # first frame
    frame0 = cv2.resize(frame0, (640, 360)) # downsample
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None) # keypoints and descriptors in first frame using sift
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # brute force matcher
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # video codec
    fps = capture.get(cv2.CAP_PROP_FPS) # video framerate
    h, w = frame0.shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    out.write(frame0)
    while True:
        ret, frame = capture.read()
        if not ret: # ret = reading frame was successful
            break
        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None) # keypoints and descriptors
        if des is not None and len(des) > 0:
            matches = bf.match(des0, des)
            matches = sorted(matches, key=lambda x: x.distance) # finding and sorting matches by quality
            source_points = np.float32([kp[m.trainIdx].pt for m in matches]) # find coordinates
            destination_points = np.float32([kp0[m.queryIdx].pt for m in matches])
            M, _ = cv2.estimateAffine2D(source_points, destination_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is None:
                M = np.eye(2, 3) # M = translating, scaling, rotating matrix
            stabilized = cv2.warpAffine(frame, M, (w, h))
        else:
            stabilized = frame # if descriptors are not found, use original frame
        out.write(stabilized) # write frame
    capture.release()
    out.release()

"""
stabilize_video("input_video.mp4", "output_stabilized_video.mp4")
"""

# EXERCISE 3
# 3a
def estimate_homography(points):
    num_pairs = points.shape[0]
    A = np.zeros((2*num_pairs, 9))
    for i in range(num_pairs):
        x1, y1, x2, y2 = points[i]
        A[2*i] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        A[2*i+1] = [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1] / V[-1, -1]  # the last column of V is the eigenvector corresponding to the smallest eigenvalue
    H = h.reshape(3, 3)
    return H

img1 = cv2.imread("data/newyork/newyork_a.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("data/newyork/newyork_b.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
I1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
I2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

points = np.loadtxt("data/newyork/newyork.txt")
points1 = points[:, :2]
points2 = points[:, 2:]
matches = np.arange(len(points1)).reshape(-1, 1)
matches = np.hstack((matches, matches))
# display of the pairs
"""
display_matches(I1, I2, points1, points2, matches)
plt.show()
"""
H = estimate_homography(points)
h2, w2 = I2.shape[:2]
I_warped = cv2.warpPerspective(I1, H, (w2, h2))
# Image wrapping
"""
_, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(I1, cmap='gray')
axs[0].set_title("Newyork a")
axs[1].imshow(I2, cmap='gray')
axs[1].set_title("Newyork b")
axs[2].imshow(I2, cmap='gray')
axs[2].imshow(I_warped, alpha = 0.5, cmap='gray')
axs[2].set_title("Newyork a wrapped to b")
plt.show()
"""

# 3b
def line_fitting_RANSAC(threshold, max_iterations):
    # data generation
    np.random.seed(42)
    noise_scale = 0.1
    start = np.random.random(2)
    end = np.random.random(2)
    a, b = get_line_equation(start, end)  # true line equation with random points
    points = []
    for x in np.linspace(0, 1, num=50): # 50 sample points near the line
        y = a * x + b
        x += (np.random.random() - 0.5) * noise_scale
        y += (np.random.random() - 0.5) * noise_scale
        if y > 0 and y < 1:
            points.append((x, y))
    best_points = None
    max_inliers = 0
    best_inliers = []
    best_outliers = []
    # main RANSAC loop
    for iteration in range(max_iterations):
        fig, ax = plt.subplots()
        inliers = []
        outliers = []
        p1, p2 = random.sample(points, 2) # randomly selects two points (line)
        a_curr, b_curr = get_line_equation(p1, p2)
        for x, y in points:
            dist = np.abs(a_curr * x + b_curr - y) / np.sqrt(a_curr ** 2 + 1)
            if dist <= threshold:
                inliers.append((x, y))
            else:
                outliers.append((x, y))
        # check if it is the best so for
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_points = (p1, p2)
            best_inliers = inliers
            best_outliers = outliers

        # plot for the iteration
        ax.cla()  # clear the plot
        inliers_np = np.array(inliers)
        outliers_np = np.array(outliers)
        for x, y in inliers_np:
            plt.plot(x, y, 'g.')
        for x, y in outliers_np:
            plt.plot(x, y, 'r.')
        # candidate points
        plt.plot(p1[0], p1[1], 'o', color='orange', markersize=4)
        plt.plot(p2[0], p2[1], 'o', color='orange', markersize=4)
        # candidate line
        ax.axline((p1[0], p1[1]), (p2[0], p2[1]), color='orange', linestyle='--', label="Candidate line")
        # best line
        if best_points is not None:
            p1_best, p2_best = best_points
            ax.axline((p1_best[0], p1_best[1]), (p2_best[0], p2_best[1]), color='blue', linestyle='--', label="Best line so far")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.axis('square')
        plt.title(f"Iteration {iteration + 1}")
        plt.legend()
        plt.draw()
        plt.pause(0.3)
        # break if line already good enough
        if len(inliers) / len(points) > 0.9:
            break
    # best line and final points
    ax.cla()
    fig, ax = plt.subplots()
    inliers_np = np.array(best_inliers)
    outliers_np = np.array(best_outliers)
    for x, y in inliers_np:
        plt.plot(x, y, 'g.')
    for x, y in outliers_np:
        plt.plot(x, y, 'r.')
    p1_best, p2_best = best_points
    plt.plot(p1_best[0], p1_best[1], 'b*')
    plt.plot(p2_best[0], p2_best[1], 'b*')
    ax.axline((p1_best[0], p1_best[1]), (p2_best[0], p2_best[1]), color='blue')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.axis('square')
    plt.title(f"Best final line, inlier percentage = {max_inliers / len(points)}")
    plt.show()

"""
line_fitting_RANSAC(0.04, 20)
"""

# 3c
def reprojection_error(H, p1, p2):
    p1_h = np.hstack([p1, np.ones((p1.shape[0], 1))])  # homogeneous
    p2_proj_h = (H @ p1_h.T).T  # apply homography
    p2_proj = p2_proj_h[:, :2] / p2_proj_h[:, 2:3]  # normalize
    errors = np.linalg.norm(p2 - p2_proj, axis=1)  # Euclidean distance
    return errors

def ransac_homography(points1, points2, matches, threshold=3.0, k=1000):
    best_inliers = []
    smallest_error = np.inf
    matches_arr = np.array(matches)
    for i in range(k):
        idx = np.random.choice(len(matches_arr), 4, replace=False)
        selected_matches = matches_arr[idx] # 4 random matches needed for homography
        p1 = points1[selected_matches[:, 0]]
        p2 = points2[selected_matches[:, 1]]
        H = estimate_homography(np.hstack([p1, p2])) # homography for those points
        all_p1 = points1[matches_arr[:, 0]]
        all_p2 = points2[matches_arr[:, 1]]
        errors = reprojection_error(H, all_p1, all_p2) # reprojection error for all matches
        inliers_idx = np.where(errors < threshold)[0]
        inliers_error = errors[inliers_idx].mean() # mean error of inliers
        # update best solution
        if len(inliers_idx) > len(best_inliers) or (len(inliers_idx) == len(best_inliers) and inliers_error < smallest_error):
            best_inliers = inliers_idx
            smallest_error = inliers_error
    # homography for the best solution
    inlier_matches = matches_arr[best_inliers]
    final_p1 = points1[inlier_matches[:, 0]]
    final_p2 = points2[inlier_matches[:, 1]]
    final_H = estimate_homography(np.hstack([final_p1, final_p2]))
    return final_H, inlier_matches

img1 = cv2.imread("data/newyork/newyork_a.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
I1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
img2 = cv2.imread("data/newyork/newyork_b.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
matches, points1, points2 = find_matches(I1, I2, sigma=1, thresh=0.4)
# matches
"""
display_matches(I1, I2, points1, points2, matches)
plt.show()
"""
H_ransac, inlier_matches = ransac_homography(points1, points2, matches, threshold=1.0, k=1000) # k = 50 reliable
# matches after RANSAC
"""
display_matches(I1, I2, points1, points2, inlier_matches)
plt.show()
"""
# warped image
h2, w2 = I2.shape[:2]
warped_img = cv2.warpPerspective(I1, H_ransac, (w2, h2))
"""
plt.imshow(I2, cmap = 'gray')
plt.imshow(warped_img, alpha= 0.5, cmap = 'gray')
plt.title("Image 1 warped to Image 2 using RANSAC homography")
plt.axis('off')
plt.show()
"""

# 3d

# 3e
def warp(img, H):
    h, w = img.shape[:2]
    warped = np.zeros(img.shape)
    H_inv = np.linalg.inv(H) # inverse of H
    for y in range(h):
        for x in range(w):
            p = np.array([x, y, 1]) # homogeneous coordinates
            ps = H_inv @ p
            ps /= ps[2]  # normalize
            xs, ys = int(ps[0]), int(ps[1])
            if 0 <= xs < w and 0 <= ys < h: # check if inside source image boundaries
                warped[y, x] = img[ys, xs]
    return warped

points = np.loadtxt("data/newyork/newyork.txt")
H = estimate_homography(points)
I_warped = warp(I1, H)
I_warped_cv = cv2.warpPerspective(I1, H, (I2.shape[1], I2.shape[0]))
"""
plt.subplot(2, 2, 1)
plt.imshow(I2, cmap='gray')
plt.imshow(I_warped, alpha = 0.5, cmap='gray')
plt.title("My custom warp")
plt.subplot(2, 2, 2)
plt.imshow(I2, cmap='gray')
plt.imshow(I_warped_cv, alpha = 0.5, cmap='gray')
plt.title("cv2 warp")
plt.show()
"""