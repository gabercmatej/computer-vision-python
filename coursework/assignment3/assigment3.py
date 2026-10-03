import cv2
from matplotlib import pyplot as plt
import numpy as np
from a3_utils import *

# EXERCISE 1
# 1b
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

# 1c
impulse = np.zeros((50, 50))
impulse[25, 25] = 1
sigma = 3
G = gauss(sigma).reshape(1, -1)
D = gaussdx(sigma).reshape(1, -1)
GT = G.T
DT = D.T

a = cv2.filter2D(impulse, -1, G)
a = cv2.filter2D(a, -1, DT)
b = cv2.filter2D(impulse, -1, D)
b = cv2.filter2D(b, -1, GT)
c = cv2.filter2D(impulse, -1, G)
c = cv2.filter2D(c, -1, GT)
d = cv2.filter2D(impulse, -1, GT)
d = cv2.filter2D(d, -1, D)
e = cv2.filter2D(impulse, -1, DT)
e = cv2.filter2D(e, -1, G)
"""
_, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0][0].imshow(impulse, cmap='gray')
axs[0][0].set_title("I")
axs[0][1].imshow(a, cmap='gray')
axs[0][1].set_title("(I * G) * DT")
axs[0][2].imshow(b, cmap='gray')
axs[0][2].set_title("(I * D) * GT")
axs[1][0].imshow(c, cmap='gray')
axs[1][0].set_title("(I * G) * GT")
axs[1][1].imshow(d, cmap='gray')
axs[1][1].set_title("(I * GT) * D")
axs[1][2].imshow(e, cmap='gray')
axs[1][2].set_title("(I * DT) * G")
plt.show()
"""

# 1d
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

I = cv2.imread('images/museum.jpg', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float32) / 255.0
sigma = 2
Ix, Iy = compute_derivatives(I, sigma)
Ixx, Iyy, Ixy = compute_second_derivatives(I, sigma)
Imag, Idir = gradient_magnitude(I, sigma)

hue = (Idir + np.pi) / (2 * np.pi)
saturation = np.ones_like(hue)
value = Imag / Imag.max()
Idir_HSV = np.stack([hue, saturation, value], axis=-1)
Idir_HSV = (Idir_HSV * 255).astype(np.uint8)
Idir_RGB = cv2.cvtColor(Idir_HSV, cv2.COLOR_HSV2RGB)
"""
_, axs = plt.subplots(3, 3, figsize=(14, 14))
axs[0, 0].imshow(I, cmap='gray')
axs[0, 0].set_title("I")
axs[0, 1].imshow(Ix, cmap='gray')
axs[0, 1].set_title("Ix")
axs[0, 2].imshow(Iy, cmap='gray')
axs[0, 2].set_title("Iy")
axs[1, 0].imshow(Ixx, cmap='gray')
axs[1, 0].set_title("Ixx")
axs[1, 1].imshow(Iyy, cmap='gray')
axs[1, 1].set_title("Iyy")
axs[1, 2].imshow(Ixy, cmap='gray')
axs[1, 2].set_title("Ixy")
axs[2, 0].imshow(Imag, cmap='gray')
axs[2, 0].set_title("Imag")
axs[2, 1].imshow(Idir, cmap='gray')
axs[2, 1].set_title("Idir")
axs[2, 2].imshow(Idir_RGB)
axs[2, 2].set_title("Idir(HSV)")
plt.tight_layout()
plt.show()
"""

# 1e
def compute_gradient_feature(I, sigma=2, grid_size=(8, 8), num_bins=8):
    Imag, Idir = gradient_magnitude(I, sigma)
    angles = Idir % (2 * np.pi) # 0-2pi
    height, width = I.shape
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    feature_descriptor = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y0, y1 = i * cell_height, (i + 1) * cell_height
            x0, x1 = j * cell_width, (j + 1) * cell_width
            cell_magnitude = Imag[y0:y1, x0:x1]
            cell_angle = angles[y0:y1, x0:x1]
            bin_index = np.floor(cell_angle / (2 * np.pi / num_bins)).astype(int)
            histogram = np.zeros(num_bins)
            for k in range(num_bins):
                histogram[k] = np.sum(cell_magnitude[bin_index == k])
            histogram /= histogram.sum() + 1e-6
            feature_descriptor.extend(histogram)
    return np.array(feature_descriptor)

I = cv2.imread('images/museum.jpg', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float32) / 255.0
feature = compute_gradient_feature(I, sigma=2, grid_size=(8, 8), num_bins=8)
# print("Feature vector shape:", feature.shape) # 8x8x8 = 512
# print("First 16 values:", feature[:16])

# EXERCISE 2
# 2a
def findedges(I, sigma, theta):
    Imag, Idir = gradient_magnitude(I, sigma)
    Ie = np.zeros(Imag.shape, dtype=Imag.dtype)
    Ie[Imag >= theta] = Imag[Imag >= theta]
    return Ie

I = cv2.imread('images/museum.jpg', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float32) / 255.0
thetas = [0.1, 0.15, 0.2, 0.3, 0.4]
"""
plt.figure(figsize=(15, 5))
for idx, th in enumerate(thetas):
    Ie = findedges(I, sigma=1, theta=th)
    plt.subplot(1, len(thetas), idx + 1)
    plt.imshow(Ie, cmap='gray')
    plt.title(f"theta = {th}")
plt.tight_layout()
plt.show()
"""

# 2b
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

sigma = 1
Imag, Idir = gradient_magnitude(I, sigma)
theta = 0.16
Ie = findedges(I, sigma, theta)
Ie_nms = non_max_suppression(Imag, Idir)
"""
_, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(Ie, cmap='gray')
axs[0].set_title("Thresholded edges")
axs[1].imshow(Ie_nms, cmap='gray')
axs[1].set_title("Non-maxima surpressed edges")
plt.show()
"""

# 2c
def hysteresis(I_nms, tlow, thigh):
    strong_edges = (I_nms >= thigh).astype(np.uint8)
    weak_edges = ((I_nms >= tlow) & (I_nms < thigh)).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(weak_edges, connectivity=8)
    I_hyst = np.zeros(I_nms.shape, dtype=np.uint8)
    I_hyst[strong_edges > 0] = 1
    for label in range(1, num_labels):
        mask = (labels == label)
        if np.any(strong_edges[mask]): # če se dotika močnega robu
            I_hyst[mask] = 1
    return I_hyst

sigma = 1
Imag, Idir = gradient_magnitude(I, sigma)
Ie_nms = non_max_suppression(Imag, Idir)
thigh = 0.12
tlow = 0.04
Ie_canny = hysteresis(Ie_nms, tlow, thigh)
"""
_, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0][0].imshow(I, cmap='gray')
axs[0][0].set_title("Original")
axs[0][1].imshow(Ie, cmap='gray')
axs[0][1].set_title(f"Thresholded (thr = {theta})")
axs[1][0].imshow(Ie_nms, cmap='gray')
axs[1][0].set_title(f"Nonmax. surpp. (thr = {theta})")
axs[1][1].imshow(Ie_canny, cmap='gray')
axs[1][1].set_title(f"Hysteris (high = {thigh}, low = {tlow})")
plt.show()
"""

# EXERCISE 3
# 3a
def hough_lines(edge_image, bins_rho=180, bins_theta=180):
    height, width = edge_image.shape
    D = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.linspace(-D, D, bins_rho)
    thetas = np.linspace(-np.pi/2, np.pi/2, bins_theta)
    accumulator = np.zeros((bins_rho, bins_theta), dtype=np.int32)
    ys, xs = np.nonzero(edge_image)
    for x, y in zip(xs, ys):
        for t_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            r_idx = np.argmin(np.abs(rhos - rho))
            accumulator[r_idx, t_idx] += 1  # voting
    return accumulator, rhos, thetas

h, w = 100, 100
points = [(50, 90)]
edge_image = np.zeros((h, w), dtype=np.uint8)
for x, y in points:
    edge_image[y, x] = 1
accumulator, rhos, thetas = hough_lines(edge_image, bins_rho=180, bins_theta=180)
"""
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(accumulator, cmap='gray')
axs[1].imshow(np.zeros((h, w)), cmap='gray')
r_idx, t_idx = np.where(accumulator > 0)
for ri, ti in zip(r_idx, t_idx):
    draw_line(rhos[ri], thetas[ti], h, w, clr='g', linewidth=0.5)
plt.show()
"""

# 3b
def hough_find_lines(edge_image, bins_rho=180, bins_theta=180, threshold=1):
    accumulator, rhos, thetas = hough_lines(edge_image, bins_rho, bins_theta)
    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] >= threshold:
                lines.append((rhos[r_idx], thetas[t_idx]))
    return lines, accumulator, rhos, thetas

h, w = 100, 100
points = [(10, 10), (20, 10)]
synthetic_image = np.zeros((h, w), dtype=np.uint8)
for x, y in points:
    synthetic_image [y, x] = 255
lines, synthetic_accumulator, synthetic_rhos, synthetic_thetas = hough_find_lines(synthetic_image, threshold=2)

I1 = cv2.imread("images/oneline.png", cv2.IMREAD_GRAYSCALE)
I1 = (I1 / 255.0).astype(np.float32)
edge_image1 = cv2.Canny((I1 * 255).astype(np.uint8), 50, 150)
lines1, accumulator1, rhos1, thetas1 = hough_find_lines(edge_image1, bins_rho=180, bins_theta=180, threshold=200)

I2 = cv2.imread("images/rectangle.png", cv2.IMREAD_GRAYSCALE)
I2 = (I2 / 255.0).astype(np.float32)
edge_image2 = cv2.Canny((I2 * 255).astype(np.uint8), 50, 150)
lines2, accumulator2, rhos2, thetas2 = hough_find_lines(edge_image2, bins_rho=180, bins_theta=180, threshold=100)
"""
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(synthetic_accumulator)
axs[1].imshow(accumulator1)
axs[2].imshow(accumulator2)
plt.show()
"""
# 3c
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

accumulator_nms = nonmaxima_suppression_box(accumulator2, k=9)
"""
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(accumulator2)
axs[1].imshow(accumulator_nms)
plt.show()
"""

# 3d
"""
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(synthetic_image, cmap='gray')
axs[0].set_title("Synthetic")
plt.sca(axs[0])  # make axs[0] current
for rho, theta in lines:
    draw_line(rho, theta, h, w, clr='r', linewidth=0.5)
axs[1].imshow(I1, cmap='gray')
axs[1].set_title("oneline.png")
plt.sca(axs[1])
h1, w1 = I1.shape
for rho, theta in lines1:
    draw_line(rho, theta, h1, w1, clr='r', linewidth=0.5)
axs[2].imshow(I2, cmap='gray')
axs[2].set_title("rectangle.png")
plt.sca(axs[2])
h2, w2 = I2.shape
for rho, theta in lines2:
    draw_line(rho, theta, h2, w2, clr='r', linewidth=0.5)
plt.show()
"""

# 3e
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

I3 = cv2.imread('images/bricks.jpg')
I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2RGB)
I3_g = cv2.cvtColor(I3, cv2.COLOR_RGB2GRAY)
I3g_float = I3_g.astype(np.float64) / 255.0

edge_image3 = cv2.Canny((I3g_float * 255).astype(np.uint8), 50, 150)
top_lines3, accumulator3, rhos3, thetas3 = hough_find_lines(edge_image3, n_lines=10, nms_size=15)
"""
fig, axs = plt.subplots(1, 2, figsize=(18, 12))
axs[0].imshow(accumulator3)
axs[0].set_title("bricks.jpg")
axs[1].imshow(I3)
h, w = edge_image3.shape
for rho, theta in top_lines3:
    draw_line(rho, theta, h, w, clr='r', linewidth=2)
plt.tight_layout()
plt.show()
"""
I4 = cv2.imread('images/pier.jpg')
I4 = cv2.cvtColor(I4, cv2.COLOR_BGR2RGB)
I4_g = cv2.cvtColor(I4, cv2.COLOR_RGB2GRAY)
I4g_float = I4_g.astype(np.float64) / 255.0
edge_image4 = cv2.Canny((I4g_float * 255).astype(np.uint8), 50, 150)
top_lines4, accumulator4, rhos4, thetas4 = hough_find_lines(edge_image4, n_lines=10, nms_size=5)
"""
fig, axs = plt.subplots(1, 2, figsize=(18, 12))
axs[0].imshow(accumulator4)
axs[0].set_title("pier.jpg")
axs[1].imshow(I4)
h, w = edge_image4.shape
for rho, theta in top_lines4:
    draw_line(rho, theta, h, w, clr='r', linewidth=1)
plt.tight_layout()
plt.show()
"""
I5 = cv2.imread('images/building.jpg')
I5 = cv2.cvtColor(I5, cv2.COLOR_BGR2RGB)
I5_g = cv2.cvtColor(I5, cv2.COLOR_RGB2GRAY)
I5g_float = I5_g.astype(np.float64) / 255.0
edge_image5 = cv2.Canny((I5g_float * 255).astype(np.uint8), 50, 150)
top_lines5, accumulator5, rhos5, thetas5 = hough_find_lines(edge_image5, n_lines=10, nms_size=13)
"""
fig, axs = plt.subplots(1, 2, figsize=(18, 12))
axs[0].imshow(accumulator5)
axs[0].set_title("building.jpg")
axs[1].imshow(I5)
h, w = edge_image5.shape
for rho, theta in top_lines5:
    draw_line(rho, theta, h, w, clr='r', linewidth=1)
plt.tight_layout()
plt.show()
"""
# 3f

# 3g
def hough_circles(edge_image, radius):
    height, width = edge_image.shape
    accumulator = np.zeros((height, width), dtype=np.int32)
    ys, xs = np.nonzero(edge_image)
    theta = np.deg2rad(np.arange(0, 360))  # 0-2pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for x, y in zip(xs, ys):
        a_vals = x - radius * cos_t
        b_vals = y - radius * sin_t
        a_vals = np.round(a_vals).astype(int)
        b_vals = np.round(b_vals).astype(int)
        mask = (a_vals >= 0) & (a_vals < width) & (b_vals >= 0) & (b_vals < height)
        a_vals = a_vals[mask]
        b_vals = b_vals[mask]
        accumulator[b_vals, a_vals] += 1
    return accumulator

I = cv2.imread('images/eclipse.jpg')
I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I_gray = (I_gray / 255.0 * 255).astype(np.uint8)
edges = cv2.Canny(I_gray, 50, 150)
radius = 48
accumulator = hough_circles(edges, radius)
threshold = np.max(accumulator) * 0.2
centers_y, centers_x = np.where(accumulator >= threshold)
"""
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
axs[0].imshow(accumulator)
axs[1].imshow(I)
for x, y in zip(centers_x, centers_y):
    circle = plt.Circle((x, y), radius, color='g', fill=False, linewidth=2)
    axs[1].add_patch(circle)
plt.show()
"""

# 3h