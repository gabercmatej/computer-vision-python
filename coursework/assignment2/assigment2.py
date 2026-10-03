import cv2
from matplotlib import pyplot as plt
import numpy as np
from assignment2.a2_utils import *
import os

# EXERCISE 1
# 1a
f = np.array([0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0])
k = np.array([0.5, 1, 0.3])
k_flipped = k[::-1]
result = np.zeros(len(f) + len(k) - 1)
for n in range(len(result)):
    conv_sum = 0
    for m in range(len(f)):
        if 0 <= n - m < len(k):
            conv_sum += f[m] * k[n - m]
        result[n] = conv_sum
# print("Convolution:", result)

# 1b
def simple_convolution(I, k):
    N = len(k) // 2
    result = np.zeros(len(I))
    k_flipped = k[::-1]
    for i in range(N, len(I) - N - 1):
        conv_sum = 0
        for j in range(len(k_flipped)):
            conv_sum += k_flipped[j] * I[i - N + j]
        result[i] = conv_sum
    return result

signal = read_data("signal.txt")
kernel = read_data("kernel.txt")
conv_result = simple_convolution(signal, kernel)
filtered = cv2.filter2D(signal.reshape(1, -1), -1, kernel.reshape(1, -1)).flatten()
"""
plt.figure(figsize=(10, 4))
plt.plot(signal, label='Original')
plt.plot(kernel, label='Kernel')
plt.plot(conv_result, label='Result')
plt.plot(filtered, label='cv2')
plt.legend()
plt.show()
"""

# 1c
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

conv_result2 = simple_convolution_fix(signal, kernel)
"""
plt.figure(figsize=(10, 4))
plt.plot(signal, label='Original')
plt.plot(kernel, label='Kernel')
plt.plot(conv_result, label='Result')
plt.plot(filtered, label='cv2')
plt.legend()
plt.show()
"""

# 1d

def gauss(sigma):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

sigmas = [0.5, 1, 2, 3, 4]
"""
plt.figure(figsize=(8, 4))
for sigma in sigmas:
    kernel = gauss(sigma)
    radius = len(kernel) // 2
    x = np.arange(-radius, radius + 1)
    plt.plot(x, kernel, label=f'sigma ={sigma}')
plt.legend()
plt.show()
"""

# 1e
k1 = gauss(2)
k2 = np.array([0.1, 0.6, 0.4])
conv1_2 = simple_convolution_fix(simple_convolution_fix(signal, k1), k2)
conv2_1 = simple_convolution_fix(simple_convolution_fix(signal, k2), k1)
k3 = simple_convolution_fix(k1, k2)
conv_3 = simple_convolution_fix(signal, k3)
"""
_, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].plot(signal)
axs[0].set_title("s")
axs[1].plot(conv1_2)
axs[1].set_title("(s * k1) * k2")
axs[2].plot(conv2_1)
axs[2].set_title("(s * k2) * k1")
axs[3].plot(conv_3)
axs[3].set_title("s * (k1 * k2)")
plt.show()
"""

# EXERCISE 2
# 2a
def gaussfilter(image, sigma):
    g = gauss(sigma)
    g_row = g.reshape(1, -1)
    img_rows_filtered = cv2.filter2D(image, -1, g_row)
    g_col = g.T
    img_filtered = cv2.filter2D(img_rows_filtered, -1, g_col)
    return img_filtered

img = cv2.imread("images/lena.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
img_gauss_noise = gauss_noise(img, magnitude=0.1)
img_sp_noise = sp_noise(img, percent=0.1)
sigma = 2
img_gauss_denoised = gaussfilter(img_gauss_noise, sigma)
img_sp_denoised = gaussfilter(img_sp_noise, sigma)
"""
plt.figure(figsize=(12, 8))
plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(2,3,2)
plt.imshow(img_gauss_noise, cmap='gray')
plt.title("Gaussian Noise")
plt.subplot(2,3,3)
plt.imshow(img_sp_noise, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.subplot(2,3,5)
plt.imshow(img_gauss_denoised, cmap='gray')
plt.title("Filtered Gaussian noise")
plt.subplot(2,3,6)
plt.imshow(img_sp_denoised, cmap='gray')
plt.title("Filtered S.P. noise")
plt.show()
"""

# 2b
img2 = cv2.imread("images/fox.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
# img2_blurred = gaussfilter(img2, 2)
# high_frequencies = img2 - img2_blurred
# img_sharpened = np.clip(img2 + high_frequencies, 0, 1)
sharpening_kernel = np.array([[0,0,0],[0,2,0],[0,0,0]]) - (1/9) * np.ones((3,3))
img_sharpened = cv2.filter2D(img2, -1, sharpening_kernel)
img_sharpened = np.clip(img_sharpened, 0, 1)
"""
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img2, cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(img_sharpened, cmap='gray')
plt.title("Sharpened")
plt.show()
"""

# 2c
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

signal = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=float)
signal2 = np.zeros(40).astype(np.float64)
signal2[10:20] = 1
corrupted_signal2 = np.copy(signal2)
corrupted_signal2[13] = 0 #p
corrupted_signal2[17] = 3 #s
signal2_gauss = simple_convolution_fix(corrupted_signal2, gauss(2))
signal2_median1 = simple_median(corrupted_signal2, 2) # widnow too small [0,1] => 0.5
signal2_median2 = simple_median(corrupted_signal2, 3) # window big enough [1,0,1] => 1
"""
_, axs = plt.subplots(1, 5, figsize=(16, 4))
axs[0].plot(signal2)
axs[0].set_title("Original")
axs[1].plot(corrupted_signal2)
axs[1].set_title("Corrupted signal")
axs[2].plot(signal2_gauss)
axs[2].set_title("Gauss")
axs[3].plot(signal2_median1 )
axs[3].set_title("Median1")
axs[4].plot(signal2_median2 )
axs[4].set_title("Median2")
plt.show()
"""

# 2d
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

img = cv2.imread("images/lena.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
img_gauss_noise = gauss_noise(img, magnitude=0.1)
img_sp_noise = sp_noise(img, percent=0.1)
window_size = 3
img_gauss_denoised = medianfilter(img_gauss_noise, window_size)
img_sp_denoised = medianfilter(img_sp_noise, window_size)
"""
plt.figure(figsize=(12, 8))
plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(2,3,2)
plt.imshow(img_gauss_noise, cmap='gray')
plt.title("Gaussian Noise")
plt.subplot(2,3,3)
plt.imshow(img_sp_noise, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.subplot(2,3,5)
plt.imshow(img_gauss_denoised, cmap='gray')
plt.title("Filtered Gaussian noise")
plt.subplot(2,3,6)
plt.imshow(img_sp_denoised, cmap='gray')
plt.title("Filtered S.P. noise")
plt.show()
"""

#EXERCISE 3
# 3a
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

# 3b
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

# 3c
img1_rgb = cv2.cvtColor(cv2.imread("dataset/object_01_1.png"), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
img2_rgb = cv2.cvtColor(cv2.imread("dataset/object_02_1.png"), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
img3_rgb = cv2.cvtColor(cv2.imread("dataset/object_03_1.png"), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
n_bins = 8
h1 = myhist3(img1_rgb, n_bins).reshape(-1)
h2 = myhist3(img2_rgb, n_bins).reshape(-1)
h3 = myhist3(img3_rgb, n_bins).reshape(-1)
"""
_, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0, 0].imshow(img1_rgb)
axs[0, 0].set_title("object_01_1")
axs[0, 1].imshow(img2_rgb)
axs[0, 1].set_title("object_02_1")
axs[0, 2].imshow(img3_rgb)
axs[0, 2].set_title("object_03_1")
axs[1, 0].bar(np.arange(len(h1)), h1)
axs[1, 0].set_ylim(0, 1)
axs[1, 1].bar(np.arange(len(h2)), h2)
axs[1, 1].set_ylim(0, 1)
axs[1, 2].bar(np.arange(len(h3)), h3)
axs[1, 2].set_ylim(0, 1)
plt.show()
"""
dist_1_2 = compare_histograms(h1, h2, method="L2")
dist_1_3 = compare_histograms(h1, h3, method="L2")
"""
print("L2 distance between object 01 and 02:", dist_1_2)
print("L2 distance between object 01 and 03:", dist_1_3)
"""

# 3d
def build_histograms(image_dir, n_bins):
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png'))])
    histograms = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(os.path.join(image_dir, f)), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        h = myhist3(img, n_bins).reshape(-1)
        histograms.append(h)
    histograms = np.array(histograms)
    np.save(os.path.join(image_dir, f"histograms_{n_bins}bins.npy"), histograms)
    return histograms

def retrieve_similar(image_dir, image, n_bins, method, top_k):
    histograms = np.load(os.path.join(image_dir, f"histograms_{n_bins}bins.npy"))
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png'))])
    reference_idx = files.index(image)
    reference_histogram = histograms[reference_idx]
    distances = []
    for i, h in enumerate(histograms):
        if i == reference_idx:  # skip reference image
            continue
        distance = compare_histograms(reference_histogram, h, method=method)
        distances.append((i, distance))
    distances.sort(key=lambda x: x[1])
    top_indices = [idx for idx, _ in distances[:top_k]]
    top_distances = [d for _, d in distances[:top_k]]
    return top_indices, top_distances

image_dir = "dataset"
n_bins = 8
hist_file = os.path.join(image_dir, f"histograms_{n_bins}bins.npy")
if not os.path.exists(hist_file):
    histograms = build_histograms(image_dir, n_bins)
else:
    histograms = np.load(hist_file)
files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png'))])

image = "object_05_4.png"
method = "Hellinger" # L2 Chi2 Intersection Hellinger
top_k = 5
top_indices, top_distances = retrieve_similar(image_dir, image, n_bins, method, top_k)
ref_idx = files.index(image)
ref_hist = histograms[ref_idx]
"""
fig, axs = plt.subplots(2, 1 + top_k, figsize=(3 * (1 + top_k), 6))
axs[0, 0].imshow(cv2.cvtColor(cv2.imread(os.path.join(image_dir, image)), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0)
axs[0, 0].set_title(image)
axs[1, 0].bar(np.arange(len(ref_hist)), ref_hist)
axs[1, 0].set_ylim(0, 1)
axs[1, 0].set_title(f"{method} = 0.000")
for j, (idx, dist) in enumerate(zip(top_indices, top_distances)):
    img_sim = cv2.cvtColor(cv2.imread(os.path.join(image_dir, files[idx])), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    hist_sim = histograms[idx]
    axs[0, j + 1].imshow(img_sim)
    axs[0, j + 1].set_title(f"{files[idx]}")
    axs[1, j + 1].bar(np.arange(len(hist_sim)), hist_sim)
    axs[1, j + 1].set_ylim(0, 1)
    axs[1, j + 1].set_title(f"{method} = {dist:.4f}")
plt.show()
"""

# 3e
image = "object_05_4.png"
method = "Hellinger"
ref_idx = files.index(image)
ref_hist = histograms[ref_idx]
top_indices, top_distances = retrieve_similar(image_dir, image, n_bins, method, top_k)
top_indices = [ref_idx] + top_indices[:-1]
top_distances = [0.000] + top_distances[:-1]
distances = []
for h in histograms:
    distances.append(compare_histograms(ref_hist, h, method=method))
distances = np.array(distances)
sorted_indices = np.argsort(distances)
"""
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(files)), distances)
for idx in top_indices:
    plt.plot(idx, distances[idx], 'ro', markersize=2)
plt.title("Unsorted")
plt.subplot(1, 2, 2)
plt.plot(range(len(files)), distances[sorted_indices])
sorted_top_positions = [np.where(sorted_indices == idx)[0][0] for idx in top_indices]
for pos in sorted_top_positions:
    plt.plot(pos, distances[sorted_indices][pos], 'ro', markersize=2)
plt.title("Sorted")
plt.show()
"""

# 3f
n_bins = 8
histograms_unweighted = histograms.copy()
combined_hist = np.sum(histograms_unweighted, axis=0)
F = combined_hist / np.sum(combined_hist)
landa = 5.0
weights = np.exp(-landa * F)
histograms_weighted = histograms_unweighted * weights
for i in range(histograms_weighted.shape[0]):
    histograms_weighted[i] /= histograms_weighted[i].sum()
methods = ["L2", "Chi2", "Intersection", "Hellinger"]
"""
fig, axs = plt.subplots(len(methods), 2, figsize=(10, 3*len(methods)))
for i, method in enumerate(methods):
    distances_unweighted = np.array([compare_histograms(ref_hist, h, method) for h in histograms_unweighted])
    distances_weighted = np.array([compare_histograms(histograms_weighted[ref_idx], h, method) for h in histograms_weighted])
    axs[i, 0].plot(range(len(files)), distances_unweighted)
    axs[i, 0].set_title(f"{method} - Unweighted")
    axs[i, 1].plot(range(len(files)), distances_weighted)
    axs[i, 1].set_title(f"{method} - Weighted")
    for idx in top_indices:
        axs[i, 0].plot(idx, distances_unweighted[idx], 'ro', markersize=2)
        axs[i, 1].plot(idx, distances_weighted[idx], 'ro', markersize=2)
plt.tight_layout()
plt.show()
"""