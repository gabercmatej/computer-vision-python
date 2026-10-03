import cv2
from matplotlib import pyplot as plt
import numpy as np

# EXERCISE 1
# 1a
I = cv2.imread('images/umbrellas.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#plt.imshow(I)
#plt.show()

#height, width, channels = I.shape
#print(height, width, channels)
#print(I.dtype)

# 1b
I_float = I.astype(np.float64)
I_grayscale = np.sum(I_float, axis=2) / 3
#plt.imshow(I_grayscale, cmap='gray')
#plt.show()

# 1c & d
#cutout = I[130:260, 240:450, 1]
I_cutout = np.copy(I)
I_cutout[130:260, 240:450, :] = 255 - I_cutout[130:260, 240:450, :]
#plt.imshow(I_cutout)
#plt.show()

# 1e
I_g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
I_g_float = I_g.astype(np.float64) / 255 #vrednosti med 0-1
I_g_reduced = I_g_float * 0.3 #vrednosti med 0-0.3
#plt.subplot(1,2,1)
#plt.imshow(I_g_float, cmap='gray', vmin=0, vmax=1)
#plt.subplot(1,2,2)
#plt.imshow(I_g_reduced, cmap='gray', vmin=0, vmax=1)
#plt.show()

# EXERCISE 2
# 2a
I2 = cv2.imread('images/bird.jpg')
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
I2_g = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
I2_g_float = I2_g.astype(np.float64) / 255.0
I_threshold1 = np.copy(I2_g_float)
threshold = 0.3
I_threshold1[I2_g_float < threshold] = 0
I_threshold1[I2_g_float >= threshold] = 1
#plt.imshow(I_threshold1, cmap='gray')
#plt.show()
I_threshold2 = np.copy(I2_g_float)
I_threshold2 = np.where(I2_g_float >= threshold, 1, 0)
#plt.imshow(I_threshold2, cmap='gray')
#plt.show()

# 2b
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

n_bins = 20
#n_bins = 100
H = myhist(I2_g_float, n_bins)
#plt.bar(np.arange(n_bins), H)
#plt.show()

#2c
Im = cv2.imread('images/umbrellas.jpg')
Im = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)
Im_g = cv2.cvtColor(Im, cv2.COLOR_RGB2GRAY)
Im_g_float = Im_g.astype(np.float64) / 255.0
Imb = cv2.imread('images/umbrellas_bright.jpg')
Imb = cv2.cvtColor(Imb, cv2.COLOR_BGR2RGB)
Imb_g = cv2.cvtColor(Imb, cv2.COLOR_RGB2GRAY)
Imb_g_float = Imb_g.astype(np.float64) / 255.0
Imd = cv2.imread('images/umbrellas_dark.jpg')
Imd = cv2.cvtColor(Imd, cv2.COLOR_BGR2RGB)
Imd_g = cv2.cvtColor(Imd, cv2.COLOR_RGB2GRAY)
Imd_g_float = Imd_g.astype(np.float64) / 255.0
#n_bins = 20
n_bins = 100
H1 = myhist(Im_g_float, n_bins)
H2 = myhist(Imb_g_float, n_bins)
H3 = myhist(Imd_g_float, n_bins)
#plt.subplot(1,3,1)
#plt.bar(np.arange(n_bins), H1)
#plt.title('Original')
#plt.subplot(1,3,2)
#plt.bar(np.arange(n_bins), H2)
#plt.title('Brightened')
#plt.subplot(1,3,3)
#plt.bar(np.arange(n_bins), H3)
#plt.title('Darkened')
#plt.show()

# The brightened image's histogram shifts to the right (toward higher intensity values)
# The darkened image's histogram shifts to the left (toward lower intensity values)
# The original image's histogram lies between the two

#2d
def otsu_threshold(image):
    thresholds = np.random.rand(1000)
    intra_variances = []
    for t in thresholds:
        c1 = image[image <= t]
        c2 = image[image > t]
        w1 = len(c1) / len(image)
        w2 = len(c2) / len(image)
        var1 = np.var(c1) if len(c1) > 0 else 0
        var2 = np.var(c2) if len(c2) > 0 else 0
        intra_var = w1 * var1 + w2 * var2
        intra_variances.append(intra_var)
    index = np.argmin(intra_variances)
    return thresholds[index]

I2_mask_otsu = np.where(I2_g_float >= otsu_threshold(I2_g_float), 1, 0)
#plt.imshow(I2_mask_otsu, cmap='gray')
#plt.show()

#EXERCISE 3
# 3a
n = 5
SE = np.ones((n,n))
I_eroded1 = cv2.erode(I_threshold1, SE)
#plt.imshow(I_eroded1, cmap='gray')
#plt.show()
I_dilated1 = cv2.dilate(I_threshold1, SE)
#plt.imshow(I_dilated1, cmap='gray')
#plt.show()

# 3b
I_threshold = np.copy(I2_g_float)
threshold = 0.2
I_threshold[I2_g_float < threshold] = 0
I_threshold[I2_g_float >= threshold] = 1
n = 15
#SE = np.ones((n,n))
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
I_dilated = cv2.dilate(I_threshold, SE)
n2 = 13
#SE2 = np.ones((n2,n2))
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n2,n2))
I_eroded = cv2.erode(I_dilated, SE2)
I2_mask = np.copy(I_eroded)
#plt.imshow(I2_mask, cmap='gray')
#plt.show()

# 3c
def immask(image, mask):
    mask = mask.astype(np.uint8)
    expanded_mask = np.expand_dims(mask, axis=-1)
    masked_image = image * expanded_mask
    return masked_image

I2_masked = immask(I2, I2_mask)
#plt.imshow(I2_masked)
#plt.show()

# 3d
I3 = cv2.imread('images/eagle.jpg')
I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2RGB)
I3_g = cv2.cvtColor(I3, cv2.COLOR_RGB2GRAY)
I3_g_float = I3_g.astype(np.float64) / 255.0
I3_threshold = np.copy(I3_g_float)
threshold = 0.6
I3_threshold[I3_g_float < threshold] = 0
I3_threshold[I3_g_float >= threshold] = 1
I3_threshold_fix = 1.0 - I3_threshold
n = 7
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
I3_dilated = cv2.dilate(I3_threshold_fix, SE)
n2 = 8
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n2,n2))
I3_eroded = cv2.erode(I3_dilated, SE2)
I3_mask = np.copy(I3_eroded)

I3_masked = immask(I3, I3_mask)
#plt.imshow(I3_masked)
#plt.show()

# 3e
I4 = cv2.imread('images/coins.jpg')
I4 = cv2.cvtColor(I4, cv2.COLOR_BGR2RGB)
I4_g = cv2.cvtColor(I4, cv2.COLOR_RGB2GRAY)
I4_g_float = I4_g.astype(np.float64) / 255.0
treshold = 0.9
I4_treshold = np.copy(I4_g_float)
I4_treshold = np.where(I4_g_float >= threshold, 1, 0)
I4_treshold_fix = 1.0 - I4_treshold
n = 20
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
I4_dilated = cv2.dilate(I4_treshold_fix, SE)
n2 = 16
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n2,n2))
I4_eroded = cv2.erode(I4_dilated, SE2)
I4_mask = np.copy(I4_eroded)
#I4_masked = immask(I4, I4_mask)
I4_mask_uint8 = (I4_mask * 255).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(I4_mask_uint8, connectivity=8)
I4_result = I4.copy()
i = 1
while i < num_labels:
    area = stats[i, -1]
    if area > 700:
        I4_result[labels == i] = [255, 255, 255]
    i += 1
#plt.imshow(I4_result)
#plt.show()