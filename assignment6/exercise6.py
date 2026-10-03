import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from a6_utils import *

# EXERCISE 1
# 1b, 1c
points = np.loadtxt('data/points.txt')
mean = np.mean(points, axis=0)
Xd = points - mean
N = Xd.shape[0]
C = (Xd.T @ Xd) / (N-1)
eigvals, eigvecs = np.linalg.eigh(C)
"""
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(points[:,0], points[:,1], color='red', label='Data points')
ax.scatter(mean[0], mean[1], color='blue', label='Mean')
for i in range(len(eigvals)):
    ax.plot([mean[0], mean[0]+np.sqrt(eigvals[i])*eigvecs[0,i]],
            [mean[1], mean[1]+np.sqrt(eigvals[i])*eigvecs[1,i]],
            lw=2, label=f'Eigenvector {i+1}')
drawEllipse(mean, C, n_std=1)
ax.set_title('PCA of 2D Data with Gaussian Ellipse')
ax.legend()
ax.grid(True)
plt.show()
"""

# 1d
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvals_normalized = eigvals / eigvals[0]
cumulative = np.cumsum(eigvals_normalized)
"""
plt.figure(figsize=(6,4))
plt.plot(range(1, len(eigvals_normalized)+1), cumulative, marker='o')
plt.xticks(range(1, len(eigvals_normalized)+1))
plt.title('Cumulative normalized eigenvalues (PCA)')
plt.grid(True)
plt.show()
total_variance = np.sum(eigvals)
variance_first = eigvals[0] / total_variance * 100
print(f"Variance explained by first eigenvector: {variance_first:.2f}%")
"""

# 1e
mean = np.mean(points, axis=0)
Xd = points - mean
N, m = Xd.shape
C = (Xd.T @ Xd) / (N-1)
eigvals, U = np.linalg.eigh(C)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
U = U[:, order]
Y = Xd @ U
Y[:, -1] = 0 # remove last eigenvalue
Xd_proj = Y @ U.T
points_proj = Xd_proj + mean
# print("Original points:\n", points)
# print("Projected points:\n", points_proj)

# 1f
points = np.loadtxt('data/points.txt')
qpoint = np.array([6,6])
distances = np.linalg.norm(points - qpoint, axis=1)
closest_idx = np.argmin(distances)
# print("Closest original point to qpoint:", points[closest_idx], ", distance:", distances[closest_idx])
mean = np.mean(points, axis=0)
Xd = points - mean
N, m = Xd.shape
C = (Xd.T @ Xd) / (N-1)
eigvals, U = np.linalg.eigh(C)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
U = U[:, order]
Y = Xd @ U
Y_q = (qpoint - mean) @ U
Y[:, -1] = 0
Y_q[-1] = 0
Xd_proj = Y @ U.T
points_proj = Xd_proj + mean
qpoint_proj = Y_q @ U.T + mean
distances_proj = np.linalg.norm(points_proj - qpoint_proj, axis=1)
closest_proj_idx = np.argmin(distances_proj)
# print("Closest projected point to qpoint:", points_proj[closest_proj_idx], ", distance:", distances_proj[closest_proj_idx])
"""
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], color='red', label='Original points')
plt.scatter(points_proj[:,0], points_proj[:,1], color='blue', label='Projected points')
plt.scatter(qpoint[0], qpoint[1], color='green', label='qpoint original')
plt.scatter(qpoint_proj[0], qpoint_proj[1], color='purple', label='qpoint projected')
for i in range(len(points)):
    plt.plot([points[i,0], points_proj[i,0]], [points[i,1], points_proj[i,1]], 'k--', alpha=0.5)
plt.title('PCA projection and closest point')
plt.legend()
plt.grid(True)
plt.show()
"""

# 1g
points = np.loadtxt('data/points_50D.txt')
mean = np.mean(points, axis=0)
Xd = points - mean
N, D = Xd.shape
C = (Xd.T @ Xd) / (N-1)
eigvals, eigvecs = np.linalg.eigh(C)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
total_variance = np.sum(eigvals)
cumulative_variance = np.cumsum(eigvals) / total_variance
threshold = 0.8 # 80%
cum_sum = 0.0
k = 0
for i, value in enumerate(eigvals):
    cum_sum += value
    k += 1
    if cum_sum / total_variance >= threshold:
        break
"""
print(f"Total dimensions: {D}")
print(f"Eigenvectors needed for 80% variance: {k}")
print(f"Eigenvectors that can be set to zero: {D - k}")
"""

# EXERCISE 2
# 3a
points = np.loadtxt('data/points.txt')
mean = np.mean(points, axis=0)
Xd = points - mean
N, m = Xd.shape
C_direct = (Xd.T @ Xd) / (N-1)
eigvals, eigvecs = np.linalg.eigh(C_direct)
C_dual = (Xd @ Xd.T) / (N-1)
Uq, S, V = np.linalg.svd(C_dual)
U = (Xd.T @ Uq) * 1/np.sqrt(S * (N-1))
"""
print("Direct PCA eigenvalues:\n", eigvals)
print("Dual PCA eigenvalues:\n", S)
print("\nDirect PCA eigenvectors:\n", eigvecs)
print("Dual PCA eigenvectors:\n", U)
"""

# 2b
Y = Xd @ U
X_reconstructed = Y @ U.T + mean
# print("Original data:\n", points)
# print("\nReconstructed data:\n", X_reconstructed)
difference = np.array(points) - X_reconstructed
# print(f"\nNumerical reconstruction error: {difference}")

# EXERCISE 3
# 3a
def load_image_series(folder_path):
    image_vectors = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_vector = img.reshape(-1)
            image_vectors.append(img_vector)
    X = np.array(image_vectors).T  # size: m*n x 64
    return X, img.shape

X, image_shape = load_image_series('data/faces/1')
# print("Shape of data matrix X:", X.shape)
# print("Original image shape:", img_shape)

# 3b
def dual_pca(X, eps=1e-15):
    mean = np.mean(X, axis=1, keepdims=True)
    Xd = X - mean
    mn, N = Xd.shape
    C_dual = (Xd.T @ Xd) / (N - 1)
    Uq, S, V = np.linalg.svd(C_dual)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps))
    U = Xd @ Uq @ S_inv_sqrt / np.sqrt(N - 1)
    return U, S, mean

U, S, mean = dual_pca(X)
n_pictures = 5
"""
plt.figure(figsize=(10, 2))
for i in range(n_pictures):
    plt.subplot(1, n_pictures, i+1)
    plt.imshow(U[:, i].reshape(img_shape), cmap='gray')
    plt.axis('off')
    plt.title(f"Eigenvector {i+1}")
plt.show()
"""
x = X[:, 0:1]
y = U.T @ (x - mean)
x_rec = U @ y + mean
"""
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(x.reshape(img_shape), cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(x_rec.reshape(img_shape), cmap='gray')
plt.title("Reconstructed")
plt.axis('off')
plt.show()
"""
error = np.linalg.norm(x - x_rec)
# print(f"Reconstruction error: {error:.2e}")

# 3c
x_new = x.copy()
pixel_idx = 3990
x_new[pixel_idx] = 0 # change pixel
y_new = U.T @ (x_new - mean)
x_new_rec = U @ y_new + mean

y_pca = y.copy()
y_pca[1] = 0 # change component
x_rec_pca = U @ y_pca + mean
"""
plt.figure(figsize=(20,3))
plt.subplot(1,4,1)
plt.imshow(x.reshape(img_shape), cmap='gray')
plt.title("Original")
plt.subplot(1,4,2)
plt.imshow(x_new.reshape(img_shape), cmap='gray')
plt.title("Changed pixel")
plt.subplot(1,4,3)
plt.imshow(x_new_rec.reshape(img_shape), cmap='gray')
plt.title("Reconstruction from changed pixel")
plt.subplot(1,4,4)
plt.imshow(x_rec_pca.reshape(img_shape), cmap='gray')
plt.title("Reconstruction from changed component")
plt.tight_layout
plt.show()
"""

# 3d
x = X[:, 0:1]
y = U.T @ (x - mean)
components_list = [32, 16, 8, 4, 2, 1]
"""
plt.figure(figsize=(18,3))
for i, component in enumerate(components_list):
    y_temp = y.copy()
    y_temp[component:] = 0  # zero out remaining components
    x_rec = U @ y_temp + mean
    plt.subplot(1, len(components_list), i+1)
    plt.imshow(x_rec.reshape(img_shape), cmap='gray')
    plt.title(f"{component}")
plt.suptitle("Reconstruction with varying number of PCA components")
plt.tight_layout()
plt.show()
"""

# 3e
X, img_shape = load_image_series('data/faces/2')
U, S, mean = dual_pca(X)
x_avg = np.mean(X, axis=1, keepdims=True)
y_avg = U.T @ (x_avg - mean)
# plt.figure()
# plt.imshow(x_avg.reshape(img_shape), cmap='gray')
# plt.close()
idx1 = 0
idx2 = 1
scale = 10
t_vals = np.linspace(0, 2*np.pi, 90)
"""
plt.figure()
for t in t_vals:
    y_modified = y_avg.copy()
    v1 = scale * np.sin(t)
    v2 = scale * np.cos(t)
    y_modified[idx1] = y_avg[idx1] + v1
    y_modified[idx2] = y_avg[idx2] + v2
    x_recreated = U @ y_modified + mean
    difference = x_recreated - x_avg
    plt.clf()
    plt.imshow(difference.reshape(img_shape), cmap='gray')
    plt.title( f"EV{idx1+1} = {v1:+.2f},  EV{idx2+1} = {v2:+.2f}")
    plt.axis('off')
    plt.draw()
    plt.pause(0.1)
plt.close()
"""
# 3f
X, img_shape = load_image_series('data/faces/1')
U, S, mean = dual_pca(X)
img = cv2.imread('data/elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, img_shape)
x_elephant = img.reshape(-1, 1)
y_elephant = U.T @ (x_elephant - mean)
x_reconstructed = U @ y_elephant + mean
"""
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(x_reconstructed.reshape(img_shape), cmap='gray')
plt.title("Reconstructed using face PCA")
plt.show()
"""
# 3g
def load_image_series(folder_path, target_shape=(200, 200)):
    image_vectors = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_shape) # only difference
            image_vectors.append(img.reshape(-1))
    X = np.array(image_vectors).T
    return X, target_shape

X, img_shape = load_image_series('data/faces/me')
U, S, mean = dual_pca(X)
threshold = 11500
"""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # face detection
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w] # cutout detected face
        face_resized = cv2.resize(face_img, img_shape)
        face_vector = face_resized.reshape(-1, 1)
        reconstruction = U @ (U.T @ (face_vector - mean)) + mean # PCA reconstruction
        error = np.linalg.norm(reconstruction - face_vector)
        label = "ME" if error < threshold else "Unknown" # label based on error
        color = (0, 255, 0) if label == "ME" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # draw rectangle
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # stop if q pressed
        break
capture.release()
cv2.destroyAllWindows()
"""