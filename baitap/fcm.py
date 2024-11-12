import cv2
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Đọc và chuẩn bị dữ liệu IRIS
data_iris = load_iris()
X_iris = data_iris.data
y_iris = data_iris.target  # Nhãn thật của IRIS dataset

# Chuẩn hóa dữ liệu IRIS
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)

# Số cụm (IRIS có 3 loại hoa)
n_clusters_iris = 3

# 1. Fuzzy C-means trên IRIS
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_iris.T, c=n_clusters_iris, m=2, error=0.005, maxiter=1000, init=None
)

# Dự đoán nhãn bằng cách lấy nhãn với độ thành viên cao nhất
fcm_labels_iris = np.argmax(u, axis=0)

# Đánh giá phân cụm FCM trên IRIS
rand_fcm_iris = adjusted_rand_score(y_iris, fcm_labels_iris)
f1_fcm_iris = f1_score(y_iris, fcm_labels_iris, average='macro')

# Đọc ảnh giao thông (ảnh vệ tinh)
image = cv2.imread('anh1.jpg')
if image is None:
    print("Error: Image not loaded. Check the file path.")
    exit()

# Chuyển ảnh sang RGB và chuyển thành mảng 2D
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image_rgb.reshape(-1, 3)

# Số cụm cho ảnh giao thông (ví dụ: 2)
n_clusters_img = 2

# 2. Fuzzy C-means trên ảnh giao thông
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    pixels.T, c=n_clusters_img, m=2, error=0.005, maxiter=1000, init=None
)
fcm_labels_img = np.argmax(u, axis=0)
fcm_image = fcm_labels_img.reshape(image.shape[:2])

# Hiển thị ảnh phân cụm
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[1].imshow(fcm_image, cmap='viridis')
axs[1].set_title('Fuzzy C-means Clustering on Traffic Image')
plt.show()

# In kết quả đánh giá cho dữ liệu IRIS
print("Adjusted Rand Index on IRIS dataset (FCM):", rand_fcm_iris)
print("F1-score on IRIS dataset (FCM):", f1_fcm_iris)
