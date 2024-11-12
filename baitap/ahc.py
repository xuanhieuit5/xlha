import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler
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

# 1. Agglomerative Hierarchical Clustering (AHC) trên IRIS
ahc_iris = AgglomerativeClustering(n_clusters=n_clusters_iris)
ahc_labels_iris = ahc_iris.fit_predict(X_iris)

# Đánh giá phân cụm AHC trên IRIS
rand_ahc_iris = adjusted_rand_score(y_iris, ahc_labels_iris)
f1_ahc_iris = f1_score(y_iris, ahc_labels_iris, average='macro')

# Đọc ảnh giao thông (ảnh vệ tinh)
image = cv2.imread('anh1.jpg')
if image is None:
    print("Error: Image not loaded. Check the file path.")
    exit()

# Chuyển ảnh sang RGB và chuyển thành mảng 2D
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image_rgb.reshape(-1, 3)

# Số cụm muốn phân (ví dụ 2)
n_clusters_img = 2

# 2. Agglomerative Hierarchical Clustering trên ảnh giao thông
ahc_img = AgglomerativeClustering(n_clusters=n_clusters_img)
ahc_labels_img = ahc_img.fit_predict(pixels)
ahc_image = ahc_labels_img.reshape(image.shape[:2])

# Hiển thị ảnh phân cụm
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[1].imshow(ahc_image, cmap='viridis')
axs[1].set_title('AHC Clustering on Traffic Image')
plt.show()

# In kết quả đánh giá cho dữ liệu IRIS
print("Adjusted Rand Index on IRIS dataset (AHC):", rand_ahc_iris)
print("F1-score on IRIS dataset (AHC):", f1_ahc_iris)
