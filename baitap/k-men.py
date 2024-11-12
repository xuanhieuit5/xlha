import cv2
import numpy as np
from sklearn.cluster import KMeans
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

# 1. K-means trên IRIS
kmeans_iris = KMeans(n_clusters=n_clusters_iris, random_state=0)
kmeans_labels_iris = kmeans_iris.fit_predict(X_iris)

# Đánh giá phân cụm K-means trên IRIS
rand_kmeans_iris = adjusted_rand_score(y_iris, kmeans_labels_iris)
f1_kmeans_iris = f1_score(y_iris, kmeans_labels_iris, average='macro')

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

# 2. K-means trên ảnh giao thông
kmeans_img = KMeans(n_clusters=n_clusters_img, random_state=0)
kmeans_labels_img = kmeans_img.fit_predict(pixels)
kmeans_image = kmeans_labels_img.reshape(image.shape[:2])

# Hiển thị ảnh phân cụm
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[1].imshow(kmeans_image, cmap='viridis')
axs[1].set_title('K-means Clustering on Traffic Image')
plt.show()

# In kết quả đánh giá cho dữ liệu IRIS
print("Adjusted Rand Index on IRIS dataset (K-means):", rand_kmeans_iris)
print("F1-score on IRIS dataset (K-means):", f1_kmeans_iris)
