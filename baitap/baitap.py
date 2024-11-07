import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def initialize_centroids(pixels, k):
    indices = np.random.choice(len(pixels), k, replace=False)
    return pixels[indices]


def assign_clusters(pixels, centroids):
    clusters = []
    for pixel in pixels:
        # Tính khoảng cách giữa pixel và mỗi centroid, chọn centroid gần nhất
        distances = [np.linalg.norm(pixel - centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)
def update_centroids(pixels, clusters, k):
    # Cập nhật lại các tâm cụm dựa trên trung bình của các điểm trong cụm
    new_centroids = []
    for i in range(k):
        # Lấy tất cả các điểm thuộc về cụm `i`
        cluster_points = pixels[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            # Nếu không có điểm nào trong cụm, giữ nguyên tâm cụm ban đầu
            new_centroids.append(pixels[np.random.randint(0, len(pixels))])
    return np.array(new_centroids)


def kmeans(pixels, k, max_iters=100, tolerance=0.01):
    # Khởi tạo tâm cụm
    centroids = initialize_centroids(pixels, k)
    for i in range(max_iters):
        # Gán các điểm vào các cụm gần nhất
        clusters = assign_clusters(pixels, centroids)
        # Tính toán lại các tâm cụm
        new_centroids = update_centroids(pixels, clusters, k)

        # Kiểm tra điều kiện dừng
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids
    return clusters, centroids


# Đọc ảnh từ file
image_dir = r"anh1.jpg"
image = Image.open(image_dir)
image = np.array(image)

# Chuyển đổi ảnh thành mảng 2D (số pixel x 3 giá trị màu RGB)
pixels = image.reshape(-1, 3).astype(float)

k = 4
clusters, centroids = kmeans(pixels, k)

# Thay thế mỗi pixel bằng màu của tâm cụm
segmented_pixels = np.array([centroids[cluster] for cluster in clusters])
segmented_image = segmented_pixels.reshape(image.shape).astype(np.uint8)

# Hiển thị ảnh gốc và ảnh sau khi phân cụm
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Ảnh gốc")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title("Ảnh sau khi phân cụm K-means")
plt.axis('off')

plt.show()