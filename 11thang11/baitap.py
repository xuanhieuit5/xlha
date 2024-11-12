import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang thang độ xám
img = cv2.imread('anh2.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng Gaussian để làm mờ
gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Áp dụng các toán tử phát hiện biên
sobel_edges = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 1, 1, ksize=3)
prewitt_edges_x = cv2.filter2D(gaussian_blurred, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
prewitt_edges_y = cv2.filter2D(gaussian_blurred, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
roberts_edges = cv2.filter2D(gaussian_blurred, -1, np.array([[1, 0], [0, -1]]))
canny_edges = cv2.Canny(gaussian_blurred, 100, 200)

# Hiển thị các kết quả
plt.figure(figsize=(10, 8))
plt.subplot(231), plt.imshow(sobel_edges, cmap='gray'), plt.title('Sobel')
plt.subplot(232), plt.imshow(prewitt_edges_x + prewitt_edges_y, cmap='gray'), plt.title('Prewitt')
plt.subplot(233), plt.imshow(roberts_edges, cmap='gray'), plt.title('Roberts')
plt.subplot(234), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny')
plt.subplot(235), plt.imshow(gaussian_blurred, cmap='gray'), plt.title('Gaussian Blurred')
plt.show()
