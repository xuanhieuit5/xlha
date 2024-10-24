import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('anh1.jpg', cv2.IMREAD_GRAYSCALE)

# Hiển thị ảnh gốc
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# Toán tử Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Tích chập với Sobel
sobel_x_image = cv2.filter2D(image, -1, sobel_x)
sobel_y_image = cv2.filter2D(image, -1, sobel_y)

# Kết hợp gradient theo trục x và y
sobel_combined = np.sqrt(sobel_x_image**2 + sobel_y_image**2)

# Hiển thị kết quả
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.show()

# Toán tử Laplace Gaussian
laplace_gaussian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

# Tích chập với Laplace Gaussian
laplace_image = cv2.filter2D(image, -1, laplace_gaussian)

# Hiển thị kết quả
plt.imshow(laplace_image, cmap='gray')
plt.title('Laplace Gaussian Edge Detection')
plt.show()