# Import các thư viện cần thiết
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# --- Phần 1: Xử lý dữ liệu IRIS ---

# Tải bộ dữ liệu IRIS
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Chia dữ liệu IRIS thành tập huấn luyện và kiểm tra
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# --- Phần 2: Xử lý ảnh nha khoa ---

# Đường dẫn đến thư mục ảnh nha khoa
image_directory = '/Users/smb/Downloads/Panoramic radiographs with periapical lesions Dataset/Periapical Dataset/Periapical Lesions/Augmentation JPG Images/'  # Đường dẫn đúng

# Sử dụng ImageDataGenerator để tải ảnh, chuyển về kích thước 64x64 và chuẩn hóa
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.3)

train_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- Phần 3: Xây dựng mô hình CNN cho phân loại ảnh nha khoa ---

cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Số lớp ảnh nha khoa
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Phần 4: Huấn luyện mô hình CNN với bộ ảnh nha khoa ---

history = cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# --- Phần 5: Đánh giá mô hình với ảnh nha khoa ---

loss, accuracy = cnn_model.evaluate(val_generator)
print(f'Accuracy on dental images: {accuracy * 100:.2f}%')

# --- Phần 6: Phân lớp với CART và ID3 ---

# CART (Gini Index)
cart_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_classifier.fit(X_train_iris, y_train_iris)
y_pred_cart = cart_classifier.predict(X_test_iris)

# ID3 (Information Gain)
id3_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_classifier.fit(X_train_iris, y_train_iris)
y_pred_id3 = id3_classifier.predict(X_test_iris)

# --- Đánh giá và hiển thị kết quả ---

# CART
accuracy_cart = accuracy_score(y_test_iris, y_pred_cart)
classification_report_cart = classification_report(y_test_iris, y_pred_cart)
conf_matrix_cart = confusion_matrix(y_test_iris, y_pred_cart)

print("CART (Gini Index) - Accuracy:", accuracy_cart)
print("\nCART Classification Report:\n", classification_report_cart)

# ID3
accuracy_id3 = accuracy_score(y_test_iris, y_pred_id3)
classification_report_id3 = classification_report(y_test_iris, y_pred_id3)
conf_matrix_id3 = confusion_matrix(y_test_iris, y_pred_id3)

print("ID3 (Information Gain) - Accuracy:", accuracy_id3)
print("\nID3 Classification Report:\n", classification_report_id3)

# Vẽ biểu đồ ma trận nhầm lẫn cho CART
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_cart, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - CART (Gini Index)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Vẽ biểu đồ ma trận nhầm lẫn cho ID3
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_id3, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - ID3 (Information Gain)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# --- Phần 7: Trực quan hóa kết quả ---

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy on Dental Images')
plt.show()