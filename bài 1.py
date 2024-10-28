import cv2
import numpy as np
from matplotlib import pyplot as plt

# Tải ảnh
image = cv2.imread('meo.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Ảnh âm tính
def negative(image):
    return 255 - image

# 2. Tăng cường độ tương phản (Sử dụng cân bằng histogram)
def enhance_contrast(image):
    return cv2.equalizeHist(image)

# 3. Biến đổi log
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    return np.array(log_image, dtype=np.uint8)

# Tạo các ảnh đã xử lý
negative_image = negative(image)
contrast_image = enhance_contrast(image)
log_image = log_transform(image)

# Cân bằng Histogram (sử dụng lại hàm)
hist_eq_image = enhance_contrast(image)

# Hiển thị các ảnh đã xử lý 
titles = ['Ảnh âm tính', 'Tăng cường độ tương phản', 'Biến đổi Log', 'Cân bằng Histogram']
images = [negative_image, contrast_image, log_image, hist_eq_image]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

# Lưu ảnh đầu ra
output_filenames = ['negative_image.jpg', 'contrast_image.jpg', 'log_image.jpg', 'hist_eq_image.jpg']
for filename, img in zip(output_filenames, images):
    cv2.imwrite(filename, img)
