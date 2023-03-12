import cv2
import numpy as np

# load gambar
img = cv2.imread('image.jpeg')

# definisikan kernel
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# lakukan operasi median filter
median_filtered_img = cv2.medianBlur(img, kernel_size)

# tampilkan hasil
cv2.imshow('Gambar Asli', img)
cv2.imshow('Median Filtered Gambar', median_filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()