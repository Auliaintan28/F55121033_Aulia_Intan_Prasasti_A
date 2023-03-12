import cv2
import numpy as np

# load gambar
img = cv2.imread('image.jpeg')

# definisikan kernel
kernel = np.ones((3,3), np.uint8)

# lakukan operasi min filter
min_filtered_img = cv2.erode(img, kernel, iterations=1)

# tampilkan hasil
cv2.imshow('Gambar Asli', img)
cv2.imshow('Min Filtered Gambar', min_filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()