import cv2
import numpy as np

img = cv2.imread('image.jpeg')
kernel_size = 3
kernel = np.ones((kernel_size,kernel_size),np.uint8)
filtered = cv2.dilate(img, kernel)
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
