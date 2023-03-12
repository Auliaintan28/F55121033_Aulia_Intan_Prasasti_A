import cv2
import numpy as np

# Load an image
img = cv2.imread('image.jpeg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Subtract the blurred image from the grayscale image to get the mask
mask = cv2.subtract(gray_img, blurred_img)

# Scale the mask to a range of 0 to 255
mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)

# Add the scaled mask to the original image
unsharp_masked = cv2.addWeighted(img, 1.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), -0.5, 0)

# Display the original and unsharp masked images
cv2.imshow('Original', img)
cv2.imshow('Unsharp Masked', unsharp_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()