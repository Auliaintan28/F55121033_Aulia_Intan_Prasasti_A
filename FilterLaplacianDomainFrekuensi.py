import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply 2D FFT to the image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Define Laplacian kernel in the frequency domain
kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
lap_kernel = np.zeros_like(img, dtype=np.float32)
lap_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
lap_kernel = np.fft.fft2(lap_kernel)
lap_kernel = np.fft.fftshift(lap_kernel)

# Apply Laplacian filter in the frequency domain
filtered_f = fshift * lap_kernel

# Apply inverse 2D FFT to get the filtered image
filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))

# Normalize the output
filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the original and filtered images
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(filtered, cmap='gray')
plt.title('Laplacian Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()