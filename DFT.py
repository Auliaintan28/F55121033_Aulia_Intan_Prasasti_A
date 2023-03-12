import cv2
import numpy as np
from matplotlib import pyplot as plt

# load gambar
img = cv2.imread('image.jpeg', 0)

# lakukan DFT
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# tampilkan hasil
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('DFT'), plt.xticks([]), plt.yticks([])
plt.show()