import cv2
import numpy as np
from matplotlib import pyplot as plt

# load gambar
img = cv2.imread('image.jpeg', 0)

# lakukan FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# tampilkan hasil
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT'), plt.xticks([]), plt.yticks([])
plt.show()