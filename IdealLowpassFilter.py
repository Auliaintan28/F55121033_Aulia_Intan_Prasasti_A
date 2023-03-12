import cv2
import numpy as np
from matplotlib import pyplot as plt

# load gambar dan hitung DFT
img = cv2.imread('image.jpeg', 0)
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# buat filter ideal lowpass
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
D = 30  # radius filter
mask = np.zeros((rows, cols), dtype=np.uint8)
cv2.circle(mask, (ccol, crow), D, 1, -1)

# aplikasikan filter ke DFT gambar
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# tampilkan hasil
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Setelah Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()