import matplotlib.pyplot as plt
import numpy as np
import tifffile

delta = 1e-8

img = tifffile.imread('X2025-06-19_21.26.18_Xlzso-NO3-3_r1_72_000.tif')
print(img.shape)
ffts = []
fft_sum = np.zeros(img.shape[1:3], dtype=np.complex128)
for i in range(img.shape[0]):
    fft = np.fft.fft2(img[i])
    fft = np.fft.fftshift(fft)
    fft_sum += fft
    # bis hier fft(sum) = sum(fft)
    mag = np.abs(fft)
    # information a+ib -> a^2 + b^2 
    # loss of phase information
    log_mag = np.log(mag + delta)
    # TODO: sigma, gamma, contrast???
    ffts.append(log_mag)

ffts = np.array(ffts)
fft_sum = np.abs(fft_sum)
fft_sum = np.log(fft_sum + delta)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(ffts[0], cmap='gray')
plt.title('FFT of first image')
plt.subplot(1, 2, 2)
plt.imshow(fft_sum, cmap='gray')
plt.title('Sum of FFTs')
plt.show()
