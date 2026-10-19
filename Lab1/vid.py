import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft


def inverse_fft(img_fft):
    ishift = np.fft.ifftshift(img_fft)
    img_back = np.fft.ifft2(ishift)
    img_back = np.abs(img_back)
    return img_back


def noise_removal(img_fft):
    # RUČNO UKLANJANJE ŠUMA - mora na kompleksnom FFT-u!
    img_fft[231, 231] = 0
    img_fft[156, 356] = 0
    img_fft[356, 156] = 0
    img_fft[281, 281] = 0

    img_filtered = inverse_fft(img_fft)

    img_filtered = np.uint8(255 * (img_filtered / np.max(img_filtered)))
    cv2.imwrite('Ulazna slika bez suma.png', img_filtered)

    return img_filtered


# --------------------
# Glavni deo
# --------------------

img = cv2.imread("slika_3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1) FFT
img_fft = fft(img)
img_fft_mag = np.abs(img_fft)

# 2) RUČNO UKLANJANJE ŠUMA
img_removed = noise_removal(img_fft.copy())

# 3) Prikaz
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Ulazna slika")

plt.subplot(2, 2, 2)
plt.imshow(np.log(1 + img_fft_mag), cmap='plasma')
plt.title("Magnituda pre uklanjanja šuma")

plt.subplot(2, 2, 3)
plt.imshow(np.log(1 + np.abs(img_fft)), cmap='plasma')
plt.title("Magnituda posle uklanjanja šuma")

plt.subplot(2, 2, 4)
plt.imshow(img_removed, cmap='gray')
plt.title("Slika bez šuma")

plt.tight_layout()
plt.show()
