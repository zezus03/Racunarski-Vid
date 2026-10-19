import cv2# type: ignore
import numpy as np# type: ignore
import matplotlib.pyplot as plt # type: ignore


def fft(img):
    img_fft = np.fft.fft2(img) 
    img_fft = np.fft.fftshift(img_fft) 
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1): 
    img_fft = complex_moduo_1 * np.exp(magnitude_log) 
    img_filtered = np.abs(np.fft.ifft2(img_fft)) 
    return img_filtered 


def noise_removal(img_fft_log, img_mag_1): 
    
    img_fft_log[231, 231] = 0 
    img_fft_log[156, 356] = 0
    img_fft_log[356, 156] = 0
    img_fft_log[281, 281] = 0
   
    img_filtered = inverse_fft(img_fft_log, img_mag_1) 

    cv2.imwrite('Ulazna slika bez suma.png', img_filtered)

    return img_filtered 

img = cv2.imread("slika_3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #


img_fft = fft(img)
img_fft_mag = np.abs(img_fft) 
img_mag_1 = img_fft / img_fft_mag 


img_fft_log = np.log(img_fft_mag) 

img_removed = noise_removal(img_fft_log, img_mag_1)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Ulazna slika')


plt.subplot(2, 2, 2)
plt.imshow(np.log(img_fft_mag), cmap='gray')
plt.title('Magnituda spektra pre uklanjanja šuma')




plt.subplot(2, 2, 3)
plt.imshow(img_fft_log, cmap='gray')
plt.title('Magnituda spektra posle uklanjanja šuma')


plt.subplot(2, 2, 4)
plt.imshow(img_removed, cmap='gray')
plt.title('Ulazna slika bez šuma')

plt.tight_layout() 
plt.show()
