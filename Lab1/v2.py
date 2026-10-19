import cv2 # type: ignore
import numpy as np# type: ignore
import matplotlib.pyplot as plt# type: ignore


def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft 


def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    return img_filtered





def bolji_noise_removal(img_fft_log, img_mag_1):
    rows, cols = img_fft_log.shape
    center_row, center_col = rows // 2, cols // 2

    img_fft_log_cleaned = img_fft_log.copy()

    mean_val = np.mean(img_fft_log)
    std_val = np.std(img_fft_log)

    threshold = mean_val + 5 * std_val

    mask = img_fft_log > threshold

    radius = 1

    y, x = np.ogrid[:rows, :cols]
    mask_area_to_keep = (x - center_col)**2 + (y - center_row)**2 <= radius**2

    mask[mask_area_to_keep] = False

    img_fft_log_cleaned[mask] = 0

    img_filtered = inverse_fft(img_fft_log_cleaned, img_mag_1)
    img_filtered = np.uint8(255 * (img_filtered / np.max(img_filtered)))
    
    return img_filtered, img_fft_log_cleaned, mask


if __name__ == "__main__":
    img = cv2.imread("slika_3.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag

    img_fft_log = np.log(img_fft_mag)

   

    img_removed_bolji, img_fft_log_cleaned, mask = bolji_noise_removal(img_fft_log, img_mag_1)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ulazna slika')

    plt.subplot(2, 2, 2)
    plt.imshow(img_fft_log, cmap='gray')
    plt.title('Spektar sa sumom')

   

    plt.subplot(2, 2, 3)
    plt.imshow(img_fft_log_cleaned, cmap='gray')
    plt.title('Cist spektar')

    plt.subplot(2, 2, 4)
    plt.imshow(img_removed_bolji, cmap='gray')
    plt.title('Slika bez suma')

    plt.tight_layout()
    plt.show()