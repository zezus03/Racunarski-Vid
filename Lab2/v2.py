import numpy as np # type: ignore
import cv2 as cv# type: ignore
import matplotlib.pyplot as plt# type: ignore


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv.dilate(src=marker, kernel=kernel)
        expanded = cv.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


original = cv.imread('coins.png')
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

_, mask_coins = cv.threshold(gray, 140, 255, cv.THRESH_BINARY_INV)


mask_coins_filtered = cv.morphologyEx(mask_coins, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (16, 16)),iterations=1)


hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV)
saturation = hsv[:, :, 1]



_, bakarni_donji = cv.threshold(saturation, 60, 255, cv.THRESH_BINARY)

_, bakarni_gornji = cv.threshold(saturation, 130, 255, cv.THRESH_BINARY_INV)
bakarni = cv.bitwise_and(bakarni_gornji, bakarni_donji)

bakarni_open = cv.morphologyEx(bakarni, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

marker = cv.bitwise_and(bakarni_open, mask_coins_filtered)

reconstructed = morphological_reconstruction(marker, mask_coins_filtered)

plt.title('Maska')
plt.imshow(reconstructed)
plt.show()
result = cv.bitwise_and(original, original, mask=reconstructed)
cv.imshow('Novcic', result)
cv.waitKey(0)
cv.destroyAllWindows()