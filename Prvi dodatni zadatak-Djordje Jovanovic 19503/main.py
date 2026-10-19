import numpy as np # type: ignore
import cv2 # type: ignore
import matplotlib.pyplot as plt# type: ignore


def geodesic_dilation(marker, mask):
  
    kernel = np.ones((3, 3), np.uint8)
    prev = np.zeros_like(marker)

    while not np.array_equal(prev, marker):
        prev = marker.copy()
        marker = cv2.dilate(marker, kernel)
        marker = cv2.bitwise_and(marker, mask)

    return marker


if __name__ == '__main__':

    path = "slika1.jpg"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Greska: Slika na putanji '{path}' nije pronadjena.")
        exit(1)

    
    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    h, w = binary_mask.shape

  
    marker = np.zeros_like(binary_mask)

    marker[0, :]  |= binary_mask[0, :]
    marker[-1, :] |= binary_mask[-1, :]
    marker[:, 0]  |= binary_mask[:, 0]
    marker[:, -1] |= binary_mask[:, -1]
    border_objects = geodesic_dilation(marker, binary_mask)

    
    result_mask = cv2.bitwise_and(
        binary_mask,
        cv2.bitwise_not(border_objects)
    )

   
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("1. Ulazna binarna maska")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(border_objects, cmap='gray')
    plt.title("2. Objekti koji dodiruju ivicu")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result_mask, cmap='gray')
    plt.title("3. Filtriran rezultat")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite("filtriran_rezultat1.png", result_mask)
