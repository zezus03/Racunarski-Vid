import cv2# type: ignore
import numpy as np# type: ignore
import matplotlib.pyplot as plt# type: ignore

def morfoloska_rekonstrukcija(marker, mask):
    kernel = np.ones((3, 3), np.uint8)
    reconstruction = marker.copy()
    
    while True:
        prev_reconstruction = reconstruction.copy()
        dilated = cv2.dilate(reconstruction, kernel, iterations=1)
        reconstruction = cv2.bitwise_and(dilated, mask)
        
        if np.array_equal(reconstruction, prev_reconstruction):
            break
            
    return reconstruction

def main():
    img = cv2.imread('coins.png')
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    mask_filled = np.zeros_like(gray)

    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(mask_filled, (x, y), r, 255, -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([20, 255, 255])
    
    lower_red2 = np.array([165, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    marker_copper = cv2.bitwise_or(mask1, mask2)
    
    kernel_clean = np.ones((3,3), np.uint8)
    marker_copper = cv2.morphologyEx(marker_copper, cv2.MORPH_OPEN, kernel_clean)

    
    final_mask = morfoloska_rekonstrukcija(marker_copper, mask_filled)

    konacan = cv2.bitwise_and(img, img, mask=final_mask)

    show_images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        mask_filled,
        marker_copper,
        cv2.cvtColor(konacan, cv2.COLOR_BGR2RGB)
    ]
    show_titles = ['Original', 'Maska', 'Marker', 'Izdvojen']

    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        if i in (0, 3):
            plt.imshow(show_images[i])
        else:
            plt.imshow(show_images[i], cmap='gray')
        plt.title(show_titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite("output.png", final_mask)
    print("'output.png'")

if __name__ == "__main__":
    main()