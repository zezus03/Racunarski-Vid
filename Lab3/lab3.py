import cv2 as cv
import numpy as np

# Ucitavanje slika
slika1 = cv.imread("1.jpg")
slika2 = cv.imread("2.jpg")
slika3 = cv.imread("3.jpg")

# -----------------------------
# Pomocne funkcije
# -----------------------------

def trim(img):
    """Uklanja crne ivice sa slike"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    coords = cv.findNonZero(thresh)
    x, y, w, h = cv.boundingRect(coords)
    return img[y:y+h, x:x+w]


def napravi_panoramu_od_dve_slike(imgL, imgR):
    """Spaja dve slike u panoramu"""

    sift = cv.SIFT_create()

    kpR, desR = sift.detectAndCompute(imgR, None)
    kpL, desL = sift.detectAndCompute(imgL, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desR, desL, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 30:
        print("Nedovoljno poklapanja")
        return imgL

    src_pts = np.float32([kpR[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpL[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    hR, wR = imgR.shape[:2]
    hL, wL = imgL.shape[:2]

    cornersR = np.float32([[0,0], [wR,0], [wR,hR], [0,hR]]).reshape(-1,1,2)
    warped_corners = cv.perspectiveTransform(cornersR, H)

    cornersL = np.float32([[0,0], [wL,0], [wL,hL], [0,hL]]).reshape(-1,1,2)

    all_corners = np.concatenate((warped_corners, cornersL), axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    panorama = cv.warpPerspective(
        imgR, T @ H, (x_max - x_min, y_max - y_min)
    )

    # Postavljanje leve slike (bez blendinga)
    roi = panorama[-y_min:hL - y_min, -x_min:wL - x_min]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # prag malo veci da "pojede" tanke crne linije
    _, mask = cv.threshold(gray, 15, 255, cv.THRESH_BINARY_INV)
    mask = mask.astype(bool)

    roi[mask] = imgL[mask]
    panorama[-y_min:hL - y_min, -x_min:wL - x_min] = roi

    panorama = trim(panorama)
    return panorama


def napravi_panoramu(slike):
    pano = slike[0]
    for i in range(1, len(slike)):
        pano = napravi_panoramu_od_dve_slike(pano, slike[i])
    return pano


# -----------------------------
# GLAVNI PROGRAM
# -----------------------------

# SREDNJA SLIKA MORA PRVA
slike = [slika2, slika1, slika3]

panorama = napravi_panoramu(slike)

cv.imshow("Panorama", panorama)
cv.imwrite("output.jpg", panorama)
cv.waitKey(0)
cv.destroyAllWindows()
