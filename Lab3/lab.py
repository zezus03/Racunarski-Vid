import cv2 as cv
import numpy as np



def trim(img):
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
   
    coords = cv.findNonZero(thresh)
    if coords is None:
        return img 
    x, y, w, h = cv.boundingRect(coords)
    return img[y:y+h, x:x+w]

def napravi_panoramu_od_dve_slike(img_base, img_new):
    sift = cv.SIFT_create()
    
    kp_base, des_base = sift.detectAndCompute(img_base, None)
    kp_new, des_new = sift.detectAndCompute(img_new, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_new, des_base, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return img_base

    src_pts = np.float32([kp_new[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_base[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    #matrica 3x3 koja nam kaze kako treba matricu iskriviti,rotirati
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    h_new, w_new = img_new.shape[:2]
    h_base, w_base = img_base.shape[:2]

    corners_new = np.float32([[0,0], [w_new,0], [w_new,h_new], [0,h_new]]).reshape(-1,1,2)
    warped_corners = cv.perspectiveTransform(corners_new, H)
    corners_base = np.float32([[0,0], [w_base,0], [w_base,h_base], [0,h_base]]).reshape(-1,1,2)
    all_corners = np.concatenate((warped_corners, corners_base), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

   
    warped_img = cv.warpPerspective(img_new, H_translation.dot(H), (x_max - x_min, y_max - y_min))

   
    base_on_canvas = np.zeros_like(warped_img)
    y_start, x_start = translation_dist[1], translation_dist[0]
    base_on_canvas[y_start:y_start+h_base, x_start:x_start+w_base] = img_base

   
    
   
    final_img = np.maximum(base_on_canvas, warped_img)

    return trim(final_img)


slika1 = cv.imread("1.jpg")
slika2 = cv.imread("2.jpg")
slika3 = cv.imread("3.jpg")



slike_list = [slika1, slika2, slika3]

panorama = slike_list[0]
for i in range(1, len(slike_list)):
    panorama = napravi_panoramu_od_dve_slike(panorama, slike_list[i])

cv.imshow("Popravljena Panorama", panorama)
cv.imwrite("output_manual.jpg", panorama)
cv.waitKey(0)
cv.destroyAllWindows()