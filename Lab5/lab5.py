import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os


VIDEO_FILENAME = 'Aruco_board.mp4'
CALIB_IMAGES_PATH = 'calib_image_*.jpg' 

MARKERS_X = 5
MARKERS_Y = 7
MARKER_LENGTH = 0.035      
MARKER_SEPARATION = 0.005  
ARUCO_DICT_TYPE = aruco.DICT_6X6_250


def get_board_object_points(marker_ids):
    object_points = []
    
    for marker_id_arr in marker_ids:
        mid = marker_id_arr[0] 
        
       
        row = mid // MARKERS_X
        col = mid % MARKERS_X
        
    
        start_x = col * (MARKER_LENGTH + MARKER_SEPARATION)
        start_y = row * (MARKER_LENGTH + MARKER_SEPARATION)
       
        corners_3d = np.array([
            [start_x, start_y, 0],                         # TL
            [start_x + MARKER_LENGTH, start_y, 0],         # TR
            [start_x + MARKER_LENGTH, start_y + MARKER_LENGTH, 0], # BR
            [start_x, start_y + MARKER_LENGTH, 0]          # BL
        ], dtype=np.float32)
        
        object_points.append(corners_3d)
        
    return np.concatenate(object_points)


def calibrate_camera():
    print(f"--- Trazim slike: {CALIB_IMAGES_PATH} ---")
    images = glob.glob(CALIB_IMAGES_PATH)
    if not images: images = glob.glob('*.png')

    if len(images) < 2:
        print("GRESKA: Nema dovoljno slika!")
        return None, None

    print(f"Pronadjeno {len(images)} slika. Detekcija...")

    # detektor
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

   
    obj_points_all = [] # 3D tacke
    img_points_all = [] # 2D tacke sa slike
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        # Detekcija
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # 1. Spajamo sve coskove u jednu listu tacaka (N*4, 2)
            # detectMarkers vraca listu nizova, treba nam jedan niz
            current_img_points = np.concatenate(corners).reshape(-1, 2).astype(np.float32)
            
            # 2. Generisemo 3D tacke za TE DETEKTOVANE markere
            current_obj_points = get_board_object_points(ids)
            
            img_points_all.append(current_img_points)
            obj_points_all.append(current_obj_points)
        else:
            print(f" -> {fname}: Nema markera")

    if len(obj_points_all) == 0:
        print("Neuspesna detekcija.")
        return None, None

    print("Pokrecem standardnu cv2.calibrateCamera...")
    
    try:
        # Koristimo standardnu funkciju koja uvek postoji
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_all, 
            img_points_all, 
            img_shape, 
            None, 
            None
        )
        print("Kalibracija USPESNA!")
        print(f"Fokusna duzina (fx): {mtx[0,0]:.2f}")
        return mtx, dist
    except Exception as e:
        print(f"Greska: {e}")
        return None, None


def process_video(mtx, dist):
    print(f"\n--- Otvaram video: {VIDEO_FILENAME} ---")
    cap = cv2.VideoCapture(VIDEO_FILENAME)

    if not cap.isOpened():
        print(f"GRESKA: Video '{VIDEO_FILENAME}' nije pronadjen.")
        return

    # Detektor
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        
        # ispravljamo 
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # detekcija na ispravljenoj slici
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        output_image = undistorted.copy()

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(output_image, corners, ids)
            
            
            # sakupljamo 2D i pretvaramo u 3D
            img_pts_flat = np.concatenate(corners).reshape(-1, 2).astype(np.float32)
            obj_pts_flat = get_board_object_points(ids)
            
            # solvePnP trazi rotaciju i translaciju celog sistema
            ret, rvec, tvec = cv2.solvePnP(obj_pts_flat, img_pts_flat, newcameramtx, np.zeros((5,1)))
            
            if ret:
                # Crtanje osa
                try:
                    cv2.drawFrameAxes(output_image, newcameramtx, np.zeros((5,1)), rvec, tvec, 0.1)
                except AttributeError:
                    
                    try:
                        aruco.drawAxis(output_image, newcameramtx, np.zeros((5,1)), rvec, tvec, 0.1)
                    except:
                        pass 

        cv2.imshow('Lab Vezba: Kalibracija + SolvePnP', output_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_matrix, dist_coeffs = calibrate_camera()
    if camera_matrix is not None:
        process_video(camera_matrix, dist_coeffs)
    else:
        print("Prekid zbog greske u kalibraciji.")