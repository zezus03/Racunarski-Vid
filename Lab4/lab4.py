import cv2
import numpy as np

IMAGE_PATH = 'input.png' 
PROTOTXT_PATH = 'bvlc_googlenet.prototxt'
MODEL_PATH = 'bvlc_googlenet.caffemodel'
OUTPUT_PATH = 'output.jpg'

WIN_W, WIN_H = 180, 180
STEP_SIZE = 180          
CONFIDENCE_THRESHOLD = 0.30 


def is_dog(class_id):
    
    return 151 <= class_id <= 268

def is_cat(class_id):
    return 281 <= class_id <= 285

print("[INFO] Ucitavam sliku i model...")
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"GRESKA: Slika '{IMAGE_PATH}' nije pronadjena.")
    exit()

try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
except:
    print("GRESKA: Model fajlovi nisu pronadjeni.")
    exit()


orig_h, orig_w = image.shape[:2]
crop_w, crop_h = 1440, 720
pomeraj_ulevo = 130 

start_x = ((orig_w - crop_w) // 2) - pomeraj_ulevo
if start_x < 0: start_x = 0
start_y = (orig_h - crop_h) // 2
end_x = start_x + crop_w
end_y = start_y + crop_h

roi_image = image[start_y:end_y, start_x:end_x]
print(f"[INFO] Slika isecena na dimenzije: {roi_image.shape}")

print("[INFO] Analiza slike u toku...")

for y in range(0, crop_h, STEP_SIZE):
    for x in range(0, crop_w, STEP_SIZE):
        
        window = roi_image[y:y + WIN_H, x:x + WIN_W]
        
        if window.shape[0] != WIN_H or window.shape[1] != WIN_W:
            continue

        blob = cv2.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))
        net.setInput(blob)
        preds = net.forward()
        
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]

        
        if confidence >= CONFIDENCE_THRESHOLD:
            label = ""
            color = (0, 0, 0)
            
            if is_dog(idx):
                label = "DOG"
                color = (0, 255, 255) 
            elif is_cat(idx):
                label = "CAT"
                color = (0, 0, 255) 
            
            if label != "":
                cv2.rectangle(roi_image, (x, y), (x + WIN_W, y + WIN_H), color, 2)
                
                text_y = y + 20 if y + 20 < crop_h else y - 10
                cv2.putText(roi_image, label, (x + 10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f" > Detektovan {label} na poziciji ({x},{y}) - {confidence*100:.1f}%")

cv2.imwrite(OUTPUT_PATH, roi_image)
print(f"[INFO] Rezultat snimljen u '{OUTPUT_PATH}'")


cv2.imshow("Rezultat", roi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()