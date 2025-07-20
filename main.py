import cv2
import numpy as np
from suspicious_detector import detect_suspicious_patterns

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print("❌ Image not found or invalid path.")
        exit()
    return image

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return eyes

def extract_and_enhance_eye(image, eyes):
    for (x, y, w, h) in eyes:
        eye_img = image[y:y+h, x:x+w]
        cv2.imwrite("cropped_eye.png", eye_img)

        # Enhance eye: sharpen and brighten
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(eye_img, -1, kernel)

        # Increase brightness
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        final_hsv = cv2.merge((h, s, v))
        bright_eye = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite("enhanced_eye.png", bright_eye)
        return bright_eye  # Return enhanced version of the first eye found

    print("❌ No eyes detected.")
    exit()

def main():
    print("🧠 Whispy AI — Lightweight Hidden Camera Reflection Detector")
    image_path = input("📸 Enter path to your face image: ")
    
    # Step 1: Load image
    image = load_image(image_path)
    print("[📷] Image loaded.")

    # Step 2: Detect and extract eyes
    eyes = detect_eyes(image)
    if len(eyes) == 0:
        print("❌ No eyes detected.")
        return

    enhanced_eye = extract_and_enhance_eye(image, eyes)
    print("[👁️] Eye extracted and saved as 'cropped_eye.png'.")
    print("[✨] Enhanced eye saved as 'enhanced_eye.png'.")

    # Step 3: Detect Suspicious Patterns
    score, reasons = detect_suspicious_patterns(enhanced_eye)

    print(f"\n[🔍] Suspicious Score: {score}")
    for r in reasons:
        print(" -", r)

    if score >= 2:
        print("\n[🚨] Potential risk detected in eye region!")
    else:
        print("\n[✅] Eye appears normal. No suspicious reflections detected.")

if __name__ == "__main__":
    main()
