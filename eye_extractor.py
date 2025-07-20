import cv2

def extract_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    if len(eyes) == 0:
        print("[⚠️] No eyes detected.")
        return image

    # Get the first detected eye
    (x, y, w, h) = eyes[0]
    cropped_eye = image[y:y + h, x:x + w]
    return cropped_eye
