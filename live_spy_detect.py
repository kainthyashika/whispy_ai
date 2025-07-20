import cv2
import numpy as np

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_glint(eye_region):
    # Detect very bright points (potential reflections)
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    count = cv2.countNonZero(thresh)
    return count

cap = cv2.VideoCapture(0)
print("[INFO] Live Webcam Spy Glint Detection Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    threat_detected = False

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            glint = detect_glint(eye_roi)

            # If too many bright pixels — suspicious
            if glint > 15:
                threat_detected = True
                cv2.putText(frame, "⚠️ SPY GLINT DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)

    if threat_detected:
        cv2.putText(frame, "⚠️ Suspicious Reflection Detected!", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Live Spy Reflection Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
