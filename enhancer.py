import cv2
import numpy as np

def enhance_eye(image):
    # Resize eye for better visibility
    image = cv2.resize(image, (200, 200))

    # Apply sharpening kernel
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)

    # Convert to HSV for brightness/contrast adjustment
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Boost brightness and contrast slightly
    v = cv2.add(v, 30)
    final_hsv = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return enhanced
