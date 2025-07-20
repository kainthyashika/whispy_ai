import cv2
import numpy as np

def detect_suspicious_patterns(image):
    suspicious_score = 0
    reasons = []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Check for very bright spots (possible lens or reflection)
    bright_pixels = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    bright_ratio = np.sum(bright_pixels == 255) / bright_pixels.size

    if bright_ratio > 0.02:
        suspicious_score += 1
        reasons.append("⚠️ Unusually bright pixels detected")

    # 2. Check for grid-like patterns (screens)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge_strength = np.mean(np.abs(sobelx) + np.abs(sobely))

    if edge_strength > 60:
        suspicious_score += 1
        reasons.append("⚠️ Screen-like edge patterns detected")

    # 3. Check for abnormal contrast
    contrast = gray.std()
    if contrast > 70:
        suspicious_score += 1
        reasons.append("⚠️ High contrast — may indicate artificial light or manipulation")

    return suspicious_score, reasons
