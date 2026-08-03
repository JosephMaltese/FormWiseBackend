import numpy as np
import math

def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Midpoint
    c = np.array(c) # Endpoint

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle

def eccentric_score(duration):
    if duration < 0.5:
        return 0
    # Gaussian curve centered at 3s, width 1s
    return round(10 * math.exp(-((duration - 3)**2) / (2 * (1**2))))