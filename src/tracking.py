import numpy as np
import math

def get_eye_state(face_landmarks, w, h):
    """ Return: (Kanan_Buka?, Kiri_Buka?) """
    if not face_landmarks: return (False, False)
    lm = face_landmarks.landmark
    def dist(idx1, idx2):
        p1 = np.array([lm[idx1].x * w, lm[idx1].y * h])
        p2 = np.array([lm[idx2].x * w, lm[idx2].y * h])
        return np.linalg.norm(p1 - p2)
    
    # Mata Kanan
    r_vert = dist(159, 145); r_horz = dist(33, 133)
    if r_horz < 1: r_horz = 1
    right_open = (r_vert / r_horz) > 0.13

    # Mata Kiri
    l_vert = dist(386, 374); l_horz = dist(362, 263)
    if l_horz < 1: l_horz = 1
    left_open = (l_vert / l_horz) > 0.13
    return (right_open, left_open)

def get_mouth_state(face_landmarks, w, h):
    """ 0=Tutup, 1=Senyum, 2=Buka """
    if not face_landmarks: return 0
    lm = face_landmarks.landmark
    def dist(idx1, idx2):
        p1 = np.array([lm[idx1].x * w, lm[idx1].y * h])
        p2 = np.array([lm[idx2].x * w, lm[idx2].y * h])
        return np.linalg.norm(p1 - p2)
    
    vertical = dist(13, 14); horizontal = dist(61, 291)
    if horizontal < 1: horizontal = 1
    ratio = vertical / horizontal
    if ratio > 0.30: return 2
    elif ratio > 0.08: return 1
    else: return 0

def get_hand_gesture(hand_landmarks):
    """
    Detect Hand State: 1 = Relax, 2 = Open Palm
    """
    if not hand_landmarks: return 1 
    lm = hand_landmarks.landmark
    
    def dist(i1, i2):
        return math.hypot(lm[i1].x - lm[i2].x, lm[i1].y - lm[i2].y)

    hand_size = dist(0, 9)
    if hand_size == 0: return 1

    wrist = 0
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]

    open_fingers = 0
    for tip, mcp in zip(tips, mcps):
        if dist(tip, wrist) > dist(mcp, wrist):
             open_fingers += 1
    
    spread = dist(4, 20) 
    spread_ratio = spread / hand_size
    
    if open_fingers >= 4 and spread_ratio > 1.2: 
        return 2 # Open Palm
    
    return 1 # Relax