"""
================================================================================
VTUBER 2D: LAYER FIX (LENGAN BAWAH DI DEPAN TORSO)
================================================================================
Urutan Layering Baru:
1. Lengan Atas (Background) -> Agar tertutup baju.
2. Torso (Middle) -> Badan utama.
3. Lengan Bawah (Foreground) -> Agar tangan muncul di depan baju saat menyilang.

Fitur Auto-Scaling (Jarak Jauh/Dekat) tetap aktif.
"""

import mediapipe as mp
import cv2
import numpy as np
import math

# ==========================================
# 1. SETUP & LOAD GAMBAR
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

FILE_LENGAN_ATAS = 'lengan_atas.png'
FILE_LENGAN_BAWAH = 'lengan_bawah.png'
FILE_TORSO = 'torso.png'

print("Loading aset...")
try:
    img_lengan_atas = cv2.imread(FILE_LENGAN_ATAS, cv2.IMREAD_UNCHANGED)
    img_lengan_bawah = cv2.imread(FILE_LENGAN_BAWAH, cv2.IMREAD_UNCHANGED)
    img_torso = cv2.imread(FILE_TORSO, cv2.IMREAD_UNCHANGED)

    if img_lengan_atas is None or img_torso is None:
        print("Error: File gambar tidak ditemukan."); exit()
    if img_lengan_bawah is None: img_lengan_bawah = img_lengan_atas.copy()

except Exception as e:
    print("Error load gambar:", e); exit()


# ==========================================
# 2. FUNGSI LOGIKA (AUTO-SCALING AKTIF)
# ==========================================

def overlay_limb_final(canvas, img_asset, point_a, point_b, length_factor=1.0, thickness_ratio=0.3, mirror=False):
    """
    thickness_ratio: Rasio ketebalan terhadap panjang lengan (0.2 - 0.4).
    Membuat tangan mengecil otomatis saat menjauh.
    """
    if img_asset is None: return canvas
    
    dist = math.dist(point_a, point_b)
    if dist < 5: return canvas

    delta_y = point_b[1] - point_a[1]
    delta_x = point_b[0] - point_a[0]
    angle_rad = math.atan2(delta_y, delta_x)

    if mirror: img_used = cv2.flip(img_asset, 0)
    else: img_used = img_asset

    h_orig, w_orig = img_used.shape[:2]

    # LOGIKA SCALING DINAMIS
    target_width = int(dist * length_factor)
    target_height = int(dist * thickness_ratio) # Tebal ikut jarak

    if target_width < 1 or target_height < 1: return canvas

    center_y = h_orig * 0.5 
    src_pts = np.float32([[0, center_y], [w_orig, center_y], [0, h_orig]])

    angle_perpendicular = angle_rad + math.pi / 2
    
    dst_p1 = np.array(point_a, dtype=np.float32)
    dst_p2 = dst_p1 + np.array([
        math.cos(angle_rad) * target_width,
        math.sin(angle_rad) * target_width
    ], dtype=np.float32)

    offset_thickness = target_height * 0.5
    dst_p3 = dst_p1 + np.array([
        math.cos(angle_perpendicular) * offset_thickness,
        math.sin(angle_perpendicular) * offset_thickness
    ], dtype=np.float32)

    dst_pts = np.float32([dst_p1, dst_p2, dst_p3])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    h_c, w_c = canvas.shape[:2]
    warped_img = cv2.warpAffine(img_used, M, (w_c, h_c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    try:
        alpha_mask = warped_img[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            canvas[:, :, c] = (alpha_inv * canvas[:, :, c] + alpha_mask * warped_img[:, :, c])
    except Exception: pass 
    return canvas


def overlay_torso_central(canvas, img_asset, l_shoulder, r_shoulder, l_hip, r_hip, width_scale=1.0, height_scale=1.0, vertical_shift=0.0):
    if img_asset is None: return canvas

    shoulder_mid = np.array([(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2], dtype=np.float32)
    hip_mid = np.array([(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2], dtype=np.float32)

    delta_y = hip_mid[1] - shoulder_mid[1]
    delta_x = hip_mid[0] - shoulder_mid[0]
    angle_spine = math.atan2(delta_y, delta_x)
    angle_perpendicular = angle_spine - math.pi / 2 
    
    shoulder_dist = math.dist(l_shoulder, r_shoulder)
    spine_dist = math.dist(shoulder_mid, hip_mid)

    target_width = int(shoulder_dist * width_scale)
    target_height = int(spine_dist * height_scale)

    if target_width < 1 or target_height < 1: return canvas
    h_orig, w_orig = img_asset.shape[:2]

    src_pts = np.float32([[w_orig // 2, 0], [w_orig // 2, h_orig], [0, 0]])

    shift_pixel = spine_dist * vertical_shift
    
    dst_p1 = shoulder_mid - np.array([math.cos(angle_spine) * shift_pixel, math.sin(angle_spine) * shift_pixel], dtype=np.float32)
    dst_p2 = dst_p1 + np.array([math.cos(angle_spine) * target_height, math.sin(angle_spine) * target_height], dtype=np.float32)
    dst_p3 = dst_p1 + np.array([math.cos(angle_perpendicular) * (target_width / 2), math.sin(angle_perpendicular) * (target_width / 2)], dtype=np.float32)

    dst_pts = np.float32([dst_p1, dst_p2, dst_p3])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    h_c, w_c = canvas.shape[:2]
    warped_img = cv2.warpAffine(img_asset, M, (w_c, h_c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    try:
        alpha_mask = warped_img[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            canvas[:, :, c] = (alpha_inv * canvas[:, :, c] + alpha_mask * warped_img[:, :, c])
    except Exception: pass
    
    return canvas


# ==========================================
# 3. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def p(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))

            if lm[11].visibility < 0.5 or lm[12].visibility < 0.5:
                cv2.imshow('VTuber Final', frame); 
                if cv2.waitKey(5) & 0xFF == ord('q'): break; continue

            ls, rs = p(11), p(12)
            
            # Virtual Hips Logic
            if (lm[23].visibility > 0.5) and (lm[24].visibility > 0.5):
                lh, rh = p(23), p(24)
            else:
                shoulder_mid_x = (ls[0] + rs[0]) / 2
                shoulder_mid_y = (ls[1] + rs[1]) / 2
                shoulder_width = math.dist(ls, rs)
                estimated_spine = shoulder_width * 1.6 
                v_hip_y = int(shoulder_mid_y + estimated_spine)
                lh = (int(shoulder_mid_x + (shoulder_width * 0.5)), v_hip_y)
                rh = (int(shoulder_mid_x - (shoulder_width * 0.5)), v_hip_y)

            # ==========================================
            # TUNING PARAMETER
            # ==========================================
            SET_PANJANG_LENGAN = 1.2 
            SET_RASIO_TEBAL = 0.4  # Jika tangan kegemukan, turunkan ini (misal 0.3)
            
            SET_LEBAR_TORSO = 2.0 
            SET_TINGGI_TORSO = 1.8 
            SET_POSISI_NAIK = 0.45 

            # ==========================================
            # PROSES RENDERING (LAYER URUTAN BARU)
            # ==========================================
            
            # --- LAYER 1: LENGAN ATAS (PALING BELAKANG) ---
            # Digambar duluan agar tertimpa oleh baju
            
            # Kiri Atas (Mirror True)
            frame = overlay_limb_final(frame, img_lengan_atas, ls, p(13), SET_PANJANG_LENGAN, SET_RASIO_TEBAL, mirror=True)
            # Kanan Atas (Mirror False)
            frame = overlay_limb_final(frame, img_lengan_atas, rs, p(14), SET_PANJANG_LENGAN, SET_RASIO_TEBAL, mirror=False)

            # --- LAYER 2: TORSO (TENGAH) ---
            # Digambar di tengah agar menutupi sambungan bahu lengan atas
            frame = overlay_torso_central(frame, img_torso, ls, rs, lh, rh, SET_LEBAR_TORSO, SET_TINGGI_TORSO, SET_POSISI_NAIK)

            # --- LAYER 3: LENGAN BAWAH (PALING DEPAN) ---
            # Digambar terakhir agar muncul di atas baju (bisa pose 'love' atau tangan di dada)
            
            # Kiri Bawah
            frame = overlay_limb_final(frame, img_lengan_bawah, p(13), p(15), SET_PANJANG_LENGAN, SET_RASIO_TEBAL, mirror=True)
            # Kanan Bawah
            frame = overlay_limb_final(frame, img_lengan_bawah, p(14), p(16), SET_PANJANG_LENGAN, SET_RASIO_TEBAL, mirror=False)

        cv2.imshow('VTuber Final', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()