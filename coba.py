import mediapipe as mp
import cv2
import numpy as np
import math

# --- 1. Inisialisasi ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- 2. Load Gambar (WAJIB PNG HORIZONTAL dengan SENDI DI KIRI) ---
# Pastikan file ini ada di folder yang sama
file_lengan_atas = 'lengan_atas.png'
file_lengan_bawah = 'lengan_bawah.png'

try:
    img_lengan_atas = cv2.imread(file_lengan_atas, cv2.IMREAD_UNCHANGED)
    img_lengan_bawah = cv2.imread(file_lengan_bawah, cv2.IMREAD_UNCHANGED)

    if img_lengan_atas is None:
        print(f"Error: {file_lengan_atas} tidak ditemukan."); exit()
    if img_lengan_atas.shape[2] < 4:
        print("Error: Gambar harus PNG dengan background transparan (4 channel)."); exit()

    if img_lengan_bawah is None: img_lengan_bawah = img_lengan_atas.copy()

except Exception as e:
    print("Error system saat load gambar:", e)
    exit()


# --- 3. FUNGSI OVERLAY FINAL (MIRROR FIX) ---
def overlay_limb_final(canvas, img_asset, point_a, point_b, length_factor=1.0, thickness_scale=1.0, mirror=False):
    if img_asset is None: return canvas

    # A. Hitung Jarak & Sudut
    dist = math.dist(point_a, point_b)
    if dist < 5: return canvas

    delta_y = point_b[1] - point_a[1]
    delta_x = point_b[0] - point_a[0]
    angle_rad = math.atan2(delta_y, delta_x)

    # B. Siapkan Gambar (Mirroring)
    # Flip Vertikal (0) untuk tangan kiri vs kanan agar jempol/otot tidak terbalik
    if mirror:
        img_used = cv2.flip(img_asset, 0) 
    else:
        img_used = img_asset

    # C. Hitung Ukuran Target
    h_orig, w_orig = img_used.shape[:2]
    target_width = int(dist * length_factor)
    target_height = int(h_orig * thickness_scale)

    if target_width < 1 or target_height < 1: return canvas

    # ============================================================
    # PERUBAHAN UTAMA ADA DI SINI (TITIK TUMPU)
    # ============================================================
    
    # D. Tentukan Titik Sumber (Source Points) RELATIF TERHADAP TENGAH GAMBAR
    # Kita ambil garis tengah (centerline) sebagai poros "tulang"
    center_y = h_orig * 0.5 
    
    src_pts = np.float32([
        [0, center_y],          # P1: Pivot ada di TENGAH KIRI (bukan (0,0))
        [w_orig, center_y],     # P2: Ujung ada di TENGAH KANAN
        [0, h_orig]             # P3: Titik Bawah Kiri (untuk referensi tebal)
    ])

    # E. Tentukan Titik Tujuan (Destination Points)
    angle_perpendicular = angle_rad + math.pi / 2
    
    # P1' = Ditempel pas di sendi (point_a)
    dst_p1 = np.array(point_a, dtype=np.float32)

    # P2' = point_a + vektor arah lengan (sepanjang target_width)
    dst_p2 = dst_p1 + np.array([
        math.cos(angle_rad) * target_width,
        math.sin(angle_rad) * target_width
    ], dtype=np.float32)

    # P3' = point_a + vektor tegak lurus (setengah ketebalan)
    # Karena src P3 (h_orig) jaraknya adalah 0.5 * tinggi dari P1 (center_y),
    # Maka dst P3 juga harus berjarak 0.5 * target_height dari dst P1.
    offset_thickness = target_height * 0.5
    
    dst_p3 = dst_p1 + np.array([
        math.cos(angle_perpendicular) * offset_thickness,
        math.sin(angle_perpendicular) * offset_thickness
    ], dtype=np.float32)

    dst_pts = np.float32([dst_p1, dst_p2, dst_p3])

    # F. Transformasi Affine (Warping)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    h_c, w_c = canvas.shape[:2]
    
    warped_img = cv2.warpAffine(img_used, M, (w_c, h_c),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0,0,0,0))

    # G. Alpha Blending (Tetap sama)
    try:
        alpha_mask = warped_img[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            canvas[:, :, c] = (alpha_inv * canvas[:, :, c] + alpha_mask * warped_img[:, :, c])
    except Exception:
        pass 

    return canvas

# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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

            # ==========================================
            # SETTINGAN (TUNING)
            # ==========================================
            SET_PANJANG = 1.2
            SET_TEBAL = 0.4

            # --- TANGAN KANAN (PAKAI MIRROR = True) ---
            # Bahu Kanan(12) -> Siku Kanan(14)
            frame = overlay_limb_final(frame, img_lengan_atas, p(12), p(14), SET_PANJANG, SET_TEBAL, mirror=False)
            # Siku Kanan(14) -> Pergelangan Kanan(16)
            frame = overlay_limb_final(frame, img_lengan_bawah, p(14), p(16), SET_PANJANG, SET_TEBAL, mirror=False)

            # --- TANGAN KIRI (PAKAI MIRROR = False) ---
            # Bahu Kiri(11) -> Siku Kiri(13)
            frame = overlay_limb_final(frame, img_lengan_atas, p(11), p(13), SET_PANJANG, SET_TEBAL, mirror=True)
            # Siku Kiri(13) -> Pergelangan Kiri(15)
            frame = overlay_limb_final(frame, img_lengan_bawah, p(13), p(15), SET_PANJANG, SET_TEBAL, mirror=True)

        cv2.imshow('VTuber Final Tracking', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()