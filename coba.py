import mediapipe as mp
import cv2
import numpy as np
import math

# Import module lokal
from src import config as cfg
from src import tracking
from src import renderer
from src.utils import load_image_safe
from src.stabilizer import StabilizerManager 

# ==============================================================================
# INITIALIZATION
# ==============================================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

print("Loading aset...")
try:
    img_lengan_atas  = cv2.imread(cfg.FILE_LENGAN_ATAS, cv2.IMREAD_UNCHANGED)
    img_lengan_bawah = cv2.imread(cfg.FILE_LENGAN_BAWAH, cv2.IMREAD_UNCHANGED)
    img_torso        = cv2.imread(cfg.FILE_TORSO, cv2.IMREAD_UNCHANGED)
    img_head         = cv2.imread(cfg.FILE_HEAD, cv2.IMREAD_UNCHANGED)
    
    img_bahu_kiri    = cv2.imread(cfg.FILE_BAHU_KIRI, cv2.IMREAD_UNCHANGED)
    img_bahu_kanan   = cv2.imread(cfg.FILE_BAHU_KANAN, cv2.IMREAD_UNCHANGED)
    
    img_mata_buka_ki  = cv2.imread(cfg.FILE_MATA_BUKA_KI, cv2.IMREAD_UNCHANGED)
    img_mata_buka_ka  = cv2.imread(cfg.FILE_MATA_BUKA_KA, cv2.IMREAD_UNCHANGED)
    img_mata_tutup_ki = cv2.imread(cfg.FILE_MATA_TUTUP_KI, cv2.IMREAD_UNCHANGED)
    img_mata_tutup_ka = cv2.imread(cfg.FILE_MATA_TUTUP_KA, cv2.IMREAD_UNCHANGED)

    img_mulut_buka   = cv2.imread(cfg.FILE_MULUT_BUKA, cv2.IMREAD_UNCHANGED)
    img_gigi         = cv2.imread(cfg.FILE_GIGI, cv2.IMREAD_UNCHANGED) 

    img_celana_atas  = cv2.imread(cfg.FILE_CELANA_ATAS, cv2.IMREAD_UNCHANGED)
    img_celana_bawah = cv2.imread(cfg.FILE_CELANA_BAWAH, cv2.IMREAD_UNCHANGED)
    
    img_bg = cv2.imread(cfg.FILE_BG) 
    if img_bg is None: img_bg = np.zeros((480, 640, 3), dtype=np.uint8)

    img_hand_open  = load_image_safe(cfg.FILE_HAND_OPEN)
    img_hand_relax = load_image_safe(cfg.FILE_HAND_RELAX)
    if img_hand_relax is not None: img_hand_relax = cv2.flip(img_hand_relax, 0) 

    if (img_head is None) or (img_mata_buka_ki is None):
        print("ERROR: Aset utama tidak ditemukan!"); exit()

    # Fallbacks
    if img_mata_tutup_ki is not None:
        h_m, w_m = img_mata_buka_ki.shape[:2]
        img_mata_tutup_ki = cv2.resize(img_mata_tutup_ki, (w_m, h_m), interpolation=cv2.INTER_AREA)
    else: img_mata_tutup_ki = img_mata_buka_ki.copy()

    if img_mata_tutup_ka is not None:
        h_m, w_m = img_mata_buka_ka.shape[:2]
        img_mata_tutup_ka = cv2.resize(img_mata_tutup_ka, (w_m, h_m), interpolation=cv2.INTER_AREA)
    else: img_mata_tutup_ka = img_mata_buka_ka.copy()

    if img_lengan_bawah is None: img_lengan_bawah = img_lengan_atas.copy()

except Exception as e:
    print(f"Error loading assets: {e}"); exit()

# ==============================================================================
# MAIN LOOP
# ==============================================================================
cap = cv2.VideoCapture(0)

# Inisialisasi Stabilizer (Alpha kecil = Halus)
stabs = StabilizerManager(alpha=0.3)

with mp_holistic.Holistic(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_face_landmarks=True 
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        debug_frame = frame.copy() 
        if img_bg is not None: vtuber_frame = cv2.resize(img_bg, (w, h))
        else: vtuber_frame = np.zeros((h, w, 3), dtype=np.uint8)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(debug_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # --- POSE LANDMARKS (STABILIZED) ---
            lm_pose = results.pose_landmarks.landmark
            def p_pose(idx): 
                return (int(lm_pose[idx].x * w), int(lm_pose[idx].y * h))

            # Stabilisasi Bahu dan Siku
            ls = stabs.get_stable('ls', p_pose(11))
            rs = stabs.get_stable('rs', p_pose(12))
            nose = stabs.get_stable('nose', p_pose(0)) 
            
            le, re = p_pose(7), p_pose(8)
            left_elbow = stabs.get_stable('l_elbow', p_pose(13))
            right_elbow = stabs.get_stable('r_elbow', p_pose(14))
            
            # Stabilisasi Lengan Bawah
            l_wrist_pose = stabs.get_stable('l_wrist_pose', p_pose(15))
            r_wrist_pose = stabs.get_stable('r_wrist_pose', p_pose(16))
            
            # Kaki
            lh, rh = stabs.get_stable('lh', p_pose(23)), stabs.get_stable('rh', p_pose(24))
            lk, rk = stabs.get_stable('lk', p_pose(25)), stabs.get_stable('rk', p_pose(26))
            la, ra = stabs.get_stable('la', p_pose(27)), stabs.get_stable('ra', p_pose(28))

            if (lm_pose[23].visibility < 0.5) or (lm_pose[24].visibility < 0.5):
                s_mid_x, s_mid_y = (ls[0]+rs[0])/2, (ls[1]+rs[1])/2
                s_width = math.dist(ls, rs)
                v_hip_y = int(s_mid_y + s_width * 1.6)
                lh = (int(s_mid_x + s_width*0.5), v_hip_y)
                rh = (int(s_mid_x - s_width*0.5), v_hip_y)
                lk = (lh[0], lh[1] + int(s_width*1.2))
                rk = (rh[0], rh[1] + int(s_width*1.2))
                la = (lk[0], lk[1] + int(s_width*1.2))
                ra = (rk[0], rk[1] + int(s_width*1.2))

            # --- HAND GESTURES  ---
            # Helper mengambil titik tangan dan menstabilkannya
            def get_hand_points(landmarks, label):
                if not landmarks: return None, None
                lm = landmarks.landmark
                raw_wrist = (int(lm[0].x * w), int(lm[0].y * h))
                raw_middle = (int(lm[9].x * w), int(lm[9].y * h))
                stab_wrist = stabs.get_stable(f'{label}_wrist', raw_wrist)
                stab_middle = stabs.get_stable(f'{label}_middle', raw_middle)
                return stab_wrist, stab_middle

            # Kanan
            hr_wrist, hr_middle = get_hand_points(results.left_hand_landmarks, 'hand_r')
            
            # Kiri 
            hl_wrist, hl_middle = get_hand_points(results.right_hand_landmarks, 'hand_l')

            # --- FACE ---
            if results.face_landmarks:
                lm_face = results.face_landmarks.landmark
                def p_face(idx): return (int(lm_face[idx].x * w), int(lm_face[idx].y * h))
                nose = stabs.get_stable('nose_face', p_face(1)) 
                p_eye_r_1 = stabs.get_stable('eye_r_1', p_face(33))
                p_eye_r_2 = stabs.get_stable('eye_r_2', p_face(133))
                p_eye_l_1 = stabs.get_stable('eye_l_1', p_face(362))
                p_eye_l_2 = stabs.get_stable('eye_l_2', p_face(263))
                p_mouth_1 = stabs.get_stable('mouth_1', p_face(61))
                p_mouth_2 = stabs.get_stable('mouth_2', p_face(291))
            else:
                p_eye_r_1, p_eye_r_2 = (0,0), (0,0)
                p_eye_l_1, p_eye_l_2 = (0,0), (0,0)
                p_mouth_1, p_mouth_2 = (0,0), (0,0)

            # Celana
            offset_pinggul_px = int(math.dist(ls, rs) * cfg.SET_OFFSET_PINGGUL_NAIK)
            offset_lutut_px   = int(math.dist(ls, rs) * cfg.SET_OFFSET_LUTUT_NAIK)
            lh_mod = (lh[0], lh[1] - offset_pinggul_px)
            rh_mod = (rh[0], rh[1] - offset_pinggul_px)
            lk_mod = (lk[0], lk[1] - offset_lutut_px)
            rk_mod = (rk[0], rk[1] - offset_lutut_px)

            # =========================================================
            # RENDER LAYER
            # =========================================================
            
            # 1. CELANA
            vtuber_frame = renderer.overlay_leg_vertical(vtuber_frame, img_celana_bawah, lk_mod, la, cfg.SET_THICKNESS_BETIS, cfg.SET_SCALE_CELANA_BAWAH_H)
            vtuber_frame = renderer.overlay_leg_vertical(vtuber_frame, img_celana_bawah, rk_mod, ra, cfg.SET_THICKNESS_BETIS, cfg.SET_SCALE_CELANA_BAWAH_H)
            vtuber_frame = renderer.overlay_leg_vertical(vtuber_frame, img_celana_atas, lh_mod, lk_mod, cfg.SET_THICKNESS_PAHA, cfg.SET_SCALE_CELANA_ATAS_H)
            vtuber_frame = renderer.overlay_leg_vertical(vtuber_frame, img_celana_atas, rh_mod, rk_mod, cfg.SET_THICKNESS_PAHA, cfg.SET_SCALE_CELANA_ATAS_H)

            # 2. TANGAN 
            # Kanan
            hand_state_r = tracking.get_hand_gesture(results.left_hand_landmarks)
            img_hand_r, mirror_r = (img_hand_open, False) if hand_state_r == 2 else (img_hand_relax, True)
            vtuber_frame = renderer.overlay_hand_dynamic(vtuber_frame, img_hand_r, hr_wrist, hr_middle, scale=cfg.SET_SCALE_TANGAN, mirror=mirror_r, rotation_offset=cfg.SET_ROTASI_TANGAN_OFFSET)
            
            # Kiri
            hand_state_l = tracking.get_hand_gesture(results.right_hand_landmarks)
            img_hand_l, mirror_l = (img_hand_open, True) if hand_state_l == 2 else (img_hand_relax, False)
            vtuber_frame = renderer.overlay_hand_dynamic(vtuber_frame, img_hand_l, hl_wrist, hl_middle, scale=cfg.SET_SCALE_TANGAN, mirror=mirror_l, rotation_offset=-cfg.SET_ROTASI_TANGAN_OFFSET)

            # 3. BODY UPPER
            vtuber_frame = renderer.overlay_torso_central(vtuber_frame, img_torso, ls, rs, lh, rh, cfg.SET_LEBAR_TORSO, cfg.SET_TINGGI_TORSO, cfg.SET_TORSO_SHIFT_Y)

            vtuber_frame = renderer.overlay_limb_final(vtuber_frame, img_lengan_atas, ls, left_elbow, cfg.SET_PANJANG_LENGAN, 0.35, mirror=True)
            vtuber_frame = renderer.overlay_limb_final(vtuber_frame, img_lengan_atas, rs, right_elbow, cfg.SET_PANJANG_LENGAN, 0.35, mirror=False)
            
            s_width_px = math.dist(ls, rs)
            ls_bahu = (ls[0] + cfg.SET_OFFSET_BAHU_X, ls[1] + cfg.SET_OFFSET_BAHU_Y)
            rs_bahu = (rs[0] - cfg.SET_OFFSET_BAHU_X, rs[1] + cfg.SET_OFFSET_BAHU_Y) 
            
            vtuber_frame = renderer.overlay_shoulder_rotatable(vtuber_frame, img_bahu_kiri, ls_bahu, left_elbow, width_ref=s_width_px, scale=cfg.SET_SCALE_BAHU, flip_h=True, anchor_y=cfg.SET_PIVOT_BAHU_Y)
            vtuber_frame = renderer.overlay_shoulder_rotatable(vtuber_frame, img_bahu_kanan, rs_bahu, right_elbow, width_ref=s_width_px, scale=cfg.SET_SCALE_BAHU, flip_h=True, anchor_y=cfg.SET_PIVOT_BAHU_Y)
            
            # Lengan Bawah
            vtuber_frame = renderer.overlay_limb_final(vtuber_frame, img_lengan_bawah, left_elbow, l_wrist_pose, cfg.SET_PANJANG_LENGAN, 0.55, mirror=True)
            vtuber_frame = renderer.overlay_limb_final(vtuber_frame, img_lengan_bawah, right_elbow, r_wrist_pose, cfg.SET_PANJANG_LENGAN, 0.55, mirror=False)

            # 4. WAJAH
            vtuber_frame = renderer.overlay_face_part(vtuber_frame, img_head, nose, ls, rs, le, re, cfg.SET_SCALE_KEPALA, cfg.SET_OFFSET_HIDUNG, flip_v=True)

            mouth_state = tracking.get_mouth_state(results.face_landmarks, w, h)
            if mouth_state == 2 and (img_mulut_buka is not None):
                vtuber_frame = renderer.overlay_mouth(vtuber_frame, img_mulut_buka, p_mouth_1, p_mouth_2, scale=cfg.SET_SCALE_MULUT, y_offset=cfg.SET_OFFSET_MULUT)
            elif mouth_state == 1 and (img_gigi is not None):
                vtuber_frame = renderer.overlay_mouth(vtuber_frame, img_gigi, p_mouth_1, p_mouth_2, scale=cfg.SET_SCALE_GIGI, y_offset=cfg.SET_OFFSET_GIGI)
            
            kanan_buka, kiri_buka = tracking.get_eye_state(results.face_landmarks, w, h)
            if kanan_buka: vtuber_frame = renderer.overlay_single_eye(vtuber_frame, img_mata_buka_ki, p_eye_r_1, p_eye_r_2, scale=cfg.SET_SCALE_MATA_BUKA, y_offset=cfg.SET_OFFSET_MATA, x_offset=cfg.SET_OFFSET_MATA_X_KANAN)
            else: vtuber_frame = renderer.overlay_single_eye(vtuber_frame, img_mata_tutup_ki, p_eye_r_1, p_eye_r_2, scale=cfg.SET_SCALE_MATA_TUTUP, y_offset=cfg.SET_OFFSET_MATA, x_offset=cfg.SET_OFFSET_MATA_X_KANAN)
            
            if kiri_buka: vtuber_frame = renderer.overlay_single_eye(vtuber_frame, img_mata_buka_ka, p_eye_l_1, p_eye_l_2, scale=cfg.SET_SCALE_MATA_BUKA, y_offset=cfg.SET_OFFSET_MATA, x_offset=cfg.SET_OFFSET_MATA_X_KIRI)
            else: vtuber_frame = renderer.overlay_single_eye(vtuber_frame, img_mata_tutup_ka, p_eye_l_1, p_eye_l_2, scale=cfg.SET_SCALE_MATA_TUTUP, y_offset=cfg.SET_OFFSET_MATA, x_offset=cfg.SET_OFFSET_MATA_X_KIRI)

        cv2.imshow('Tracking Debug', debug_frame)
        cv2.imshow('VTuber Final', vtuber_frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()