import mediapipe as mp
import cv2
import numpy as np # Kita akan butuh numpy untuk operasi gambar

# --- 1. Inisialisasi MediaPipe dan Utilitas ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- 2. Muat Gambar Avatar ---
# Ganti 'avatar.png' dengan nama file gambar Anda
avatar_img_path = 'D:\ITS\sem 5\pcv\projectvtuber\pala.png' 
try:
    avatar_original = cv2.imread(avatar_img_path, cv2.IMREAD_UNCHANGED)
    if avatar_original is None:
        raise FileNotFoundError(f"File gambar tidak ditemukan: {avatar_img_path}")
    print(f"Gambar avatar '{avatar_img_path}' berhasil dimuat.")
except FileNotFoundError as e:
    print(e)
    print("Pastikan gambar avatar.png ada di folder yang sama.")
    exit() # Keluar jika gambar tidak ditemukan

# --- 3. Inisialisasi Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# --- 4. Inisialisasi Holistic Model MediaPipe ---
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break
        
        # Balik frame secara horizontal untuk tampilan cermin
        frame = cv2.flip(frame, 1)

        # Recolor Feed (BGR ke RGB) untuk MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Buat deteksi dengan MediaPipe
        results = holistic.process(image)
        
        # Recolor image kembali ke BGR untuk ditampilkan
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # --- Bagian VTuber Sederhana: Menempatkan Avatar ---
        # Kita akan menggunakan koordinat landmark wajah untuk menempatkan avatar
        
        # Pastikan landmark wajah terdeteksi
        if results.face_landmarks:
            # landmark hidung (index 1 di FACEMESH_CONTOURS)
            # Anda bisa coba indeks lain seperti 0 (dahi) atau 6 (mulut)
            nose_landmark = results.face_landmarks.landmark[1] 
            
            # Ubah koordinat landmark (normalisasi 0-1) ke koordinat piksel
            # cv2.cvtColor(image, cv2.COLOR_RGB2BGR) menghasilkan image berukuran sama dengan frame
            img_h, img_w, _ = image.shape
            nose_x = int(nose_landmark.x * img_w)
            nose_y = int(nose_landmark.y * img_h)
            
            # --- Menyesuaikan Ukuran Avatar ---
            # Mari kita buat avatar mengikuti skala wajah (opsional, bisa juga fixed size)
            # Misalnya, kita asumsikan lebar avatar sekitar 1.5x lebar hidung ke pipi
            # Untuk sederhana, kita asumsikan lebar avatar konstan dulu.
            
            # Tentukan ukuran avatar yang diinginkan (misal, 150x150 piksel)
            avatar_width = 150
            avatar_height = 150
            
            # Resize avatar
            avatar_resized = cv2.resize(avatar_original, (avatar_width, avatar_height), interpolation=cv2.INTER_AREA)
            
            # Hitung posisi (top-left corner) avatar
            # Kita ingin hidung ada di tengah bawah avatar, atau di bagian tertentu
            # Misalnya, letakkan avatar agar bagian bawahnya di hidung, dan tengahnya di hidung
            
            # Posisi x: center avatar di hidung
            x_offset = nose_x - avatar_width // 2
            # Posisi y: letakkan bagian bawah avatar di hidung
            y_offset = nose_y - avatar_height 
            
            # Pastikan avatar tidak keluar dari batas layar
            x_offset = max(0, min(x_offset, img_w - avatar_width))
            y_offset = max(0, min(y_offset, img_h - avatar_height))
            
            # --- Menempatkan Avatar ke Frame ---
            # Kita perlu menumpuk gambar transparan. Ini agak tricky.
            # Referensi: https://answers.opencv.org/question/25529/how-to-put-a-png-image-with-transparent-background-on-another-image-in-opencv-python/
            
            # Ambil bagian frame yang akan ditutupi avatar
            roi = image[y_offset : y_offset + avatar_height, x_offset : x_offset + avatar_width]
            
            # Buat mask dari channel alpha avatar (jika ada)
            # Jika avatar Anda tidak punya channel alpha (RGBA), Anda perlu buat sendiri
            if avatar_resized.shape[2] == 4: # RGBA
                alpha_channel = avatar_resized[:, :, 3]
                mask = alpha_channel / 255.0
                alpha_factor = 1.0 - mask
            else: # RGB, anggap tidak transparan (atau Anda bisa tentukan warna kunci)
                mask = np.ones((avatar_height, avatar_width), dtype=float)
                alpha_factor = 0.0 # Tidak ada transparansi, gambar menimpa sepenuhnya

            # Gabungkan gambar (Blending)
            for c in range(0, 3):
                # Campurkan RGB channel dari avatar
                roi[:, :, c] = (alpha_factor * roi[:, :, c] +
                                mask * avatar_resized[:, :, c])
            
        # --- Menggambar Landmark MediaPipe (opsional, bisa dinonaktifkan nanti) ---
        # Ini adalah kode lama Anda untuk menggambar titik-titik
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                                 
        cv2.imshow('VTuber Sederhana', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()