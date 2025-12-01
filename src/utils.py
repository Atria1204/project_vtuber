import cv2
import numpy as np

def load_image_safe(path):
    """ Load gambar & konversi png putih ke Transparan """
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: 
            return None
        
        # Jika gambar 3 channel (JPG/BGR), tambahkan Alpha channel
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            # Simple Chroma Key: Ubah warna putih (230-255) jadi transparan
            white_thresh = 230
            mask = (img[:, :, 0] > white_thresh) & (img[:, :, 1] > white_thresh) & (img[:, :, 2] > white_thresh)
            img[mask, 3] = 0 
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None