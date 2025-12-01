import os

ASSETS_DIR = 'assets'

def get_path(filename):
    return os.path.join(ASSETS_DIR, filename)

# Nama File
FILE_LENGAN_ATAS   = get_path('lengan_atas.png')
FILE_LENGAN_BAWAH  = get_path('lengan_bawah.png')
FILE_TORSO         = get_path('torso1.png')
FILE_HEAD          = get_path('head.png')
FILE_BAHU_KIRI     = get_path('bahu_kiri.png')   
FILE_BAHU_KANAN    = get_path('bahu_kanan.png')  

FILE_MATA_BUKA_KI  = get_path('mata_buka_kiri.png')
FILE_MATA_BUKA_KA  = get_path('mata_buka_kanan.png')
FILE_MATA_TUTUP_KI = get_path('mata_tutup_kiri.png')
FILE_MATA_TUTUP_KA = get_path('mata_tutup_kanan.png')

FILE_MULUT_BUKA    = get_path('mulut_buka.png') 
FILE_GIGI          = get_path('gigi.png') 

FILE_CELANA_ATAS   = get_path('celana_atas1.png')
FILE_CELANA_BAWAH  = get_path('celana_bawah1.png')

FILE_BG            = get_path('bg.jpg')

FILE_HAND_OPEN     = get_path('open_palm1.jpg')
FILE_HAND_RELAX    = get_path('relax_palm.jpg')


# PARAMETERS
# ==============================================================================

# -- BODY --
SET_PANJANG_LENGAN = 1.30  
SET_LEBAR_TORSO    = 0.85  
SET_TINGGI_TORSO   = 1.30  
SET_TORSO_SHIFT_Y  = 0.25  # Posisi vertikal torso

# -- BAHU --
SET_SCALE_BAHU     = 0.5  
SET_PIVOT_BAHU_Y   = 0.50 
SET_OFFSET_BAHU_X  = 0    
SET_OFFSET_BAHU_Y  = 0    

# -- CELANA --
SET_OFFSET_PINGGUL_NAIK = 0.6
SET_OFFSET_LUTUT_NAIK   = 0.4
SET_SCALE_CELANA_ATAS_W = 1.3
SET_SCALE_CELANA_ATAS_H = 1.3 
SET_SCALE_CELANA_BAWAH_W = 1.3
SET_SCALE_CELANA_BAWAH_H = 1.3
SET_THICKNESS_PAHA      = 0.35
SET_THICKNESS_BETIS     = 0.35

# -- HEAD & FACE --
SET_SCALE_KEPALA      = 0.65
SET_OFFSET_HIDUNG     = 0.3 

SET_SCALE_MATA_BUKA   = 3.0
SET_SCALE_MATA_TUTUP  = 3.0 
SET_OFFSET_MATA       = 0.8
SET_OFFSET_MATA_X_KANAN = 0.15 
SET_OFFSET_MATA_X_KIRI  = 0.25 

SET_SCALE_MULUT       = 2.0 
SET_OFFSET_MULUT      = 1.15
SET_SCALE_GIGI        = 1.20 
SET_OFFSET_GIGI       = 1.15 

# -- TANGAN --
SET_SCALE_TANGAN = 0.9 
SET_ROTASI_TANGAN_OFFSET = 0.0