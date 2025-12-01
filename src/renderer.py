import cv2
import numpy as np
import math

def overlay_img(canvas, img, M):
    h_c, w_c = canvas.shape[:2]
    warped_img = cv2.warpAffine(img, M, (w_c, h_c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    try:
        alpha_mask = warped_img[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3): 
            canvas[:, :, c] = (alpha_inv * canvas[:, :, c] + alpha_mask * warped_img[:, :, c])
    except: pass
    return canvas

# [UPDATE] Sekarang menerima point_wrist dan point_middle (tuple x,y) yang sudah stabil
# Parameter w, h dihapus karena point sudah dalam pixel
def overlay_hand_dynamic(canvas, img_asset, point_wrist, point_middle, scale=1.0, mirror=False, rotation_offset=0):
    if img_asset is None or point_wrist is None or point_middle is None: return canvas
    
    p_wrist = np.array(point_wrist, dtype=np.float32)
    p_middle = np.array(point_middle, dtype=np.float32) 
    
    dy = p_middle[1] - p_wrist[1]
    dx = p_middle[0] - p_wrist[0]
    angle = math.atan2(dy, dx)
    
    h_orig, w_orig = img_asset.shape[:2]
    src_pts = np.float32([[w_orig/2, h_orig], [w_orig/2, 0], [0, h_orig]]) 
    hand_len_cam = np.linalg.norm(p_middle - p_wrist) * 3.0 
    
    dst_p1 = p_wrist
    rot_angle_rad = rotation_offset
    cos_rot = math.cos(rot_angle_rad); sin_rot = math.sin(rot_angle_rad)
    
    vx = math.cos(angle) * hand_len_cam * scale
    vy = math.sin(angle) * hand_len_cam * scale
    vx_final = vx * cos_rot - vy * sin_rot
    vy_final = vx * sin_rot + vy * cos_rot
    
    dst_p2 = dst_p1 + np.array([vx_final, vy_final])
    aspect = w_orig / h_orig
    width_px = (hand_len_cam * scale) * aspect
    angle_left = angle - (math.pi / 2) + rotation_offset
    vec_left = np.array([math.cos(angle_left), math.sin(angle_left)]) * (width_px / 2)
    dst_p3 = dst_p1 + vec_left
    
    img_render = img_asset
    if mirror: img_render = cv2.flip(img_asset, 1)
        
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_render, M)

def overlay_single_eye(canvas, img_asset, point1, point2, scale=1.0, y_offset=0.0, x_offset=0.0):
    if img_asset is None: return canvas
    p1 = np.array(point1, dtype=np.float32)
    p2 = np.array(point2, dtype=np.float32)
    center = (p1 + p2) / 2
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    angle = math.atan2(dy, dx)
    dist = np.linalg.norm(p2 - p1)
    target_w = int(dist * scale)
    h_orig, w_orig = img_asset.shape[:2]
    target_h = int(target_w * (h_orig / w_orig))
    if target_w < 1: return canvas
    shift_y_px = target_h * y_offset
    vec_y = np.array([math.sin(angle), -math.cos(angle)]) * shift_y_px
    shift_x_px = target_w * x_offset
    vec_x = np.array([math.cos(angle), math.sin(angle)]) * shift_x_px
    final_center = center + vec_y + vec_x
    src_pts = np.float32([[w_orig/2, h_orig/2], [w_orig/2 + 100, h_orig/2], [w_orig/2, h_orig/2 + 100]])
    scale_ratio = target_w / w_orig
    dst_p1 = final_center
    dst_p2 = dst_p1 + np.array([math.cos(angle), math.sin(angle)]) * (100 * scale_ratio)
    dst_p3 = dst_p1 + np.array([-math.sin(angle), math.cos(angle)]) * (100 * scale_ratio)
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_asset, M)

def overlay_mouth(canvas, img_asset, point1, point2, scale=1.0, y_offset=0.0):
    if img_asset is None: return canvas
    p1 = np.array(point1, dtype=np.float32)
    p2 = np.array(point2, dtype=np.float32)
    center = (p1 + p2) / 2
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    angle = math.atan2(dy, dx)
    dist = np.linalg.norm(p2 - p1)
    target_w = int(dist * scale)
    h_orig, w_orig = img_asset.shape[:2]
    target_h = int(target_w * (h_orig / w_orig))
    if target_w < 1: return canvas
    shift_px = target_h * y_offset
    final_center = center + np.array([math.sin(angle)*shift_px, -math.cos(angle)*shift_px])
    src_pts = np.float32([[w_orig/2, h_orig/2], [w_orig/2 + 100, h_orig/2], [w_orig/2, h_orig/2 + 100]])
    scale_ratio = target_w / w_orig
    dst_p1 = final_center
    dst_p2 = dst_p1 + np.array([math.cos(angle), math.sin(angle)]) * (100 * scale_ratio)
    dst_p3 = dst_p1 + np.array([-math.sin(angle), math.cos(angle)]) * (100 * scale_ratio)
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_asset, M)

def overlay_limb_final(canvas, img_asset, point_a, point_b, length_factor=1.0, thickness_ratio=0.3, mirror=False):
    if img_asset is None: return canvas
    dist = math.dist(point_a, point_b)
    if dist < 5: return canvas
    delta_y = point_b[1] - point_a[1]; delta_x = point_b[0] - point_a[0]
    angle_rad = math.atan2(delta_y, delta_x)
    img_used = cv2.flip(img_asset, 0) if mirror else img_asset
    h_orig, w_orig = img_used.shape[:2]
    target_width = int(dist * length_factor)
    target_height = int(dist * thickness_ratio)
    if target_width < 1: return canvas
    center_y = h_orig * 0.5 
    src_pts = np.float32([[0, center_y], [w_orig, center_y], [0, h_orig]])
    angle_perp = angle_rad + math.pi / 2
    dst_p1 = np.array(point_a, dtype=np.float32)
    dst_p2 = dst_p1 + np.array([math.cos(angle_rad)*target_width, math.sin(angle_rad)*target_width], dtype=np.float32)
    offset = target_height * 0.5
    dst_p3 = dst_p1 + np.array([math.cos(angle_perp)*offset, math.sin(angle_perp)*offset], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_asset, M)

def overlay_leg_vertical(canvas, img_asset, point_a, point_b, width_factor=0.6, length_factor=1.0):
    if img_asset is None: return canvas
    dist = math.dist(point_a, point_b)
    if dist < 5: return canvas
    dy = point_b[1] - point_a[1]; dx = point_b[0] - point_a[0]
    angle = math.atan2(dy, dx)
    h_orig, w_orig = img_asset.shape[:2]
    src_pts = np.float32([[w_orig // 2, 0], [w_orig // 2, h_orig], [0, 0]])
    target_len_px = dist * length_factor
    target_width_px = dist * width_factor
    dst_p1 = np.array(point_a, dtype=np.float32) 
    vec_len = np.array([math.cos(angle), math.sin(angle)]) * target_len_px
    dst_p2 = dst_p1 + vec_len
    angle_left = angle - (math.pi / 2)
    vec_width = np.array([math.cos(angle_left), math.sin(angle_left)]) * (target_width_px / 2)
    dst_p3 = dst_p1 + vec_width
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_asset, M)

def overlay_torso_central(canvas, img_asset, l_shoulder, r_shoulder, l_hip, r_hip, width_scale=1.0, height_scale=1.0, vertical_shift=0.0):
    if img_asset is None: return canvas
    s_mid = np.array([(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2], dtype=np.float32)
    h_mid = np.array([(l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2], dtype=np.float32)
    angle_spine = math.atan2(h_mid[1] - s_mid[1], h_mid[0] - s_mid[0])
    angle_perp = angle_spine - math.pi / 2 
    s_dist = math.dist(l_shoulder, r_shoulder)
    spine_dist = math.dist(s_mid, h_mid)
    target_w = int(s_dist * width_scale)
    target_h = int(spine_dist * height_scale)
    if target_w < 1: return canvas
    h_orig, w_orig = img_asset.shape[:2]
    src_pts = np.float32([[w_orig // 2, 0], [w_orig // 2, h_orig], [0, 0]])
    shift = spine_dist * vertical_shift
    dst_p1 = s_mid - np.array([math.cos(angle_spine)*shift, math.sin(angle_spine)*shift], dtype=np.float32)
    dst_p2 = dst_p1 + np.array([math.cos(angle_spine)*target_h, math.sin(angle_spine)*target_h], dtype=np.float32)
    dst_p3 = dst_p1 + np.array([math.cos(angle_perp)*(target_w/2), math.sin(angle_perp)*(target_w/2)], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_asset, M)

def overlay_face_part(canvas, img_asset, nose_point, l_shoulder, r_shoulder, l_ear, r_ear, scale=1.0, y_offset=0.0, flip_v=False, flip_h=False):
    if img_asset is None: return canvas
    img_used = img_asset.copy()
    if flip_v: img_used = cv2.flip(img_used, 0)
    if flip_h: img_used = cv2.flip(img_used, 1)
    
    neck_point = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    dy = nose_point[1] - neck_point[1]
    dx = nose_point[0] - neck_point[0]
    angle_up = math.atan2(dy, dx)
    angle_right = angle_up + (math.pi / 2)

    s_width = math.dist(l_shoulder, r_shoulder)
    target_w = int(s_width * scale)
    h_orig, w_orig = img_used.shape[:2]
    target_h = int(target_w * (h_orig / w_orig))
    
    if target_w < 1: return canvas
    center_x, center_y = w_orig // 2, h_orig // 2
    src_pts = np.float32([[center_x, center_y], [center_x, 0], [0, center_y]])
    
    shift_px = target_h * y_offset
    anchor = np.array(nose_point, dtype=np.float32) + np.array([math.cos(angle_up)*shift_px, math.sin(angle_up)*shift_px], dtype=np.float32)
    
    dst_p1 = anchor
    dst_p2 = dst_p1 + np.array([math.cos(angle_up)*(target_h/2), math.sin(angle_up)*(target_h/2)], dtype=np.float32)
    dst_p3 = dst_p1 + np.array([math.cos(angle_right + math.pi)*(target_w/2), math.sin(angle_right + math.pi)*(target_w/2)], dtype=np.float32)
    
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_used, M)

def overlay_shoulder_rotatable(canvas, img_asset, shoulder_pt, elbow_pt, width_ref, scale=1.0, flip_h=False, anchor_y=0.0):
    if img_asset is None: return canvas
    img_render = img_asset.copy()
    if flip_h: img_render = cv2.flip(img_asset, 1)

    dy = elbow_pt[1] - shoulder_pt[1]
    dx = elbow_pt[0] - shoulder_pt[0]
    angle = math.atan2(dy, dx) 
    
    h_orig, w_orig = img_render.shape[:2]
    target_w = int(width_ref * scale)
    if target_w < 1: return canvas
    ratio = target_w / w_orig
    
    pivot_y_src = h_orig * anchor_y
    src_pts = np.float32([
        [w_orig/2, pivot_y_src],         
        [w_orig/2, pivot_y_src + 100],   
        [w_orig/2 + 100, pivot_y_src]    
    ])
    
    dst_p1 = np.array(shoulder_pt, dtype=np.float32)
    scale_100 = 100 * ratio
    vec_down = np.array([math.cos(angle), math.sin(angle)]) * scale_100
    dst_p2 = dst_p1 + vec_down
    
    angle_perp = angle - (math.pi / 2)
    vec_perp = np.array([math.cos(angle_perp), math.sin(angle_perp)]) * scale_100
    dst_p3 = dst_p1 + vec_perp
    
    M = cv2.getAffineTransform(src_pts, np.float32([dst_p1, dst_p2, dst_p3]))
    return overlay_img(canvas, img_render, M)