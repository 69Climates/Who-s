import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time
from scipy.spatial import distance as dist

st.set_page_config(
    page_title="FairFace AI",       # This sets the tab title
    page_icon="🧠"
)



st.info("📝 Note: This comparison is based only on the image, not real-life appearance.")
st.warning("⚠️ DISCLAIMER: For image comparison only. Misuse is not allowed.")

# Cache the MediaPipe Face Mesh to avoid reloading
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

@st.cache_resource
def load_haar_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

# Set page title and description
st.title("Know who's more Beautiful")
st.title("Compare 2 Faces")
st.write("Upload two face images to see which scores higher.")

# Preload the detectors
face_mesh = load_face_mesh()
face_cascade, eye_cascade = load_haar_cascades()

def calculate_face_shape(landmarks, image_shape):
    h, w = image_shape[:2]
    try:
        if not landmarks or len(landmarks) < 468:
            return 0
        
        # Jawline: approximate using MediaPipe landmarks
        jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67]
        jaw = [landmarks[i] for i in jaw_indices]

        # Forehead width: distance between temples
        forehead_width = np.linalg.norm(np.array(landmarks[103]) - np.array(landmarks[332]))
        # Face height: chin to forehead top
        face_height = np.linalg.norm(np.array(landmarks[152]) - np.array(landmarks[10]))
        # Cheek width: distance between cheek landmarks
        cheek_width = np.linalg.norm(np.array(landmarks[234]) - np.array(landmarks[454]))

        # Estimate forehead height using landmarks above eyebrows
        forehead_height = (landmarks[151][1] + landmarks[9][1]) / 2 - landmarks[1][1]
        total_face_height = face_height + forehead_height

        # Ratios for classification
        if forehead_width == 0:
            return 0
        
        aspect_ratio = total_face_height / forehead_width
        cheek_to_jaw_ratio = cheek_width / forehead_width

        # Classification based on geometric ratios
        if aspect_ratio < 1.3 and cheek_to_jaw_ratio < 0.9:
            return 8
        elif aspect_ratio >= 1.3 and cheek_to_jaw_ratio < 0.9:
            return 18
        elif cheek_to_jaw_ratio >= 1.0:
            return 11
        elif cheek_to_jaw_ratio < 0.8 and aspect_ratio > 1.5:
            return 15
        elif landmarks[234][0] - landmarks[454][0] > cheek_width / 2 and landmarks[152][1] > landmarks[234][1]:
            return 26
        elif cheek_width < forehead_width and aspect_ratio > 1.4:
            return 0
        else:
            return 22
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return 0

def detect_face_shape(image_path, face_mesh=face_mesh):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                face_shape = calculate_face_shape(landmarks, image.shape)
                return face_shape * 100 / 26
        return 0
    except Exception as e:
        st.error(f"Error in face shape detection: {e}")
        return 0

def get_average_skin_color(image_path, face_mesh=face_mesh):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                # Approximate face bounding box using landmarks
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                roi = image[y_min:y_max, x_min:x_max]
                step = max(1, min(roi.shape[0], roi.shape[1]) // 20)
                skin_pixels = [roi[i, j] for i in range(0, roi.shape[0], step) for j in range(0, roi.shape[1], step)]
                avg_bgr = np.mean(skin_pixels, axis=0)
                S = np.sqrt(sum(avg_bgr ** 2))
                return S
        return 0
    except Exception as e:
        st.error(f"Error in skin color detection: {e}")
        return 0

def calculate_facial_symmetry(landmarks, h, w):
    """Advanced bilateral facial symmetry analysis"""
    try:
        if not landmarks or len(landmarks) < 468:
            return 50
        
        # Key facial points for symmetry comparison
        left_points = [33, 133, 362, 263, 61, 291, 50, 280]
        right_points = [362, 263, 33, 133, 291, 61, 280, 50]
        
        face_center_x = landmarks[1][0]  # Nose tip x-coordinate
        
        symmetry_scores = []
        for left_idx, right_idx in zip(left_points, right_points):
            if left_idx < len(landmarks) and right_idx < len(landmarks):
                left_dist = abs(landmarks[left_idx][0] - face_center_x)
                right_dist = abs(landmarks[right_idx][0] - face_center_x)
                score = 100 - min(abs(left_dist - right_dist) * 2, 100)
                symmetry_scores.append(score)
        
        return np.mean(symmetry_scores) if symmetry_scores else 50
    except (IndexError, TypeError, ValueError) as e:
        return 50

def calculate_golden_ratio_score(landmarks, h, w):
    """Calculate facial proportions based on golden ratio (1.618)"""
    try:
        if not landmarks or len(landmarks) < 468:
            return 50
        
        golden_ratio = 1.618
        scores = []
        
        # Face length to width ratio
        face_length = dist.euclidean(landmarks[10], landmarks[152])
        face_width = dist.euclidean(landmarks[234], landmarks[454])
        if face_width > 0:
            length_width_ratio = face_length / face_width
            scores.append(max(0, 100 - min(abs(length_width_ratio - golden_ratio) * 30, 100)))
        
        # Eye to mouth distance vs nose to chin distance
        eye_center = ((landmarks[33][0] + landmarks[263][0]) / 2, (landmarks[33][1] + landmarks[263][1]) / 2)
        mouth_center = landmarks[13]
        nose_tip = landmarks[1]
        chin = landmarks[152]
        
        eye_mouth = np.linalg.norm(np.array(eye_center) - np.array(mouth_center))
        nose_chin = np.linalg.norm(np.array(nose_tip) - np.array(chin))
        
        if nose_chin > 0:
            proportion = eye_mouth / nose_chin
            scores.append(max(0, 100 - min(abs(proportion - golden_ratio) * 40, 100)))
        
        # Nose width to mouth width ratio
        nose_width = dist.euclidean(landmarks[129], landmarks[358])
        mouth_width = dist.euclidean(landmarks[61], landmarks[291])
        if mouth_width > 0 and nose_width > 0:
            nose_mouth_ratio = mouth_width / nose_width
            scores.append(max(0, 100 - min(abs(nose_mouth_ratio - golden_ratio) * 30, 100)))
        
        return np.mean(scores) if scores else 50
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        return 50

def analyze_skin_texture(image_path):
    """Enhanced skin texture and quality analysis"""
    try:
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            return 50
        
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            return 50
        
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Texture smoothness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        smoothness_score = max(0, min(100, 100 - min(laplacian_var / 10, 100)))
        
        # Skin uniformity using standard deviation
        roi = gray[int(gray.shape[0]*0.2):int(gray.shape[0]*0.8), 
                   int(gray.shape[1]*0.2):int(gray.shape[1]*0.8)]
        
        if roi.size == 0:
            return 50
        
        uniformity_score = max(0, min(100, 100 - min(np.std(roi) / 2, 100)))
        
        return max(0, min(100, smoothness_score * 0.6 + uniformity_score * 0.4))
    except (cv2.error, ValueError, TypeError) as e:
        return 50

def analyze_nose_shape(landmarks, h, w):
    """Analyze nose proportions and shape"""
    try:
        if not landmarks or len(landmarks) < 468:
            return 50
        
        nose_bridge_top = landmarks[6]
        nose_bridge_bottom = landmarks[4]
        nose_left = landmarks[129]
        nose_right = landmarks[358]
        nose_tip = landmarks[1]
        
        bridge_length = dist.euclidean(nose_bridge_top, nose_bridge_bottom)
        nose_width = dist.euclidean(nose_left, nose_right)
        
        # Ideal nose width to bridge ratio
        if bridge_length > 0:
            width_ratio = nose_width / bridge_length
            ratio_score = max(0, min(100, 100 - min(abs(width_ratio - 0.7) * 100, 100)))
        else:
            ratio_score = 50
        
        # Nose symmetry
        nose_center_x = nose_tip[0]
        left_dist = abs(nose_left[0] - nose_center_x)
        right_dist = abs(nose_right[0] - nose_center_x)
        symmetry_score = max(0, min(100, 100 - min(abs(left_dist - right_dist) * 5, 100)))
        
        return max(0, min(100, ratio_score * 0.5 + symmetry_score * 0.5))
    except (IndexError, TypeError, ValueError) as e:
        return 50

def analyze_lips(landmarks, h, w):
    """Analyze lip fullness and symmetry"""
    try:
        if not landmarks or len(landmarks) < 468:
            return 50
        
        upper_lip_top = landmarks[13]
        upper_lip_bottom = landmarks[14]
        lower_lip_top = landmarks[14]
        lower_lip_bottom = landmarks[17]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        
        # Lip fullness
        upper_thickness = dist.euclidean(upper_lip_top, upper_lip_bottom)
        lower_thickness = dist.euclidean(lower_lip_top, lower_lip_bottom)
        mouth_width = dist.euclidean(mouth_left, mouth_right)
        
        if mouth_width > 0:
            fullness_ratio = (upper_thickness + lower_thickness) / mouth_width
            fullness_score = max(0, min(100, fullness_ratio * 500))
        else:
            fullness_score = 50
        
        # Lip symmetry
        mouth_center_x = landmarks[13][0]
        left_dist = abs(mouth_left[0] - mouth_center_x)
        right_dist = abs(mouth_right[0] - mouth_center_x)
        symmetry_score = max(0, min(100, 100 - min(abs(left_dist - right_dist) * 3, 100)))
        
        # Upper to lower lip ratio (ideal ~1:1.6)
        if lower_thickness > 0:
            lip_ratio = upper_thickness / lower_thickness
            ratio_score = max(0, min(100, 100 - min(abs(lip_ratio - 0.625) * 100, 100)))
        else:
            ratio_score = 50
        
        return max(0, min(100, fullness_score * 0.4 + symmetry_score * 0.3 + ratio_score * 0.3))
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        return 50

def analyze_cheekbones(landmarks, h, w):
    """Analyze cheekbone prominence and positioning"""
    try:
        if not landmarks or len(landmarks) < 468:
            return 50
        
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        nose_bridge = landmarks[6]
        chin = landmarks[152]
        
        # Cheekbone width
        cheek_width = dist.euclidean(left_cheek, right_cheek)
        face_length = dist.euclidean(nose_bridge, chin)
        
        # Ideal cheekbone to face length ratio
        if face_length > 0:
            prominence_score = max(0, min(100, (cheek_width / face_length) * 80))
        else:
            prominence_score = 50
        
        # Cheekbone height positioning
        eye_level = (landmarks[33][1] + landmarks[263][1]) / 2
        cheek_level = (left_cheek[1] + right_cheek[1]) / 2
        
        ideal_position = nose_bridge[1] + (chin[1] - nose_bridge[1]) * 0.35
        position_score = max(0, min(100, 100 - min(abs(cheek_level - ideal_position) * 0.5, 100)))
        
        return max(0, min(100, prominence_score * 0.6 + position_score * 0.4))
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        return 50

def rate_jawline(image_path, face_mesh=face_mesh):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = img.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                # Enhanced jawline indices
                jaw_indices = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
                jawline = np.array([landmarks[i] for i in jaw_indices])

                mid_idx = len(jawline) // 2
                left_jawline = jawline[:mid_idx]
                right_jawline = jawline[mid_idx+1:]
                center_point = jawline[mid_idx]
                
                min_len = min(len(left_jawline), len(right_jawline))
                left_distances = [np.linalg.norm(pt - center_point) for pt in left_jawline[:min_len]]
                right_distances = [np.linalg.norm(pt - center_point) for pt in right_jawline[:min_len][::-1]]
                symmetry_score = 100 - np.mean(np.abs(np.array(left_distances) - np.array(right_distances))) * 2
                symmetry_score = max(0, min(100, symmetry_score))

                def calculate_angle(a, b, c):
                    ba = a - b
                    bc = c - b
                    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
                    if norm_product == 0:
                        return 90
                    cosine_angle = np.dot(ba, bc) / norm_product
                    cosine_angle = min(1.0, max(-1.0, cosine_angle))
                    return np.degrees(np.arccos(cosine_angle))

                angles = [calculate_angle(jawline[i-1], jawline[i], jawline[i+1]) for i in range(1, len(jawline)-1)]
                # Ideal jawline angle around 115-125 degrees for defined jaw
                sharpness_score = 100 - np.mean(np.abs(np.array(angles) - 120)) * 0.8
                sharpness_score = max(0, min(100, sharpness_score))
                
                # Jawline definition (distance from ear to chin)
                jaw_definition = dist.euclidean(jawline[0], jawline[-1])
                face_width = dist.euclidean(landmarks[234], landmarks[454])
                if face_width > 0:
                    definition_score = min((jaw_definition / face_width) * 60, 100)
                else:
                    definition_score = 50

                jawline_rating = 0.4 * symmetry_score + 0.35 * sharpness_score + 0.25 * definition_score
                return jawline_rating
        return 0
    except Exception as e:
        st.error(f"Error in jawline detection: {e}")
        return 0

def calculate_eye_shape(eye_landmarks):
    width = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    height = (np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])) +
              np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))) / 2
    aspect_ratio = height / width
    if aspect_ratio < 0.25:
        return 38
    elif 0.25 <= aspect_ratio <= 0.35:
        return 28
    elif aspect_ratio > 0.35:
        return 22
    else:
        return 12

def detect_eyes_shape(image_path, face_mesh=face_mesh):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = img.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                # Enhanced eye landmarks
                left_eye_indices = [33, 246, 161, 160, 159, 158, 133, 173, 157, 154, 153, 145, 144, 163, 7]
                right_eye_indices = [362, 398, 384, 385, 386, 387, 263, 466, 388, 387, 386, 385, 384, 398, 362]
                left_eye = [landmarks[i] for i in left_eye_indices[:6]]
                right_eye = [landmarks[i] for i in right_eye_indices[:6]]
                
                left_eye_shape = calculate_eye_shape(left_eye)
                right_eye_shape = calculate_eye_shape(right_eye)
                
                # Eye spacing analysis
                left_corner = landmarks[33]
                right_corner = landmarks[263]
                eye_distance = dist.euclidean(left_corner, right_corner)
                nose_width = dist.euclidean(landmarks[129], landmarks[358])
                
                # Ideal eye spacing is about 1 eye width apart
                if nose_width > 0:
                    spacing_ratio = eye_distance / nose_width
                    spacing_score = 100 - min(abs(spacing_ratio - 3.5) * 20, 100)
                else:
                    spacing_score = 50
                
                shape_score = ((left_eye_shape + right_eye_shape) * 100 / 72)
                return (shape_score * 0.7 + spacing_score * 0.3)
        return 0
    except Exception as e:
        st.error(f"Error in eye shape detection: {e}")
        return 0

def classify_eye_color(rgb_values):
    r, g, b = rgb_values
    if r > 100 and g < 70 and b < 40:
        return 5
    elif r > 140 and g > 100 and b < 60:
        return 19
    elif r < 100 and g < 100 and b > 120:
        return 29
    elif r < 100 and g > 120 and b < 100:
        return 14
    elif r > 100 and g > 80 and b < 60:
        return 9
    elif r < 100 and g < 100 and b < 80:
        return 24
    else:
        return 15

def get_eye_rgb_value(eye_image):
    return np.mean(eye_image, axis=(0, 1))

def detect_eye_colors(image_path, face_cascade=face_cascade, eye_cascade=eye_cascade):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return 0
        (x, y, w, h) = faces[0]
        face_region = image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        if len(eyes) < 2:
            return 0
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        left_eye = face_region[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
        right_eye = face_region[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]]
        left_eye_rgb = get_eye_rgb_value(left_eye)
        right_eye_rgb = get_eye_rgb_value(right_eye)
        left_eye_color = classify_eye_color(left_eye_rgb)
        right_eye_color = classify_eye_color(right_eye_rgb)
        S = (left_eye_color + right_eye_color) / 2
        return S * 100 / 30
    except Exception as e:
        st.error(f"Error in eye color detection: {e}")
        return 0

def calculate_hair_color_score(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0, 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hair_region = image_rgb[:int(image.shape[0] * 0.4), :]
        step = max(1, min(hair_region.shape[0], hair_region.shape[1]) // 30)
        samples = [hair_region[i, j] for i in range(0, hair_region.shape[0], step) for j in range(0, hair_region.shape[1], step)]
        avg_rgb = np.mean(samples, axis=0)
        return avg_rgb[0], avg_rgb[1], avg_rgb[2]
    except Exception as e:
        st.error(f"Error in hair color calculation: {e}")
        return 0, 0, 0

def calculate_hair_density_and_baldness(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return 0, 0
        
        max_dimension = 500
        height, width = image.shape
        if height < 10 or width < 10:
            return 0, 0
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            image = cv2.resize(image, (new_width, new_height))
        
        edges = cv2.Canny(image, 100, 200)
        mask = np.zeros_like(edges)
        mask[:max(1, int(image.shape[0] * 0.4)), :] = 255
        total_pixels = np.sum(mask > 0)
        
        if total_pixels == 0:
            return 0, 0
        
        hair_pixels = np.sum(cv2.bitwise_and(edges, mask) > 0)
        S1 = min(100, (hair_pixels / total_pixels) * 100)
        scalp_pixels = np.sum(cv2.bitwise_and(edges, mask) == 0)
        S2 = min(100, (scalp_pixels / total_pixels) * 100)
        return S1, S2
    except (cv2.error, ValueError, TypeError) as e:
        return 0, 0

def calculate_final_score(image_path):
    try:
        a, b, c = calculate_hair_color_score(image_path)
        S1, S2 = calculate_hair_density_and_baldness(image_path)
        
        if a == 0 and b == 0 and c == 0:
            return 0
        
        max_distance = 256 * 1.74
        current_distance = (a**2 + b**2 + c**2)**0.5
        color_score = max(0, min(100, (max_distance - current_distance) * 100 / max_distance))
        
        S = (color_score + S1 + S2) / 3
        return max(0, min(100, S))
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return 0

def mark_winner(image_path, is_winner=True):
    try:
        image = Image.open(image_path)
        if is_winner:
            draw = ImageDraw.Draw(image)
            text = "Hott ONE"
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except IOError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            image_width, image_height = image.size
            position_x = (image_width - text_width) // 2
            position_y = image_height - text_height - 20
            rectangle_position = (position_x - 10, position_y - 5, position_x + text_width + 10, position_y + text_height + 5)
            draw.rectangle(rectangle_position, fill="white")
            draw.text((position_x, position_y), text, fill="black", font=font)
        return image
    except Exception as e:
        st.error(f"Error marking winner: {e}")
        return Image.open(image_path)

def analyze_image(image_path, progress_callback=None):
    try:
        metrics = {}
        if progress_callback:
            progress_callback(0, "Detecting face shape...")
        metrics['face_shape'] = detect_face_shape(image_path)
        if metrics['face_shape'] == 0:
            metrics.update({
                'skin_color': 0, 'skin_score': 0, 'skin_texture': 0, 'jawline': 0, 
                'eye_shape': 0, 'eye_color': 0, 'hair_score': 0, 'facial_symmetry': 0,
                'golden_ratio': 0, 'nose_score': 0, 'lip_score': 0, 'cheekbone_score': 0,
                'final_score': 0
            })
            if progress_callback:
                progress_callback(100, "No face detected!")
            return metrics

        if progress_callback:
            progress_callback(8, "Analyzing facial symmetry...")
        # Get landmarks for advanced analysis
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                metrics['facial_symmetry'] = calculate_facial_symmetry(landmarks, h, w)
                
                if progress_callback:
                    progress_callback(15, "Analyzing golden ratio proportions...")
                metrics['golden_ratio'] = calculate_golden_ratio_score(landmarks, h, w)
                
                if progress_callback:
                    progress_callback(22, "Analyzing nose shape...")
                metrics['nose_score'] = analyze_nose_shape(landmarks, h, w)
                
                if progress_callback:
                    progress_callback(28, "Analyzing lips...")
                metrics['lip_score'] = analyze_lips(landmarks, h, w)
                
                if progress_callback:
                    progress_callback(35, "Analyzing cheekbones...")
                metrics['cheekbone_score'] = analyze_cheekbones(landmarks, h, w)
        else:
            metrics['facial_symmetry'] = 50
            metrics['golden_ratio'] = 50
            metrics['nose_score'] = 50
            metrics['lip_score'] = 50
            metrics['cheekbone_score'] = 50

        if progress_callback:
            progress_callback(42, "Analyzing skin quality...")
        metrics['skin_color'] = get_average_skin_color(image_path)
        metrics['skin_score'] = (1.5 * 100 * metrics['skin_color'] / (256 * (3 ** 0.5)))
        metrics['skin_texture'] = analyze_skin_texture(image_path)

        if progress_callback:
            progress_callback(52, "Analyzing jawline...")
        metrics['jawline'] = rate_jawline(image_path)

        if progress_callback:
            progress_callback(62, "Analyzing eye shape...")
        metrics['eye_shape'] = detect_eyes_shape(image_path)

        if progress_callback:
            progress_callback(72, "Analyzing eye color...")
        metrics['eye_color'] = detect_eye_colors(image_path)

        if progress_callback:
            progress_callback(85, "Analyzing hair...")
        metrics['hair_score'] = calculate_final_score(image_path)

        # Enhanced weighted scoring system
        metrics['final_score'] = (
            metrics['facial_symmetry'] * 15 +      # Symmetry is crucial
            metrics['golden_ratio'] * 12 +         # Proportions matter
            metrics['face_shape'] * 10 +           # Face shape
            metrics['skin_score'] * 12 +           # Skin tone
            metrics['skin_texture'] * 10 +         # Skin quality
            metrics['jawline'] * 10 +              # Jawline definition
            metrics['cheekbone_score'] * 8 +       # Cheekbone structure
            metrics['eye_shape'] * 7 +             # Eye shape
            metrics['eye_color'] * 4 +             # Eye color
            metrics['nose_score'] * 6 +            # Nose proportions
            metrics['lip_score'] * 6 +             # Lip fullness/symmetry
            metrics['hair_score'] * 10             # Hair quality
        ) / 110 * 100  # Normalize to 100
        
        if progress_callback:
            progress_callback(100, "Analysis complete!")
        return metrics
    except Exception as e:
        st.error(f"Error in image analysis: {e}")
        return {k: 0 for k in ['face_shape', 'skin_color', 'skin_score', 'skin_texture', 'jawline', 
                                'eye_shape', 'eye_color', 'hair_score', 'facial_symmetry', 'golden_ratio',
                                'nose_score', 'lip_score', 'cheekbone_score', 'final_score']}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Face 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")
with col2:
    st.subheader("Face 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    st.write("Processing images... This may take a moment.")
    temp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file1.write(uploaded_file1.getvalue())
    temp_file1.close()
    temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file2.write(uploaded_file2.getvalue())
    temp_file2.close()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        def update_progress(percent, message=""):
            progress_bar.progress(percent / 100)
            if message:
                status_text.text(message)

        status_text.text("Analyzing first image...")
        metrics1 = analyze_image(temp_file1.name, lambda percent, msg: update_progress(percent / 2, f"Image 1: {msg}"))
        status_text.text("Analyzing second image...")
        metrics2 = analyze_image(temp_file2.name, lambda percent, msg: update_progress(50 + percent / 2, f"Image 2: {msg}"))
        status_text.text("Comparing results...")

        s1, s2 = metrics1['final_score'], metrics2['final_score']
        winner_image = temp_file1.name if s1 >= s2 else temp_file2.name
        winner_pil = mark_winner(winner_image, True)

        status_text.empty()
        st.subheader("Results")
        def clamp_score(score):
            """Ensure score is between 0 and 100"""
            return max(0, min(100, score))
        
        col3, col4 = st.columns(2)
        with col3:
            st.image(uploaded_file1, caption="Image 1")
            st.metric("Overall Beauty Score", f"{s1:.2f}/100", delta=f"{s1-s2:.2f}" if s1 >= s2 else f"{s1-s2:.2f}")
            with st.expander("📊 Detailed Analysis - Image 1"):
                st.markdown("**🔍 Facial Structure**")
                st.progress(clamp_score(metrics1.get('facial_symmetry', 0)) / 100)
                st.write(f"Facial Symmetry: {clamp_score(metrics1.get('facial_symmetry', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('golden_ratio', 0)) / 100)
                st.write(f"Golden Ratio Proportions: {clamp_score(metrics1.get('golden_ratio', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('face_shape', 0)) / 100)
                st.write(f"Face Shape: {clamp_score(metrics1.get('face_shape', 0)):.1f}/100")
                
                st.markdown("**✨ Skin Quality**")
                st.progress(clamp_score(metrics1.get('skin_score', 0)) / 100)
                st.write(f"Skin Tone: {clamp_score(metrics1.get('skin_score', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('skin_texture', 0)) / 100)
                st.write(f"Skin Texture: {clamp_score(metrics1.get('skin_texture', 0)):.1f}/100")
                
                st.markdown("**👤 Facial Features**")
                st.progress(clamp_score(metrics1.get('jawline', 0)) / 100)
                st.write(f"Jawline Definition: {clamp_score(metrics1.get('jawline', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('cheekbone_score', 0)) / 100)
                st.write(f"Cheekbone Structure: {clamp_score(metrics1.get('cheekbone_score', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('nose_score', 0)) / 100)
                st.write(f"Nose Proportions: {clamp_score(metrics1.get('nose_score', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('lip_score', 0)) / 100)
                st.write(f"Lip Aesthetics: {clamp_score(metrics1.get('lip_score', 0)):.1f}/100")
                
                st.markdown("**👁️ Eyes & Hair**")
                st.progress(clamp_score(metrics1.get('eye_shape', 0)) / 100)
                st.write(f"Eye Shape & Spacing: {clamp_score(metrics1.get('eye_shape', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('eye_color', 0)) / 100)
                st.write(f"Eye Color: {clamp_score(metrics1.get('eye_color', 0)):.1f}/100")
                st.progress(clamp_score(metrics1.get('hair_score', 0)) / 100)
                st.write(f"Hair Quality: {clamp_score(metrics1.get('hair_score', 0)):.1f}/100")
        
        with col4:
            st.image(uploaded_file2, caption="Image 2")
            st.metric("Overall Beauty Score", f"{s2:.2f}/100", delta=f"{s2-s1:.2f}" if s2 >= s1 else f"{s2-s1:.2f}")
            with st.expander("📊 Detailed Analysis - Image 2"):
                st.markdown("**🔍 Facial Structure**")
                st.progress(clamp_score(metrics2.get('facial_symmetry', 0)) / 100)
                st.write(f"Facial Symmetry: {clamp_score(metrics2.get('facial_symmetry', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('golden_ratio', 0)) / 100)
                st.write(f"Golden Ratio Proportions: {clamp_score(metrics2.get('golden_ratio', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('face_shape', 0)) / 100)
                st.write(f"Face Shape: {clamp_score(metrics2.get('face_shape', 0)):.1f}/100")
                
                st.markdown("**✨ Skin Quality**")
                st.progress(clamp_score(metrics2.get('skin_score', 0)) / 100)
                st.write(f"Skin Tone: {clamp_score(metrics2.get('skin_score', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('skin_texture', 0)) / 100)
                st.write(f"Skin Texture: {clamp_score(metrics2.get('skin_texture', 0)):.1f}/100")
                
                st.markdown("**👤 Facial Features**")
                st.progress(clamp_score(metrics2.get('jawline', 0)) / 100)
                st.write(f"Jawline Definition: {clamp_score(metrics2.get('jawline', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('cheekbone_score', 0)) / 100)
                st.write(f"Cheekbone Structure: {clamp_score(metrics2.get('cheekbone_score', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('nose_score', 0)) / 100)
                st.write(f"Nose Proportions: {clamp_score(metrics2.get('nose_score', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('lip_score', 0)) / 100)
                st.write(f"Lip Aesthetics: {clamp_score(metrics2.get('lip_score', 0)):.1f}/100")
                
                st.markdown("**👁️ Eyes & Hair**")
                st.progress(clamp_score(metrics2.get('eye_shape', 0)) / 100)
                st.write(f"Eye Shape & Spacing: {clamp_score(metrics2.get('eye_shape', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('eye_color', 0)) / 100)
                st.write(f"Eye Color: {clamp_score(metrics2.get('eye_color', 0)):.1f}/100")
                st.progress(clamp_score(metrics2.get('hair_score', 0)) / 100)
                st.write(f"Hair Quality: {clamp_score(metrics2.get('hair_score', 0)):.1f}/100")
        st.subheader("Hott One ⚡")
        st.image(winner_pil, caption=f"Face {'1' if s1 >= s2 else '2'} Wins!")
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        try:
            os.unlink(temp_file1.name)
            os.unlink(temp_file2.name)
        except:
            pass
else:
    st.info("Please upload both images to compare.")
