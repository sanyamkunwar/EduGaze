import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
from scipy.spatial import distance as dist

# ------------------------------
# Load emotion model
# ------------------------------
# Note: The model path is now relative to the backend_fastapi directory
model = load_model("../models/emotion_model.h5")
labels = ["angry", "happy", "neutral", "sad", "surprised"]

# ------------------------------
# Init MediaPipe FaceMesh
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# Landmark indices
# ------------------------------
class EAR_Landmark:
    # Based on standard 6-point EAR calculation
    LEFT_EYE = [362, 385, 387, 263, 373, 380] # p1, p2, p3, p4, p5, p6
    RIGHT_EYE = [33, 158, 159, 133, 144, 153] # p1, p2, p3, p4, p5, p6

class MAR_Landmark:
    # Points for vertical and horizontal distance of the mouth
    LIPS_VERTICAL = [13, 14] # Upper and lower lip inner points
    LIPS_HORIZONTAL = [78, 308] # Left and right mouth corners

# ------------------------------
# Analysis Functions
# ------------------------------

def calculate_ear(landmarks, eye_indices):
    # Extract the (x, y)-coordinates of the landmark points
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    
    # Vertical eye distances
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    
    # Horizontal eye distance
    C = dist.euclidean(points[0], points[3])
    
    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(landmarks, vertical_indices, horizontal_indices, img_shape):
    h, w = img_shape
    # De-normalize coordinates to get true distances
    v_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in vertical_indices])
    h_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in horizontal_indices])

    # Vertical mouth distance
    A = dist.euclidean(v_points[0], v_points[1])
    
    # Horizontal mouth distance
    B = dist.euclidean(h_points[0], h_points[1])

    # Handle potential division by zero if horizontal distance is not detected
    if B == 0:
        return 0.0

    # Mouth Aspect Ratio
    mar = A / B
    return mar

def calculate_head_orientation(landmarks, img_shape):
    h, w = img_shape
    nose_2d = (landmarks[1].x * w, landmarks[1].y * h)
    # A simplified 3D model of the head
    face_3d = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    # 2D Image points
    face_2d = np.array([
        nose_2d,
        (landmarks[152].x * w, landmarks[152].y * h), # Chin
        (landmarks[263].x * w, landmarks[263].y * h), # Left eye corner
        (landmarks[33].x * w, landmarks[33].y * h), # Right eye corner
        (landmarks[287].x * w, landmarks[287].y * h), # Left mouth corner
        (landmarks[57].x * w, landmarks[57].y * h) # Right mouth corner
    ], dtype=np.float64)

    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    return angles[0], angles[1], angles[2] # x, y, z

def get_face_bounding_box(landmarks, img_shape):
    h, w = img_shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

# ------------------------------
# Main Analysis Function
# ------------------------------
def analyze_frame(img):
    h, w = img.shape[:2]
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # Bbox
    x1, y1, x2, y2 = get_face_bounding_box(landmarks, (h,w))

    # Emotion
    face_crop = img[y1:y2, x1:x2]
    emotion, emo_conf = predict_emotion(face_crop)
    
    # Eye Metrics (Blink)
    left_ear = calculate_ear(landmarks, EAR_Landmark.LEFT_EYE)
    right_ear = calculate_ear(landmarks, EAR_Landmark.RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    
    # Yawn Metric
    mar = calculate_mar(landmarks, MAR_Landmark.LIPS_VERTICAL, MAR_Landmark.LIPS_HORIZONTAL, (h, w))
    
    # Head Pose
    pitch, yaw, roll = calculate_head_orientation(landmarks, (h,w))

    # Eye Gaze (simplified)
    eye_focus = 1 - (abs(yaw) / 25) # Normalize based on typical yaw range
    eye_focus = max(0, min(1, eye_focus))

    return {
        "emotion": emotion,
        "emo_conf": emo_conf,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "eye_focus": eye_focus, # Simplified gaze
        "head_orientation_x": pitch,
        "head_orientation_y": yaw,
        "head_orientation_z": roll,
        "face_bbox": [x1, y1, x2, y2]
    }

# ------------------------------
# Emotion prediction
# ------------------------------
def predict_emotion(face_img):
    try:
        img = cv2.resize(face_img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]
        emotion = labels[np.argmax(pred)]
        return emotion, float(max(pred))
    except Exception as e:
        # print(f"Emotion prediction failed: {e}")
        return "neutral", 0.0