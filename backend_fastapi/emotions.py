import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp

# ------------------------------
# Load emotion model
# ------------------------------
model = load_model("../models/emotion_model.h5")
labels = ["angry", "happy", "neutral", "sad", "surprised"]

# ------------------------------
# Init MediaPipe FaceMesh
# ------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ------------------------------
# Eye focus calculator
# ------------------------------
def calculate_eye_focus(landmarks):
    LEFT = [33, 159, 145, 133]
    RIGHT = [362, 386, 374, 263]

    def eye_ratio(points):
        p1,p2,p3,p4 = points
        vertical = abs(landmarks[p2].y - landmarks[p3].y)
        horizontal = abs(landmarks[p1].x - landmarks[p4].x)
        return vertical / horizontal

    left_ratio = eye_ratio(LEFT)
    right_ratio = eye_ratio(RIGHT)

    ear = (left_ratio + right_ratio) / 2  # EAR = eye aspect ratio

    # Normalize EAR (0 → 1)
    # Typical human EAR = 0.15 (drowsy) to 0.30 (alert)
    normalized = (ear - 0.15) / (0.30 - 0.15)
    normalized = max(0, min(1, normalized))

    return normalized




# ------------------------------
# Head orientation calculator
# ------------------------------
def calculate_head_orientation(landmarks):
    nose = landmarks[1].x
    left_face = landmarks[234].x
    right_face = landmarks[454].x

    center = (left_face + right_face) / 2
    max_range = (right_face - left_face) / 2  # half width of face

    diff = abs(nose - center)

    normalized = 1 - (diff / max_range)
    normalized = max(0, min(1, normalized))

    return normalized


# ------------------------------
# Get eye + head metrics
# ------------------------------
def get_focus_metrics(img):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        eye = calculate_eye_focus(lm)
        head = calculate_head_orientation(lm)
        return eye, head

    return 0.0, 0.0  # no face detected

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
    except:
        return "neutral", 0.0

