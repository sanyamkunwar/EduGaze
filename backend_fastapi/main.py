from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from emotions import predict_emotion, get_focus_metrics
from scoring import compute_engagement
from logger import write_log

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    # ------------------------------
    # 1. Get face crop (simple center crop)
    # ------------------------------
    h, w = img.shape[:2]
    face = img[int(h*0.2):int(h*0.85), int(w*0.25):int(w*0.75)]

    # ------------------------------
    # 2. Emotion prediction
    # ------------------------------
    emotion, emo_conf = predict_emotion(face)

    # ------------------------------
    # 3. Eye & Head focus (via mediapipe)
    # ------------------------------
    eye_focus, head_orientation = get_focus_metrics(img)

    # ------------------------------
    # 4. Engagement
    # ------------------------------
    score, status = compute_engagement(eye_focus, emotion, head_orientation)

    # ------------------------------
    # 5. Log
    # ------------------------------
    write_log(emotion, eye_focus, head_orientation, score)

    return {
        "emotion": emotion,
        "eye_focus": eye_focus,
        "head_orientation": head_orientation,
        "score": float(score),
        "status": status
    }
