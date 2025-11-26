import streamlit as st
import cv2
import requests
from utils import convert_frame_to_bytes

st.set_page_config(layout="wide", page_title="EduGaze")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Webcam Feed")
    FRAME = st.image([])

with col2:
    st.header("Engagement Summary")
    status_box = st.empty()
    score_box = st.empty()
    emo_box = st.empty()

backend_url = "http://127.0.0.1:8000/analyze"

import time
last_sent = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot access webcam")
        break

    # Show smaller webcam
    small_frame = cv2.resize(frame, (350, 250))
    FRAME.image(small_frame[:, :, ::-1])

    if time.time() - last_sent > 1:
        b = convert_frame_to_bytes(frame)
        resp = requests.post(backend_url, files={"file": ("f.jpg", b, "image/jpeg")}).json()

        emo_box.write(f"### Emotion: **{resp['emotion']}**")
        status_box.write(f"### Status: **{resp['status']}**")
        score_box.metric("Engagement Score", f"{resp['score']:.2f}")

        last_sent = time.time()
