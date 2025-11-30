import streamlit as st
import cv2
import requests
import time
import pandas as pd
import numpy as np
import uuid
import base64

from utils import convert_frame_to_bytes, draw_focus_ring

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="EduGaze")

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = "Student View"
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'engagement_history' not in st.session_state:
    st.session_state.engagement_history = []
if 'last_sent' not in st.session_state:
    st.session_state.last_sent = 0

# --- Backend URL ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Sidebar for Navigation ---
with st.sidebar:
    st.title("EduGaze")
    st.session_state.page = st.radio("Navigate", ["Student View", "Teacher Dashboard"])
    st.info(f"Your User ID: {st.session_state.user_id}")

# ======================================================================================
# --- Student View ---
# ======================================================================================
if st.session_state.page == "Student View":
    st.header("Student Engagement Monitor")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        FRAME = st.image([])
        cap = cv2.VideoCapture(0)

    with col2:
        st.subheader("Engagement Summary")
        status_box = st.empty()
        alert_box = st.empty()
        score_box = st.empty()
        
        st.divider()
        
        col2_1, col2_2, col2_3 = st.columns(3)
        emo_box = col2_1.empty()
        blink_box = col2_2.empty()
        yawn_box = col2_3.empty()
        
        st.divider()

        st.subheader("Engagement Trend")
        chart_box = st.empty()

    while cap.isOpened():
        # If we've navigated away, stop the camera loop
        if st.session_state.page != "Student View":
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam")
            break

        # Analyze frame every 1 second
        if time.time() - st.session_state.last_sent > 0.1:
            try:
                b = convert_frame_to_bytes(frame)
                resp = requests.post(
                    f"{BACKEND_URL}/analyze/{st.session_state.user_id}",
                    files={"file": ("f.jpg", b, "image/jpeg")},
                    timeout=5
                ).json()

                # --- Update UI Elements ---
                status = resp.get("status", "N/A")
                score = resp.get("score", 0)
                emotion = resp.get("emotion", "N/A")
                blinks = resp.get("blinks", 0)
                yawns = resp.get("yawns", 0)
                face_bbox = resp.get("face_bbox")

                status_box.write(f"### Status: **{status}**")
                score_box.metric("Engagement Score", f"{score:.2f}")
                emo_box.metric("Emotion", emotion.capitalize())
                blink_box.metric("Blinks", blinks)
                yawn_box.metric("Yawns", yawns)

                # --- Engagement Alert ---
                if status == "Low Engagement":
                    alert_box.error("Low Engagement Detected!", icon="⚠️")
                else:
                    alert_box.empty()

                # --- Update Engagement History & Chart ---
                st.session_state.engagement_history.append({"time": pd.Timestamp.now(), "score": score})
                if len(st.session_state.engagement_history) > 100: # Keep last 100 points
                    st.session_state.engagement_history.pop(0)
                
                history_df = pd.DataFrame(st.session_state.engagement_history).set_index("time")
                chart_box.line_chart(history_df)

                # --- Draw Face Focus Ring ---
                if face_bbox:
                    frame = draw_focus_ring(frame, face_bbox, status)

                st.session_state.last_sent = time.time()

            except requests.exceptions.RequestException as e:
                st.error(f"Backend connection error: {e}")
                time.sleep(2) # Avoid spamming errors

        # Display the frame
        FRAME.image(frame[:, :, ::-1])

    cap.release()

# ======================================================================================
# --- Teacher Dashboard View ---
# ======================================================================================
elif st.session_state.page == "Teacher Dashboard":
    st.header("Teacher Dashboard")
    st.subheader("Live Student Grid")

    placeholder = st.empty()

    while True:
        try:
            resp = requests.get(f"{BACKEND_URL}/dashboard/data", timeout=5).json()
            
            with placeholder.container():
                if not resp:
                    st.info("No student data available yet. Ask students to open the Student View.")
                else:
                    student_ids = list(resp.keys())
                    num_students = len(student_ids)
                    
                    cols = st.columns(4) # Display up to 4 students per row
                    
                    for i in range(num_students):
                        user_id = student_ids[i]
                        data = resp[user_id]
                        col = cols[i % 4]

                        with col:
                            st.subheader(f"Student #{user_id[:4]}")
                            
                            # Decode and display live frame
                            img_data = data.get("live_frame", "").split(",")[1]
                            if img_data:
                                img_bytes = base64.b64decode(img_data)
                                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                                st.image(img[:,:,::-1], width='stretch')

                            score = data.get('score', 0)
                            status = data.get('status', 'N/A')
                            is_flagged = data.get('is_flagged', False)

                            if is_flagged:
                                st.error("🚩 FLAGGED: Low Engagement > 1 min")
                            
                            st.metric("Score", f"{score:.2f}")
                            
                            if not is_flagged: # Don't show status if already flagged
                                if status == "Low Engagement":
                                    st.warning(f"Status: {status}")
                                elif status == "Medium Engagement":
                                    st.info(f"Status: {status}")
                                else:
                                    st.success(f"Status: {status}")

        except requests.exceptions.RequestException as e:
            with placeholder.container():
                st.error(f"Could not connect to backend: {e}")
        
        except Exception as e:
            with placeholder.container():
                st.error(f"An error occurred: {e}")
            break

        time.sleep(2) # Refresh rate for the dashboard