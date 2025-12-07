import streamlit as st
import cv2
import requests
import time
import pandas as pd
import numpy as np
import uuid
import base64
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils import convert_frame_to_bytes, draw_focus_ring

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="EduGaze")

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = "Student View"
if 'engagement_history' not in st.session_state:
    st.session_state.engagement_history = []

# --- Backend URL ---
BACKEND_URL = "https://edugaze-backend.onrender.com"

# --- Thread-safe data sharing ---
lock = threading.Lock()
latest_analysis_data = {}
last_error = None

# --- WebRTC Video Processor ---
class EduGazeVideoProcessor(VideoProcessorBase):
    def __init__(self, user_id: str):
        self.last_sent = 0
        self.user_id = user_id

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global latest_analysis_data, last_error
        img = frame.to_ndarray(format="bgr24")

        # Analyze frame every ~1 second
        if time.time() - self.last_sent > 1.0:
            self.last_sent = time.time()
            try:
                b = convert_frame_to_bytes(img)
                resp = requests.post(
                    f"{BACKEND_URL}/analyze/{self.user_id}",
                    files={"file": ("f.jpg", b, "image/jpeg")},
                    timeout=20  # Increased timeout
                ).json()

                with lock:
                    latest_analysis_data = resp
                    last_error = None # Clear last error on success

            except requests.exceptions.RequestException as e:
                with lock:
                    last_error = "Error: Backend connection failed."
                print(f"Backend connection error: {e}")
            except Exception as e:
                with lock:
                    last_error = "Error: Analysis failed."
                print(f"An error occurred in video processor: {e}")

        # Draw feedback on the frame
        with lock:
            face_bbox = latest_analysis_data.get("face_bbox")
            status = latest_analysis_data.get("status")
            current_error = last_error
        
        if face_bbox:
            img = draw_focus_ring(img, face_bbox, status)
        
        if current_error:
            cv2.putText(img, current_error, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Factory function to create the processor ---
def processor_factory():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    return EduGazeVideoProcessor(user_id=st.session_state.user_id)

# --- Sidebar for Navigation ---
with st.sidebar:
    st.title("EduGaze")
    st.session_state.page = st.radio("Navigate", ["Student View", "Teacher Dashboard"])
    if 'user_id' in st.session_state:
        st.info(f"Your User ID: {st.session_state.user_id}")

# ======================================================================================
# --- Student View ---
# ======================================================================================
if st.session_state.page == "Student View":
    st.header("Student Engagement Monitor")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        st.write("Click 'Start' to begin the session. Your browser will ask for camera permission.")
        webrtc_streamer(
            key="student-camera",
            video_processor_factory=processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

    with col2:
        st.subheader("Engagement Summary")
        
        # Add a placeholder for the debug JSON
        debug_placeholder = st.empty()
        
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

    # UI update loop
    while st.session_state.page == "Student View":
        with lock:
            analysis = latest_analysis_data.copy()

        # Display the raw analysis data for debugging
        debug_placeholder.json(analysis)

        # If there is any analysis data, update the UI
        if analysis:
            status = analysis.get("status", "N/A")
            score = analysis.get("score", 0)
            emotion = analysis.get("emotion", "N/A")
            blinks = analysis.get("blinks", 0)
            yawns = analysis.get("yawns", 0)

            status_box.write(f"### Status: **{status}**")
            score_box.metric("Engagement Score", f"{score:.2f}")
            emo_box.metric("Emotion", emotion.capitalize())
            blink_box.metric("Blinks", blinks)
            yawn_box.metric("Yawns", yawns)

            if status == "Low Engagement":
                alert_box.error("Low Engagement Detected!", icon="⚠️")
            else:
                alert_box.empty()
            
            # Update engagement history (check for new score to avoid duplicates)
            if not st.session_state.engagement_history or st.session_state.engagement_history[-1].get("score") != score:
                st.session_state.engagement_history.append({"time": pd.Timestamp.now(), "score": score})
                if len(st.session_state.engagement_history) > 100:
                    st.session_state.engagement_history.pop(0)
        
        if st.session_state.engagement_history:
            history_df = pd.DataFrame(st.session_state.engagement_history).set_index("time")
            chart_box.line_chart(history_df)
        
        time.sleep(1)


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