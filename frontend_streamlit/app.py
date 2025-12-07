import streamlit as st
import cv2
import requests
import time
import pandas as pd
import numpy as np
import uuid
import base64
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
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# --- Backend URL ---
BACKEND_URL = "https://edugaze-backend.onrender.com"

# --- WebRTC Video Processor ---
# This processor sends frames to the backend and draws the response on the frame.
# It does NOT communicate back to the main Streamlit thread.
class EduGazeVideoProcessor(VideoProcessorBase):
    def __init__(self, user_id: str):
        self.last_sent = 0
        self.user_id = user_id
        self.last_analysis_data = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Send frame to backend for analysis
        if time.time() - self.last_sent > 1.5:
            self.last_sent = time.time()
            try:
                b = convert_frame_to_bytes(img)
                resp = requests.post(
                    f"{BACKEND_URL}/analyze/{self.user_id}",
                    files={"file": ("f.jpg", b, "image/jpeg")},
                    timeout=10
                ).json()
                self.last_analysis_data = resp
            except requests.exceptions.RequestException as e:
                print(f"Backend POST error: {e}")
                self.last_analysis_data = {"error": "Backend connection failed."}
            except Exception as e:
                print(f"An error occurred in video processor: {e}")
                self.last_analysis_data = {"error": "Analysis failed."}

        # Draw feedback on the frame using the last known data
        face_bbox = self.last_analysis_data.get("face_bbox")
        status = self.last_analysis_data.get("status")
        error = self.last_analysis_data.get("error")
        
        if face_bbox:
            img = draw_focus_ring(img, face_bbox, status)
        
        if error:
            cv2.putText(img, error, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
        st.write("Click 'Start' to begin the session. Your browser will ask for camera permission.")
        webrtc_streamer(
            key="student-camera",
            video_processor_factory=lambda: EduGazeVideoProcessor(user_id=st.session_state.user_id),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

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

    # UI update loop - now polls the backend for its own data
    while st.session_state.page == "Student View":
        analysis = None
        try:
            # Fetch all dashboard data
            all_data = requests.get(f"{BACKEND_URL}/dashboard/data", timeout=5).json()
            # Get data for the current user
            analysis = all_data.get(st.session_state.user_id)
        except requests.exceptions.RequestException:
            # This is expected if the backend is slow to respond initially
            pass
        except Exception:
            # Catch other potential errors like JSON decoding
            pass

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
            
            # Update engagement history
            if not st.session_state.engagement_history or st.session_state.engagement_history[-1].get("score") != score:
                st.session_state.engagement_history.append({"time": pd.Timestamp.now(), "score": score})
                if len(st.session_state.engagement_history) > 100:
                    st.session_state.engagement_history.pop(0)
        
        if st.session_state.engagement_history:
            history_df = pd.DataFrame(st.session_state.engagement_history).set_index("time")
            chart_box.line_chart(history_df)
        
        time.sleep(2) # Poll every 2 seconds

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
                    
                    cols = st.columns(4)
                    
                    for i in range(num_students):
                        user_id = student_ids[i]
                        data = resp[user_id]
                        col = cols[i % 4]

                        with col:
                            st.subheader(f"Student #{user_id[:4]}")
                            
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
                            
                            if not is_flagged:
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

        time.sleep(2)