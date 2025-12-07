from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import time
import asyncio
import firebase_admin
from firebase_admin import credentials, db

from analysis import analyze_frame
from scoring import compute_engagement, EAR_THRESH, MAR_THRESH

# --- Constants ---
INACTIVE_USER_TIMEOUT_SECONDS = 15 # Time before a user is considered disconnected

# --- Firebase Setup (Placeholder) ---
# IMPORTANT: Replace with your actual Firebase project credentials
try:
    # cred = credentials.Certificate("path/to/your/firebase-credentials.json")
    # firebase_admin.initialize_app(cred, {
    #     'databaseURL': 'https://your-database-name.firebaseio.com'
    # })
    # FIREBASE_ENABLED = True
    # For this example, we'll assume Firebase is not configured.
    raise ValueError("Firebase not configured")
except Exception as e:
    print(f"Firebase not initialized: {e}")
    print("Running in offline mode. Teacher dashboard will use in-memory data.")
    FIREBASE_ENABLED = False
# ------------------------------------

app = FastAPI()

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory state (per-user) ---
USER_STATES = {}
# In-memory data store for the dashboard
DASHBOARD_DATA = {}


def get_user_state(user_id: str):
    """Gets or creates a state for a given user_id."""
    now = time.time()
    if user_id not in USER_STATES:
        USER_STATES[user_id] = {
            "blink_counter": 0,
            "is_blinking": False,
            "yawn_counter": 0,
            "is_yawning": False,
            "yawn_frames": 0,
            "low_score_start_time": 0, # Timestamp of when score first dropped
            "is_flagged": False,
        }
    USER_STATES[user_id]["last_update"] = now
    return USER_STATES[user_id]


@app.post("/analyze/{user_id}")
async def analyze(user_id: str, file: UploadFile = File(...)):
    global DASHBOARD_DATA

    state = get_user_state(user_id)

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    analysis_results = analyze_frame(img)
    if not analysis_results:
        raise HTTPException(status_code=400, detail="No face detected")

    # --- Blink & Yawn Counting Logic (Improved) ---
    ear = analysis_results["ear"]
    mar = analysis_results["mar"]

    # Blink detection
    if ear < EAR_THRESH:
        state["is_blinking"] = True
    elif state["is_blinking"]:
        state["blink_counter"] += 1
        state["is_blinking"] = False

    # Yawn detection
    if mar > MAR_THRESH:
        state["yawn_frames"] += 1
        if state["yawn_frames"] > 5: # Min frames to count as a yawn
             state["is_yawning"] = True
    elif state["is_yawning"]:
        state["yawn_counter"] += 1
        state["is_yawning"] = False
        state["yawn_frames"] = 0
    else:
        state["yawn_frames"] = 0
    
    # --- Engagement Score ---
    score, status = compute_engagement(
        analysis_results, 
        state["blink_counter"], 
        state["yawn_counter"]
    )

    # --- Flagging Logic ---
    LOW_SCORE_THRESHOLD = 0.4
    FLAG_DURATION_SECONDS = 60

    if score < LOW_SCORE_THRESHOLD:
        if state["low_score_start_time"] == 0:
            # Start the timer if it's not already running
            state["low_score_start_time"] = time.time()
        else:
            # Check if the duration has exceeded the limit
            elapsed_time = time.time() - state["low_score_start_time"]
            if elapsed_time > FLAG_DURATION_SECONDS:
                state["is_flagged"] = True
    else:
        # Reset timer and flag if score recovers
        state["low_score_start_time"] = 0
        state["is_flagged"] = False

    # --- Update Dashboard Data ---
    _, buffer = cv2.imencode(".jpg", cv2.resize(img, (160, 120)))
    img_str = base64.b64encode(buffer).decode("utf-8")

    user_data = {
        "last_updated": time.time(),
        "score": score,
        "status": status,
        "emotion": analysis_results["emotion"],
        "blinks": state["blink_counter"],
        "yawns": state["yawn_counter"],
        "is_flagged": state["is_flagged"],
        "live_frame": f"data:image/jpeg;base64,{img_str}"
    }
    DASHBOARD_DATA[user_id] = user_data

    # --- Push to Firebase (if enabled) ---
    if FIREBASE_ENABLED:
        try:
            db.reference(f'dashboard/{user_id}').set(user_data)
        except Exception as e:
            print(f"Firebase update failed: {e}")

    return {
        **analysis_results,
        "score": float(score),
        "status": status,
        "blinks": state["blink_counter"],
        "yawns": state["yawn_counter"],
        "is_flagged": state["is_flagged"],
    }

@app.get("/dashboard/data")
async def get_dashboard_data():
    """
    Endpoint for the teacher dashboard. It cleans up inactive users
    before returning the data.
    """
    now = time.time()
    
    # Identify and collect inactive user IDs
    inactive_user_ids = [
        user_id for user_id, data in DASHBOARD_DATA.items()
        if now - data.get("last_updated", 0) > INACTIVE_USER_TIMEOUT_SECONDS
    ]
    
    # Remove inactive users from all data stores
    for user_id in inactive_user_ids:
        print(f"Cleaning up inactive user: {user_id}")
        DASHBOARD_DATA.pop(user_id, None)
        USER_STATES.pop(user_id, None)
        
        if FIREBASE_ENABLED:
            try:
                db.reference(f'dashboard/{user_id}').delete()
            except Exception as e:
                print(f"Firebase delete failed for user {user_id}: {e}")

    # Add a version number for debugging
    return {
        "__v__": "backend_v3_debug",
        **DASHBOARD_DATA
    }