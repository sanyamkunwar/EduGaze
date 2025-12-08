# EduGaze: AI-Powered Student Engagement Monitor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://edugaze.streamlit.app/)

EduGaze is a real-time monitoring tool designed to help educators gauge student engagement during remote learning sessions. Using a student's webcam feed, it leverages computer vision and machine learning to analyze facial expressions, head movements, and other visual cues to produce a live engagement score. The application features a dedicated view for students to monitor their own engagement and a dashboard for teachers to oversee the entire class at a glance.

---

## Live Demo

**You can try the live application here:** [**https://edugaze.streamlit.app/**](https://edugaze.streamlit.app/)
**Render Url:** [**https://edugaze-backend.onrender.com/**](https://edugaze-backend.onrender.com/)

*Note: The backend is hosted on a free Render instance, which may "spin down" from inactivity. The first analysis may take up to 30 seconds to complete as the server wakes up.*

---

## Features

-   **Weighted Engagement Scoring:** A sophisticated algorithm that combines multiple metrics into a single, easy-to-understand engagement score.
-   **Emotion Detection:** A custom-trained CNN model classifies student emotions (e.g., happy, neutral, sad).
-   **Blink & Yawn Detection:** Monitors eye and mouth aspect ratios to detect signs of fatigue.
-   **Head Pose Estimation:** Tracks head orientation to determine if a student is looking away.
-   **Live Visual Feedback:** A colored ring around the student's face provides instant feedback on their engagement level.
-   **Real-time Engagement Chart:** A time-series graph plots the student's engagement score, allowing them to see trends.
-   **Teacher Dashboard:** A central view for educators with a multi-student grid displaying each student's live feed and key metrics.
-   **Automated Student Flagging:** The dashboard automatically flags students whose engagement has been critically low for an extended period.

---

## Tech Stack & Architecture

This project uses a modern Python stack and a decoupled frontend/backend architecture.

**Tech Stack:**
-   **Backend:** Python, FastAPI, Uvicorn
-   **Frontend:** Streamlit
-   **AI/ML:** TensorFlow/Keras, Google MediaPipe, OpenCV
-   **Deployment:** Render (Backend), Streamlit Community Cloud (Frontend)

**Architecture:**
1.  **Frontend (Streamlit):** The user-facing web application deployed on Streamlit Cloud. It uses `streamlit-webrtc` to capture video from the user's browser.
    -   The **video processor thread** sends frames to the backend for analysis and draws real-time feedback (the face circle) on the video stream.
    -   The **main UI thread** polls the backend for the latest analysis data to display the metrics and charts for the student.
2.  **Backend (FastAPI):** A high-performance server deployed on Render. It exposes API endpoints to:
    -   Receive image data from the frontend.
    -   Run the analysis pipeline (face detection, landmark extraction, model inference).
    -   Store the latest results for all active students in memory.
    -   Serve the collected data to the Teacher and Student dashboards.
3.  **AI/ML Models:**
    -   A **TensorFlow/Keras CNN model** (`models/emotion_model.h5`) is used for emotion classification.
    -   **Google's MediaPipe Face Mesh** provides real-time facial landmarks for calculating EAR, MAR, and head position.

---

## Local Setup and Running

### Prerequisites

-   Python 3.10+
-   An environment manager like `venv` or `conda` is highly recommended.
-   The trained model `models/emotion_model.h5` must be present.

### 1. Clone the Repository

```bash
git clone https://github.com/sanyamkunwar/EduGaze.git
cd EduGaze
```

### 2. Create a Virtual Environment & Install Dependencies

Create and activate a virtual environment, then install all required packages from the unified `requirements.txt` file.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate
# Or on Windows:
# venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Start the Backend Server

In your terminal, start the FastAPI backend.

```bash
uvicorn backend_fastapi.main:app --reload
```
The backend will be running at `http://127.0.0.1:8000`.

### 4. Start the Frontend Application

In a **new terminal** (while the backend is still running), start the Streamlit frontend.

```bash
streamlit run frontend_streamlit/app.py
```
Your default web browser should open with the EduGaze application running.

---

## Deployment

This application is deployed using a free-tier strategy:
-   The **FastAPI backend** is deployed as a Web Service on **Render**.
-   The **Streamlit frontend** is deployed on **Streamlit Community Cloud**.

### Handling Secrets (Firebase)

To enable Firebase integration for a more persistent dashboard, the backend is configured to use environment variables (this is essential for deployment).

On Render, in the "Environment" tab for your backend service, you would set the following:
-   `PYTHON_VERSION`: `3.10` (or your desired version)
-   `FIREBASE_DB_URL`: The URL for your Firebase Realtime Database.
-   `FIREBASE_CREDS_JSON`: The entire content of your Firebase service account JSON file, pasted as a single line.
