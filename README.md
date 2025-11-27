# EduGaze: AI-Powered Student Engagement Monitor

EduGaze is a real-time monitoring tool designed to help educators gauge student engagement during remote learning sessions. Using a student's webcam feed, it leverages computer vision and machine learning to analyze facial expressions, head movements, and other visual cues to produce a live engagement score. The application features a dedicated view for students to monitor their own engagement and a dashboard for teachers to oversee the entire class at a glance.

The project is built with a Python backend using **FastAPI** for high-performance API services and a **Streamlit** frontend for rapid, interactive UI development. The core AI/ML functionality is powered by **TensorFlow/Keras** and **Google's MediaPipe**.

## Features

-   **Weighted Engagement Scoring:** A sophisticated algorithm that combines multiple metrics into a single, easy-to-understand engagement score.
-   **Emotion Detection:** A custom-trained CNN model classifies student emotions (e.g., happy, neutral, sad).
-   **Blink Detection:** Monitors eye aspect ratio to count blinks.
-   **Yawn Detection:** Monitors mouth aspect ratio to detect yawns.
-   **Head Pose Estimation:** Tracks head orientation (pitch, yaw, roll) to determine if a student is looking away.
-   **Face Focus Ring:** A colored ring around the student's face in their live feed provides instant visual feedback on their engagement level.
-   **Live Engagement Trend Graph:** A time-series chart that plots the student's engagement score, allowing them to see trends.
-   **Teacher Dashboard:** A central view for educators with a multi-student grid displaying each student's live feed and key metrics.
-   **Automated Student Flagging:** The dashboard automatically flags students whose engagement has been critically low for over a minute, drawing the teacher's attention.

## How It Works

The application consists of three main components:

1.  **Frontend (Streamlit):** The user-facing web application. It captures video from the user's webcam and sends frames to the backend for processing. It then visualizes the returned data in either the Student View or the Teacher Dashboard.
2.  **Backend (FastAPI):** The server that handles all the heavy lifting. It exposes API endpoints to receive image data, runs the analysis pipeline, and sends back a JSON object with all the calculated metrics.
3.  **AI/ML Models:**
    *   A **TensorFlow/Keras CNN model** (`models/emotion_model.h5`) is used for emotion classification.
    *   **Google's MediaPipe Face Mesh** is used to get detailed facial landmarks in real-time, which are then used to calculate Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head position.

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   An environment manager like `conda` or `venv` is highly recommended.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd EduGaze
```

### 2. Install Backend Dependencies

Navigate to the backend directory and install the required packages.

```bash
cd backend_fastapi
pip install -r requirements.txt
cd ..
```

### 3. Install Frontend Dependencies

Navigate to the frontend directory and install the required packages.

```bash
cd frontend_streamlit
pip install -r requirements.txt
cd ..
```

## Running the Application

Before starting the servers, you must have the emotion detection model in place.

### 1. Train the Model (First-Time Setup)

The pre-trained model `emotion_model.h5` is required by the backend. The script to train this model is included, but it requires a dataset that is not provided in this repository.

-   **Prepare the Dataset:** You must provide your own dataset of facial expression images. Inside the `models/` directory, create two subdirectories: `train/` and `test/`. Inside each of these, create subdirectories for each emotion: `angry`, `happy`, `neutral`, `sad`, `surprised`. Place the corresponding images in these folders.
-   **Run the Training Script:** Once the data is in place, navigate to the `models` directory and run the training script:
    ```bash
    cd models
    python train_emotion_model.py
    ```
    This will create the `emotion_model.h5` file in the `models/` directory.

### 2. Start the Backend Server

With the model in place, you can start the backend.

```bash
cd backend_fastapi
uvicorn main:app --reload
```

The backend will be running at `http://127.0.0.1:8000`.

### 3. Start the Frontend Application

In a new terminal, start the frontend.

```bash
cd frontend_streamlit
streamlit run app.py
```

Your default web browser should open with the EduGaze application running.

## Optional: Firebase Integration

For a more robust, multi-user dashboard experience, the backend is set up to sync data with Google Firebase. To enable this:

1.  Go to the [Firebase Console](https://console.firebase.google.com/) and create a new project.
2.  Create a new **Realtime Database**.
3.  In your Firebase project, go to **Project Settings > Service Accounts** and generate a new private key. This will download a JSON credentials file.
4.  Rename the downloaded file to `firebase-credentials.json` and place it in the `backend_fastapi/` directory.
5.  In `backend_fastapi/main.py`, update the `databaseURL` to match your Firebase project's URL.

## Project Structure

```
.
├── backend_fastapi/
│   ├── main.py             # FastAPI server, API endpoints
│   ├── analysis.py         # Core CV/ML analysis logic
│   ├── scoring.py          # Engagement score calculation
│   └── requirements.txt    # Backend dependencies
│
├── frontend_streamlit/
│   ├── app.py              # Main Streamlit application UI and logic
│   ├── utils.py            # Helper functions for the UI
│   └── requirements.txt    # Frontend dependencies
│
└── models/
    ├── train_emotion_model.py  # Script to train the emotion model
    └── emotion_model.h5        # (Generated) The trained Keras model
```
