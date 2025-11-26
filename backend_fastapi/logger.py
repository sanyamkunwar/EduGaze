import pandas as pd
from datetime import datetime
import os

LOG_FILE = "logs/engagement_log.csv"

def write_log(emotion, eye_focus, head_orientation, score):
    os.makedirs("logs", exist_ok=True)

    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion,
        "eye_focus": eye_focus,
        "head_orientation": head_orientation,
        "score": score
    }])

    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
