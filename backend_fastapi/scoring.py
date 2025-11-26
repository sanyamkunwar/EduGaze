# scoring.py

emotion_weights = {
    "happy": 1.0,
    "surprised": 0.8,
    "neutral": 0.6,
    "sad": 0.2,
    "angry": 0.1
}

def compute_engagement(eye_focus, emotion, head_orientation):
    # Convert emotion to positivity score
    emo_pos = emotion_weights.get(emotion, 0.0)

    # Engagement formula from the PDF (Page 3)
    # Engagement = (EyeFocus + EmotionPositivity + HeadOrientation) / 3
    score = (eye_focus + emo_pos + head_orientation) / 3

    # Categorize engagement levels
    if score > 0.66:
        status = "High Engagement"
    elif score > 0.33:
        status = "Medium Engagement"
    else:
        status = "Low Engagement"

    return score, status
