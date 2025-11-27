# scoring.py

# Weights for each metric. These can be tuned.
WEIGHTS = {
    "emotion": 0.4,
    "eye_focus": 0.3,
    "head_pose": 0.2,
    "activity": 0.1, # For blinks and yawns
}

EMOTION_SCORES = {
    "happy": 1.0,
    "surprised": 0.8,
    "neutral": 0.6,
    "sad": 0.2,
    "angry": 0.1
}

# Thresholds for activity detection
EAR_THRESH = 0.21  # Eye Aspect Ratio for blink
MAR_THRESH = 0.40  # Mouth Aspect Ratio for yawn (lowered)

def compute_engagement(analysis_results, blink_counter, yawn_counter):
    # 1. Emotion Score
    emo_score = EMOTION_SCORES.get(analysis_results["emotion"], 0.5)

    # 2. Eye Focus Score (already normalized in analysis.py)
    eye_focus_score = analysis_results["eye_focus"]

    # 3. Head Pose Score (based on yaw and pitch)
    yaw = abs(analysis_results["yaw"])
    pitch = abs(analysis_results["pitch"])
    # Penalize extreme head poses
    head_pose_score = (1 - (yaw / 45)) * (1 - (pitch / 45))
    head_pose_score = max(0, min(1, head_pose_score))

    # 4. Activity Score (penalize excessive yawning, reward blinking)
    # Simple logic: penalize if yawning, slightly reward if blinking
    yawn_penalty = 0.5 if yawn_counter > 0 else 0
    blink_reward = 0.1 if blink_counter > 0 else 0
    activity_score = 1 - yawn_penalty + blink_reward
    activity_score = max(0, min(1, activity_score))

    # Final Weighted Score
    score = (
        emo_score * WEIGHTS["emotion"] +
        eye_focus_score * WEIGHTS["eye_focus"] +
        head_pose_score * WEIGHTS["head_pose"] +
        activity_score * WEIGHTS["activity"]
    )
    
    # Normalize to be between 0 and 1
    score = score / sum(WEIGHTS.values())

    # Categorize engagement levels
    if score > 0.7:
        status = "High Engagement"
    elif score > 0.4:
        status = "Medium Engagement"
    else:
        status = "Low Engagement"

    return score, status