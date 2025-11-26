import cv2
import numpy as np

def convert_frame_to_bytes(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()
