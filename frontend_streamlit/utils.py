import cv2
import numpy as np

def convert_frame_to_bytes(frame):
    """Converts a video frame to a byte buffer for API transmission."""
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()

def draw_focus_ring(frame, bbox, status):
    """
    Draws a colored ring around the user's face based on engagement status.
    
    Args:
        frame: The video frame to draw on.
        bbox: The bounding box of the face [x1, y1, x2, y2].
        status: The engagement status string.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Make radius responsive to face size
    radius = (x2 - x1) // 2 + 20 

    color_map = {
        "High Engagement": (0, 255, 0),   # Green
        "Medium Engagement": (0, 255, 255), # Yellow
        "Low Engagement": (0, 0, 255),    # Red
    }
    color = color_map.get(status, (128, 128, 128)) # Default to gray

    # Draw a thicker ring
    cv2.circle(frame, (center_x, center_y), radius, color, thickness=5)

    return frame