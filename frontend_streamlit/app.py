import streamlit as st
import time
import queue
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(layout="wide", page_title="Debug Test")

# 1. The queue for communication
result_queue: "queue.Queue[dict]" = queue.Queue()

# 2. A simple processor that just counts and sends data
class CounterProcessor(VideoProcessorBase):
    def __init__(self, result_queue: queue.Queue):
        self.result_queue = result_queue
        self.frame_count = 0
        self.last_put = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Every second, put a simple dictionary into the queue
        if time.time() - self.last_put > 1.0:
            self.frame_count += 1
            self.result_queue.put({"count": self.frame_count, "time": time.time()})
            self.last_put = time.time()
        
        # Return the frame unmodified
        return frame

st.header("Minimal Queue Test")
st.write("This is a diagnostic script. It does not use your backend.")

webrtc_streamer(
    key="test-cam",
    video_processor_factory=lambda: CounterProcessor(result_queue=result_queue),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

st.subheader("Result from Queue:")
debug_area = st.empty()

# 3. The simple UI loop to get data from the queue
while True:
    try:
        result = result_queue.get(timeout=1.0)
        debug_area.json(result)
    except queue.Empty:
        debug_area.write("Queue is empty. Waiting for data from video processor...")
    
    time.sleep(0.5)
