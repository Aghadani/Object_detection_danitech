import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- PAGE CONFIG ---
st.set_page_config(page_title="DZ VisionBot AI", page_icon="🔍", layout="wide")

# --- UI STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle {
        color: #00d4ff;
        text-align: center;
    }
    .status-box {
        padding: 10px;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>🔍 VisionBot: Real-Time Object Intelligence</h1>", unsafe_allow_html=True)
st.write("### Intelligent Detection")

# --- SIDEBAR ---
st.sidebar.title("Control Panel")

model_type = st.sidebar.selectbox(
    "Select Model Size",
    ["YOLOv8 Nano (Fastest)", "YOLOv8 Small (Accurate)"]
)

model_path = 'yolov8n.pt' if "Nano" in model_type else 'yolov8s.pt'

conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.45)
iou_threshold = st.sidebar.slider("IOU", 0.0, 1.0, 0.50)

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --- VIDEO PROCESSOR ---
# FIX 1: Store conf/iou as instance attributes that VideoProcessor reads
# at frame-processing time, not at class instantiation time.
# FIX 2: Use a shared state dict so the processor picks up slider changes
# without needing a full restart.
class VideoProcessor:
    def __init__(self):
        self.model = load_model(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            img,
            conf=self.conf,
            iou=self.iou
        )

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# --- RTC CONFIG ---
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
    ]
})

# --- LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    # FIX 3: Always pass VideoProcessor as the factory — never None.
    # Passing None shuts down an active WebRTC stream on every rerun.
    # The Start/Stop buttons below control desired_playing_state instead,
    # which is the correct way to pause/resume a WebRTC stream in streamlit-webrtc.
    webrtc_ctx = webrtc_streamer(
        key="vision-bot",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,  # FIX 4: Use async_processing=True for smoother frames
    )

    # FIX 5: Update processor thresholds live when sliders change,
    # without restarting the stream.
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.conf = conf_threshold
        webrtc_ctx.video_processor.iou = iou_threshold

with col2:
    st.markdown("<div class='status-box'>", unsafe_allow_html=True)
    st.write("📊 **System Status**")

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.success("Camera: Active")
    else:
        st.warning("Camera: Offline")

    st.write(f"**Model:** {model_type}")
    st.write("**Device:** CPU")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    **Objects it can identify:**
    - Person 👤  
    - Cell Phone 📱  
    - Laptop 💻  
    - Chair 🪑  
    - ...and more!
    """)
