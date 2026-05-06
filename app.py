import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- UI CONFIG ---
st.set_page_config(page_title="VisionBot AI", page_icon="🔍", layout="wide")

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

# --- START BUTTON (IMPORTANT FIX) ---
run = st.checkbox("Start Camera")

# --- VIDEO PROCESSOR ---
class VideoProcessor:
    def __init__(self):
        # Load model here to avoid blocking stream start
        self.model = YOLO(model_path)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold
        )

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# --- RTC CONFIG (IMPROVED) ---
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
    ]
})

col1, col2 = st.columns([3, 1])

with col1:
    if run:
        webrtc_ctx = webrtc_streamer(
            key="vision-bot",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,  # IMPORTANT FIX
        )
    else:
        webrtc_ctx = None
        st.info("Click 'Start Camera' to begin")

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
