import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# --- UI/UX CONFIGURATION ---
st.set_page_config(page_title="VisionBot AI", page_icon="🔍", layout="wide")

# Custom CSS for a modern "Dark Mode" look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTitle {
        color: #00d4ff;
        font-family: 'Helvetica Neue', sans-serif;
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
st.write("### Intelligent Detection with Sub-50ms Latency")

# --- SIDEBAR UI ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103807.png", width=100)
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

# Model Selection
model_type = st.sidebar.selectbox("Select Model Size", ["YOLOv8 Nano (Fastest)", "YOLOv8 Small (Accurate)"])
model_path = 'yolov8n.pt' if "Nano" in model_type else 'yolov8s.pt'

# Confidence and IOU Sliders
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
iou_threshold = st.sidebar.slider("Overlap (IOU) Threshold", 0.0, 1.0, 0.50)

st.sidebar.markdown("---")
st.sidebar.info("This AI can detect 80 different classes like people, cars, laptops, and more.")

# --- ALGORITHM LOGIC ---
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

model = load_yolo_model(model_path)

class VideoProcessor:
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 1. Run Inference
        results = self.model.predict(img, conf=conf_threshold, iou=iou_threshold)

        # 2. Annotate Frame (Draws boxes and names)
        # plot() automatically adds the class name and probability score
        annotated_image = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# --- DEPLOYMENT INTERFACE ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="vision-bot",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("<div class='status-box'>", unsafe_allow_html=True)
    st.write("📊 **System Status**")
    if webrtc_ctx.state.playing:
        st.success("Camera: Active")
    else:
        st.warning("Camera: Offline")
    
    st.write(f"**Model:** {model_type}")
    st.write(f"**Device:** CPU/GPU (Auto)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    **Objects it can identify:**
    - Person 👤
    - Cell Phone 📱
    - Laptop 💻
    - Chair 🪑
    - ...and 76 more!
    """)