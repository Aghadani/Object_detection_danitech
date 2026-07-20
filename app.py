import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# PAGE CONFIG 
st.set_page_config(page_title="VisionBot AI", page_icon="🔍", layout="wide")

# UI STYLE 
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { color: #00d4ff; text-align: center; }
    .status-box {
        padding: 10px;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #00d4ff;
    }
    [data-testid="stCameraInput"] video {
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>🔍 VisionBot: Real-Time Object Intelligence</h1>", unsafe_allow_html=True)
st.write("### Intelligent Detection")

# SIDEBAR 
st.sidebar.title("Control Panel")

model_type = st.sidebar.selectbox(
    "Select Model Size",
    ["YOLOv8 Nano (Fastest)", "YOLOv8 Small (Accurate)"]
)
model_path = 'yolov8n.pt' if "Nano" in model_type else 'yolov8s.pt'

conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.45)
iou_threshold = st.sidebar.slider("IOU", 0.0, 1.0, 0.50)

st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Allow camera access when prompted\n2. Point your camera at objects\n3. Click the capture button — detection runs instantly")

#  LOAD MODEL 
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)

#  LAYOUT
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("#### 📷 Live Camera Feed")
    camera_frame = st.camera_input(
        label="Camera",
        label_visibility="collapsed"
    )

    if camera_frame is not None:
        file_bytes = np.asarray(bytearray(camera_frame.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Detection Output", use_column_width=True)

        detected = results[0].names
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            labels = [detected[int(c)] for c in boxes.cls]
            unique_labels = list(set(labels))
            st.success(f"Detected: {', '.join(unique_labels)}")
        else:
            st.info("No objects detected — try lowering the Confidence slider.")

with col2:
    st.markdown("<div class='status-box'>", unsafe_allow_html=True)
    st.write("📊 **System Status**")

    if camera_frame is not None:
        st.success("Camera: Active ✅")
    else:
        st.warning("Camera: Waiting...")

    st.write(f"**Model:** {model_type}")
    st.write("**Device:** CPU")
    st.write(f"**Confidence:** {conf_threshold:.2f}")
    st.write(f"**IOU:** {iou_threshold:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    **Objects it can identify:**
    - Person 👤
    - Cell Phone 📱
    - Laptop 💻
    - Chair 🪑
    - Car 🚗
    - Bottle 🍼
    - ...and 74 more!
    """)
