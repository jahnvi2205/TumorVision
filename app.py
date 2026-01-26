import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.keras.models import load_model
from utils import predict

PORT = int(os.environ.get("PORT", 8501))

# 🔹 Load model ONCE
@st.cache_resource
def load_dl_model():
    return load_model("model.h5")

model = load_dl_model()

# Page config
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>Brain Tumor Detector 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI scan to detect brain tumor</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    st.markdown("### 🔍 Prediction Result")

    with st.spinner("Analyzing image..."):
        label, confidence = predict(image_np, model)   # ✅ model passed

    if label == "Tumor":
        st.error(f"⚠️ **Tumor Detected** ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ **No Tumor Detected** ({confidence:.2f}% confidence)")

    st.markdown("---")
    st.info("⚠️ This tool is for educational purposes only.")

