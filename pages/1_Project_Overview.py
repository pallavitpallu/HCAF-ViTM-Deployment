
import streamlit as st
from style import apply_custom_style

st.set_page_config(
    page_title="HCAF-ViT-M | AI Driver Safety",
    page_icon="🚗",
    layout="wide"
)

apply_custom_style()

# Hero section
st.markdown("""
<div class="project-hero">
    <div class="eyebrow">INTELLIGENT TRANSPORTATION • DEEP LEARNING</div>
    <h1>HCAF-ViT-M</h1>
    <h3 style="color:#61C9FF !important;">
        Hybrid CNN–Vision Transformer
    </h3>
    <p>
        An AI-powered driver drowsiness detection research system
        combining facial visual features and behavioural analysis
        for driver-state classification.
    </p>
    <span style="color:#49D8F0;font-weight:700;">
        ● RESEARCH PROTOTYPE &nbsp; | &nbsp; REAL-TIME COMPUTER VISION
    </span>
</div>
""", unsafe_allow_html=True)

# Key results
st.markdown('<div class="section-label">MODEL PERFORMANCE</div>',
            unsafe_allow_html=True)
st.subheader("Test Dataset Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", "93.62%")
m2.metric("ROC-AUC", "98.11%")
m3.metric("Macro F1", "93.62%")
m4.metric("Balanced Accuracy", "93.61%")

st.divider()

# Project overview
st.markdown('<div class="section-label">PROJECT OVERVIEW</div>',
            unsafe_allow_html=True)
st.header("Why this project?")

left, right = st.columns([1.2, 1])

with left:
    st.markdown("""
    <div class="project-card">
        <h3>🎯 The Problem</h3>
        <p>
        Driver fatigue can reduce alertness and reaction time.
        Detecting signs of fatigue through facial and behavioural
        patterns is an important research direction in road safety.
        </p>
        <br>
        <h3>💡 Our Approach</h3>
        <p>
        HCAF-ViT-M combines CNN-based visual feature extraction,
        Vision Transformer attention features and a behavioural
        feature branch to classify driver state.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="project-card">
        <h3>⚡ System Highlights</h3>
        <p>◈ Hybrid CNN and Transformer architecture</p>
        <p>◈ Facial behavioural feature extraction</p>
        <p>◈ Feature-level fusion</p>
        <p>◈ Active / Fatigue classification</p>
        <p>◈ Webcam-based live inference</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Architecture
st.markdown('<div class="section-label">MODEL ARCHITECTURE</div>',
            unsafe_allow_html=True)
st.header("Three branches. One intelligent prediction.")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="project-card">
        <div class="eyebrow">BRANCH 01</div>
        <h2>🧠 ResNet18</h2>
        <h3>CNN branch</h3>
        <p>Extracts local visual representations from facial images.</p>
        <h2 style="color:#22D3EE;">256-D</h2>
        <p>Projected feature vector</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="project-card">
        <div class="eyebrow">BRANCH 02</div>
        <h2>🔍 ViT-Base</h2>
        <h3>Transformer branch</h3>
        <p>Uses self-attention to capture global visual relationships.</p>
        <h2 style="color:#22D3EE;">256-D</h2>
        <p>Projected feature vector</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="project-card">
        <div class="eyebrow">BRANCH 03</div>
        <h2>📊 Behavioural MLP</h2>
        <h3>Feature branch</h3>
        <p>Processes EAR, MAR and head movement features.</p>
        <h2 style="color:#22D3EE;">128-D</h2>
        <p>Behavioural representation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="project-card" style="text-align:center;">
    <div class="eyebrow">FEATURE FUSION</div>
    <h2>256 + 256 + 128 = 640 Features</h2>
    <p>Combined representation → Fusion network → Two-class classifier</p>
    <h2 style="color:#22D3EE;">ACTIVE / FATIGUE</h2>
</div>
""", unsafe_allow_html=True)

st.divider()

# Dataset
st.markdown('<div class="section-label">DATASET OVERVIEW</div>',
            unsafe_allow_html=True)
st.header("UTA-RLDD Dataset")

d1, d2, d3 = st.columns(3)
d1.metric("Training Images", "5,539")
d2.metric("Validation Images", "692")
d3.metric("Test Images", "693")

st.caption(
    "The loaded image split contains two classes: Alert and Drowsy. "
    "The model maps these to Active (0) and Fatigue (1)."
)

st.divider()

# Project information
st.markdown('<div class="section-label">PROJECT INFORMATION</div>',
            unsafe_allow_html=True)

p1, p2 = st.columns(2)

with p1:
    st.markdown("""
    <div class="project-card">
        <h3>🛠 Technologies</h3>
        <p>Python • PyTorch • TorchVision</p>
        <p>Hugging Face Transformers • OpenCV</p>
        <p>MediaPipe • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="project-card">
        <h3>🔬 Research Focus</h3>
        <p>Hybrid deep learning</p>
        <p>Vision Transformer feature learning</p>
        <p>Facial behavioural analysis</p>
        <p>Real-time driver monitoring</p>
    </div>
    """, unsafe_allow_html=True)

st.info(
    "Evaluation metrics shown here are from the held-out test dataset. "
    "They do not represent live webcam accuracy. This is a research "
    "prototype, not a certified driver-safety system."
)
