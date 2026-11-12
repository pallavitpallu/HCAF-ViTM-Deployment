import streamlit as st
import pandas as pd
from style import apply_custom_style

st.set_page_config(
    page_title="HCAF-ViT-M | Model Performance",
    page_icon="📊",
    layout="wide"
)

apply_custom_style()

st.title("📊 Model Performance Analysis")
st.caption("Evaluation of HCAF-ViT-M on the held-out test dataset.")

st.subheader("Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", "93.62%")
c2.metric("Precision (Fatigue)", "95.40%")
c3.metric("Recall (Fatigue)", "91.61%")
c4.metric("Macro F1-score", "93.62%")

c1, c2 = st.columns(2)
c1.metric("Balanced Accuracy", "93.61%")
c2.metric("ROC-AUC", "98.11%")

st.divider()

st.subheader("Confusion Matrix")
st.caption("Actual and predicted class counts from the test set.")

confusion_matrix = pd.DataFrame(
    [[436, 20], [38, 415]],
    index=["Actual Active", "Actual Fatigue"],
    columns=["Predicted Active", "Predicted Fatigue"]
)

st.dataframe(
    confusion_matrix.style.background_gradient(cmap="Blues"),
    use_container_width=True
)

st.markdown("""
**How to understand this matrix:**
- 436 Active samples were correctly classified.
- 20 Active samples were incorrectly classified as Fatigue.
- 38 Fatigue samples were incorrectly classified as Active.
- 415 Fatigue samples were correctly classified.
""")

st.divider()

st.subheader("Class-wise Performance")

class_metrics = pd.DataFrame({
    "Metric": ["Fatigue Precision", "Fatigue Recall", "Specificity"],
    "Score (%)": [95.40, 91.61, 95.61]
})

st.bar_chart(
    class_metrics.set_index("Metric"),
    y="Score (%)"
)

st.divider()

st.subheader("Model Information")

info1, info2 = st.columns(2)

with info1:
    st.markdown("""
    <div class="project-card">
        <h3>Model Architecture</h3>
        <p>ResNet18 CNN + ViT-Base + Behavioral MLP</p>
        <p>Feature fusion followed by a two-class classifier.</p>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="project-card">
        <h3>Classification Classes</h3>
        <p><b>Active:</b> Class 0</p>
        <p><b>Fatigue:</b> Class 1</p>
    </div>
    """, unsafe_allow_html=True)

st.info(
    "These are recorded test-set evaluation results. "
    "They are not measurements of live webcam accuracy."
)