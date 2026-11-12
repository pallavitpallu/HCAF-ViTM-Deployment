
import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Webcam Connection Test")

st.write("Testing browser webcam access.")

webrtc_streamer(
    key="webcam-test",
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

