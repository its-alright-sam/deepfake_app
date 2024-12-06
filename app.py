import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Load the pretrained model
MODEL_PATH = 'D:\github project stuff\model.h5'
model = load_model(MODEL_PATH)

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to model's input size
    frame = preprocess_input(frame)
    return frame

# Function to process video and detect deepfake
def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file."

    predictions = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analyze every 10th frame
        if frame_count % 10 == 0:
            preprocessed_frame = preprocess_frame(frame)
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
            pred = model.predict(preprocessed_frame)
            predictions.append(pred[0][0])

            # Display progress in the UI
            st.write(f"Processing frame {frame_count}/{total_frames}...")

    cap.release()
    avg_fake_prob = np.mean(predictions)
    return avg_fake_prob, None

# Streamlit UI
st.title("Deepfake Video Detector")
st.write("Upload a video to check if it's a deepfake.")

# Upload video file
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video locally
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Process video
    st.write("Analyzing video...")
    fake_prob, error = detect_deepfake("uploaded_video.mp4")

    if error:
        st.error(error)
    else:
        st.success(f"Analysis complete! Average fake probability: {fake_prob:.4f}")
        if fake_prob > 0.5:
            st.error("The video is likely a deepfake.")
        else:
            st.success("The video is likely real.")
