import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import soundfile as sf
import tempfile
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

# Emotion labels
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
AVAILABLE_EMOTIONS = ['calm', 'happy', 'fearful', 'disgust']

# Feature extractor
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_path, sr=None)
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feat))

    return result

# --- Streamlit UI ---
st.title("🎤 Emotion Detector from Voice")
st.write("Record your voice or upload a `.wav` file to detect your emotion!")

def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    prob_dict = {label: prob for label, prob in zip(model.classes_, probabilities)}
    return prediction, prob_dict

# --- Option 1: Upload a file ---
uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # play uploaded audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(uploaded_file.read())
        temp_path = tmpfile.name

    prediction, prob_dict = predict_emotion(temp_path)
    st.success(f"Predicted Emotion: **{prediction.upper()}** 🎯")

    # Display probabilities as bar chart
    df = pd.DataFrame(prob_dict, index=[0]).T.rename(columns={0: "Probability"})
    st.bar_chart(df)

# --- Option 2: Record from microphone ---
st.write("Or record your voice:")
duration = st.slider("Recording duration (seconds)", 1, 10, 3)
if st.button("Record"):
    st.info("Recording...")
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()
    st.success("Recording complete!")

    # Save temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, recording, 44100)
        temp_path = tmpfile.name

    st.audio(temp_path, format="audio/wav")  # play recorded audio
    prediction, prob_dict = predict_emotion(temp_path)
    st.success(f"Predicted Emotion: **{prediction.upper()}** 🎯")

    # Display probabilities as bar chart
    df = pd.DataFrame(prob_dict, index=[0]).T.rename(columns={0: "Probability"})
    st.bar_chart(df)
