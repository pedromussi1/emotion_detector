import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

# Feature extraction
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

def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    prob_dict = {label: prob for label, prob in zip(model.classes_, probabilities)}
    return prediction, prob_dict

# Streamlit UI
st.title("🎤 Emotion Detector from Voice")
st.write("Upload a `.wav` file to detect the emotion in the voice!")

uploaded_file = st.file_uploader("Upload your audio file", type=["wav"])

if uploaded_file is not None:
    # Play uploaded audio
    st.audio(uploaded_file, format="audio/wav")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Predict emotion
    with st.spinner("Analyzing emotion..."):
        prediction, prob_dict = predict_emotion(temp_path)

    st.success(f"Detected Emotion: **{prediction.upper()}** 🎯")

    # Show probability chart
    df = pd.DataFrame(prob_dict, index=[0]).T.rename(columns={0: "Probability"})
    st.bar_chart(df)
