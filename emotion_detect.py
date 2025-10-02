import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# 🎧 Emotion labels from RAVDESS
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

# ✅ Choose only some emotions (optional, makes training faster at first)
AVAILABLE_EMOTIONS = {'calm', 'happy', 'fearful', 'disgust'}

# 🎼 Feature extractor (no resampy needed)
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    # sr=None -> keep native sampling rate, avoids resampy
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


# 📂 Load dataset
# 📂 Load dataset (recursive)
def load_data(data_path="data/"):
    x, y = [], []
    for root, _, files in os.walk(data_path):
        for file in files:
            if not file.endswith(".wav"):
                continue
            emotion = emotions[file.split("-")[2]]
            if emotion not in AVAILABLE_EMOTIONS:
                continue
            file_path = os.path.join(root, file)
            feature = extract_features(file_path)
            x.append(feature)
            y.append(emotion)
    return np.array(x), np.array(y)

print("Loading data...")
X, y = load_data("data/")  # 👈 change this if your dataset path is different
print(f"Loaded {len(X)} samples.")

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train model
print("Training model...")
model = MLPClassifier(hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(X_train, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

# Save trained model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
