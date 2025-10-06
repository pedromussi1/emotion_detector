import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ------------------------------
# 1. Extract Features from Audio
# ------------------------------
def extract_features(file_path):
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ------------------------------
# 2. Load Dataset
# ------------------------------
def load_data(dataset_path):
    emotions = []
    features = []

    print("Loading dataset...")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Customize emotion extraction logic based on your dataset file naming
                if "angry" in file.lower():
                    label = "angry"
                elif "happy" in file.lower():
                    label = "happy"
                elif "sad" in file.lower():
                    label = "sad"
                elif "neutral" in file.lower() or "calm" in file.lower():
                    label = "neutral"
                elif "fear" in file.lower():
                    label = "fearful"
                elif "disgust" in file.lower():
                    label = "disgust"
                elif "surprise" in file.lower():
                    label = "surprise"
                else:
                    continue  # skip unrecognized labels

                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    emotions.append(label)

    X = np.array(features)
    y = np.array(emotions)
    return X, y


# ------------------------------
# 3. Train Model
# ------------------------------
def train_and_save_model(dataset_path, model_output_path="model.pkl"):
    X, y = load_data(dataset_path)

    print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} emotion classes")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nEvaluation Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(model, model_output_path)
    print(f"\n✅ Model saved as {model_output_path}")


# ------------------------------
# 4. Run Script
# ------------------------------
if __name__ == "__main__":
    dataset_dir = "data/"  # 👈 change this to your dataset folder path
    train_and_save_model(dataset_dir)
