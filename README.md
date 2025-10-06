<h1 align="center">🎤 Emotion Detector from Voice</h1>

<p align="center">
  <a href="https://youtu.be/OthfVoaf9wE"><img src="https://i.imgur.com/AhK9NWN.gif" alt="YouTube Demonstration" width="800"></a>
</p>

<p align="center">A machine learning web application that detects human emotions from voice recordings using audio feature extraction and a trained model, powered by Streamlit, Librosa, and Scikit-learn.</p>

<h3>Try the live app here: <a href="https://emotiondetector-asthtb5zdfd2dcqtg3pq3i.streamlit.app/">https://emotion-detector-app.streamlit.app/</a></h3>

<h2>Description</h2>
<p>The Emotion Detector from Voice is an interactive web application that analyzes short voice clips to predict the speaker’s emotion. Using advanced audio processing techniques like MFCC, Chroma, and Mel spectrograms, the system extracts meaningful features from audio signals and classifies emotions such as <b>happy</b>, <b>calm</b>, <b>fearful</b>, and <b>disgust</b>. This project demonstrates the power of machine learning in speech emotion recognition, a growing field with applications in human-computer interaction, sentiment analysis, and affective computing.</p>

<h2>Languages and Utilities Used</h2>
<ul>
    <li><b>Python:</b> The core programming language for feature extraction, model training, and integration with Streamlit.</li>
    <li><b>Streamlit:</b> Used to build the interactive web interface for emotion detection and audio playback.</li>
    <li><b>Librosa:</b> Handles audio loading, processing, and extraction of MFCC, Chroma, and Mel features.</li>
    <li><b>Scikit-learn:</b> Used for training and saving the machine learning model.</li>
    <li><b>Joblib:</b> Saves and loads the trained emotion detection model efficiently.</li>
    <li><b>Soundfile:</b> Enables Streamlit to play uploaded `.wav` audio files.</li>
    <li><b>Pandas:</b> Used for visualizing model prediction probabilities as a bar chart.</li>
</ul>

<h2>Environments Used</h2>
<ul>
    <li><b>Windows 11</b></li>
    <li><b>Visual Studio Code</b></li>
</ul>

<h2>Installation</h2>
<ol>
    <li><strong>Clone the Repository:</strong>
        <pre><code>git clone https://github.com/yourusername/emotion-detector-app.git
cd emotion-detector-app</code></pre>
    </li>
    <li><strong>Create and Activate a Virtual Environment:</strong>
        <pre><code>python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`</code></pre>
    </li>
    <li><strong>Install Dependencies:</strong>
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li><strong>Run the Application:</strong>
        <pre><code>streamlit run emotion_app.py</code></pre>
        The application will launch automatically in your browser.
    </li>
</ol>

<h2>Usage</h2>
<ol>
    <li>Open the web application in your browser.</li>
    <li>Upload a short `.wav` voice recording.</li>
    <li>The app will extract audio features and analyze the emotion using the trained model.</li>
    <li>View the predicted emotion and a bar chart showing the probabilities for each class.</li>
</ol>

<h2>Code Structure</h2>
<ul>
    <li><strong>emotion_app.py:</strong> The main Streamlit app file responsible for UI, feature extraction, and emotion prediction.</li>
    <li><strong>model.pkl:</strong> The pre-trained emotion detection model loaded by the app.</li>
    <li><strong>requirements.txt:</strong> Contains all necessary Python dependencies for the project.</li>
</ul>

<h2>Known Issues</h2>
<ul>
    <li>Only `.wav` files are supported for audio uploads.</li>
    <li>Background noise or very short clips may affect prediction accuracy.</li>
    <li>Live microphone input is not supported on Streamlit Cloud (upload `.wav` instead).</li>
</ul>

<h2>Contributing</h2>
<p>Contributions are welcome! Feel free to fork this repository, make improvements, and open a pull request. For major changes, please open an issue first to discuss your ideas.</p>

<h2>Deployment</h2>
<p>The application is hosted on <b>Streamlit Cloud</b>, which automatically builds the environment based on <code>requirements.txt</code> and serves the app in a web-friendly format. Streamlit handles dependency installation, deployment, and version control integration with GitHub for seamless updates.</p>

<h2><a href="https://github.com/pedromussi1/emotion-detector/blob/main/train_model.py">Model Training Code (optional link)</a></h2>

<h3>Upload Audio</h3>
<p align="center">
    <img src="https://i.imgur.com/WThXjye.png" alt="Upload Audio">
</p>
<p>The main interface allows the user to upload a short `.wav` file. The application then processes the file, extracts audio features, and performs emotion classification.</p>

<hr>

<h3>Emotion Prediction Results</h3>
<p align="center">
    <img src="https://i.imgur.com/7lUzBuL.png" alt="Prediction Results">
</p>
<p>After analysis, the application displays the predicted emotion and a probability chart representing confidence across different emotion classes.</p>
