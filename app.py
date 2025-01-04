from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from werkzeug.utils import secure_filename
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('ResNet50V2_Model.h5')

# Define emotion classes
Emotion_Classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load music player data
Music_Player = pd.read_csv('data_moods.csv')

# Ensure the uploads directory exists
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Preprocess the image and detect faces
def load_and_prep_image(filename, img_shape=224):
    try:
        img = cv2.imread(filename)

        if img is None:
            logging.error("Failed to load image.")
            return None

        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

        if len(faces) == 0:
            logging.warning("No face detected in the image.")
            return None

        # Process the first detected face
        x, y, w, h = faces[0]
        face_img = img[y:y + h, x:x + w]

        # Resize and normalize the face image
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (img_shape, img_shape))
        rgb_img = rgb_img / 255.0

        return rgb_img
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return None

# Predict emotion and recommend songs
def predict_emotion_and_recommend(filename):
    img = load_and_prep_image(filename)

    if img is None:
        return "Error", []

    try:
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_class = Emotion_Classes[np.argmax(pred)]
        recommendations = recommend_songs(pred_class)
        return pred_class, recommendations
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "Error", []

# Recommend songs based on predicted emotions
def recommend_songs(pred_class):
    mood_map = {
        'Disgust': 'Sad',
        'Happy': 'Happy',
        'Sad': 'Happy',
        'Fear': 'Calm',
        'Angry': 'Calm',
        'Surprise': 'Energetic',
        'Neutral': 'Energetic'
    }

    mood = mood_map.get(pred_class, 'Neutral')
    return get_music_recommendations(mood)

# Get music recommendations for a given mood
def get_music_recommendations(mood):
    songs = Music_Player[Music_Player['mood'] == mood]
    songs = songs.sort_values(by="popularity", ascending=False)[:5]
    return songs[['album', 'artist', 'name', 'popularity', 'release_date']].to_dict(orient='records')

@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for prediction and recommendation
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Run prediction and get recommendations
        emotion, recommendations = predict_emotion_and_recommend(file_path)

        if emotion == "Error":
            return jsonify({'error': 'Failed to process image. Ensure the image contains a clear face.'}), 400

        return jsonify({'emotion': emotion, 'recommendations': recommendations}), 200
    except Exception as e:
        logging.error(f"Unexpected error in /predict endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
