from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import cv2
import numpy as np
import os
import random
import joblib
import pygame
import time

app = Flask(__name__)

# Initialize Pygame mixer for music playback
pygame.mixer.init()

# Initialize global variables
current_emotion = None
camera = None

# Load the pre-trained emotion model
emotion_model = joblib.load('emotion_model.pkl')

# Define a function to extract features from face (mouth width and height)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    mouth = mouth_cascade.detectMultiScale(gray_face, 1.1, 3)
    if len(mouth) == 0:
        return None
    mx, my, mw, mh = mouth[0]
    return [mw, mh]

# Function to detect emotion from an image using the trained model
def detect_emotion(image):
    features = extract_features(image)
    if features is None:
        return "No face detected"
    emotion = emotion_model.predict([features])
    return emotion[0]

# Function to play song based on emotion
def play_song_for_emotion(emotion):
    global current_emotion
    if emotion == current_emotion:
        return
    current_emotion = emotion
    emotion_folder = os.path.join('song', emotion)
    if not os.path.exists(emotion_folder) or not os.listdir(emotion_folder):
        print(f"No songs found for emotion: {emotion}")
        return
    song = random.choice(os.listdir(emotion_folder))
    song_path = os.path.join(emotion_folder, song)
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

def stop_song():
    pygame.mixer.music.stop()

# Real-time video stream
def gen_frames():
    global camera
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        emotion = detect_emotion(frame)
        play_song_for_emotion(emotion)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for the home page
@app.route('/')
def index():
    stop_song()  # Stop music when navigating back to home
    return render_template('index.html')

# Route for real-time emotion detection
@app.route('/real_time')
def real_time():
    global camera
    camera = cv2.VideoCapture(0)  # Start the camera
    return render_template('real_time.html')

# Route for displaying real-time video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to stop the camera when leaving the page
@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()  # Release the camera when leaving the page
    stop_song()  # Stop music when leaving the real-time page
    return redirect(url_for('index'))

# Route for image upload
@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Read the image file
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            emotion = detect_emotion(img)
            play_song_for_emotion(emotion)  # Ensure song is played after emotion is detected
            return render_template('upload_image.html', emotion=emotion)
    return render_template('upload_image.html', emotion=None)

# Route to stop music when moving back to home screen
@app.route('/stop_music')
def stop_music():
    stop_song()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)