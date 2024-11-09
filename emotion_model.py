import cv2
import numpy as np
from sklearn.svm import SVC
import joblib
import os

# Load the pre-trained Haar Cascade classifiers for face and mouth detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Define a function to extract features from face (mouth width and height)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    
    # Convert to grayscale for feature extraction
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Detect mouth region within the face
    mouth = mouth_cascade.detectMultiScale(gray_face, 1.1, 3)
    
    if len(mouth) == 0:
        return None

    mx, my, mw, mh = mouth[0]
    mouth_width = mw
    mouth_height = mh
    
    return [mouth_width, mouth_height]

# Function to detect emotion from an image using the trained model
def detect_emotion(image):
    # Load the trained emotion model
    model = joblib.load('emotion_model.pkl')

    # Extract features from the image
    features = extract_features(image)
    if features is None:
        return "No face detected"
    
    # Predict the emotion using the model
    emotion = model.predict([features])
    return emotion[0]  # Return the predicted emotion

# Function to train the emotion model (useful for testing)
def train_emotion_model():
    data = []  # Features
    labels = []  # True labels for each image

    # Define the emotion labels (assume folders are named as emotions)
    emotion_labels = ["happy", "sad", "angry", "surprise", "neutral"]
    for emotion in emotion_labels:
        emotion_folder = f'./dataset/{emotion}'  # Path to the training data folder
        for image_name in os.listdir(emotion_folder):
            image_path = os.path.join(emotion_folder, image_name)
            image = cv2.imread(image_path)

            # Extract features from the image
            features = extract_features(image)
            if features:
                data.append(features)
                labels.append(emotion)

    # Convert data to numpy arrays for training
    X_train = np.array(data)
    y_train = np.array(labels)

    # Train a classifier (Support Vector Machine in this case)
    model = SVC(kernel='linear')  # You can experiment with different classifiers
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'emotion_model.pkl')
    print("Emotion model trained and saved!")

# Uncomment this to train the model
train_emotion_model() 