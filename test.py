import cv2
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from emotion_model import extract_features  # Assuming `extract_features` is a function in emotion_model.py

# Define a function to test the emotion model
def test_emotion_model_on_images(test_folder):
    # Load the trained emotion model
    try:
        model = joblib.load("emotion_model.pkl")
        print("Emotion model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    data = []  # Features
    labels = []  # True labels for each image

    # Define the emotion labels (assume folders are named as emotions)
    emotion_labels = ["happy", "sad", "angry", "surprise", "neutral"]
    
    for emotion in emotion_labels:
        emotion_folder = os.path.join(test_folder, emotion)  # Path to each emotion's folder
        if not os.path.exists(emotion_folder):
            print(f"Warning: Folder for '{emotion}' not found in the test directory.")
            continue
        
        for image_name in os.listdir(emotion_folder):
            image_path = os.path.join(emotion_folder, image_name)
            image = cv2.imread(image_path)

            # Extract features from the image
            features = extract_features(image)
            if features:
                data.append(features)
                labels.append(emotion)

    # Check if there are any test samples
    if len(data) == 0:
        print("No test data available for evaluation.")
        return

    # Convert data to numpy arrays for prediction
    X_test = np.array(data)
    y_test = np.array(labels)

    # Predict emotions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Printing accuracy as a percentage

    # Generate and print the classification report and confusion matrix
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=emotion_labels))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Run the testing function
if __name__ == "__main__":
    test_emotion_model_on_images('./test')  # Provide the path to your test data folder
