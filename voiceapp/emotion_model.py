# from keras.models import load_model
# import numpy as np
# import librosa

# model = load_model("C:/Users/Tejas V/emotion_model.keras")
# labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
#   # adjust based on your classes

# import numpy as np
# import librosa

# def extract_features(file_path):
#     audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     mfccs_processed = np.mean(mfccs.T, axis=0)  # shape will be (40,)
#     return mfccs_processed


# def predict_emotion(file_path):
#     features = extract_features(file_path)
#     features = features.reshape(1, 40)  # reshape to match input shape (None, 40)
#     prediction = model.predict(features)
#     emotion = labels[np.argmax(prediction)]
#     return emotion

import numpy as np
import librosa
import tensorflow as tf

# Load your saved model (update the path if needed)
model = tf.keras.models.load_model("C:/Users/Tejas V/emotion_model.keras")

# Define the emotion labels (should match your model's output order)
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Shape: (40,)
    return mfccs_processed

# Prediction function
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.reshape(features, (1, -1))  # Shape: (1, 40)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    return labels[predicted_index]
