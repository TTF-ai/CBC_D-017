from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import io
import numpy as np
from PIL import Image
import librosa
import tensorflow as tf
import cv2
import tempfile
import os

# Load models once when the server starts
# Audio model
audio_model = tf.keras.models.load_model("C:/Users/Tejas V/emotion_model.keras")
audio_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Video model
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
video_model = tf.keras.models.load_model("C:/Users/Tejas V/emotiondetector.h5")  # Assuming you've saved as .h5
video_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

@csrf_exempt
def emotion_detection(request):
    if request.method == 'GET':
        # Serve the HTML page
        return render(request, 'emotion_center.html')
    
    elif request.method == 'POST':
        # Handle both audio and video predictions
        if 'audio' in request.FILES:
            # Audio processing
            try:
                audio_file = request.FILES['audio']
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    for chunk in audio_file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name

                # Extract features
                audio, sample_rate = librosa.load(tmp_path, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfccs_processed = np.mean(mfccs.T, axis=0)
                features = np.reshape(mfccs_processed, (1, -1))
                
                # Predict
                prediction = audio_model.predict(features)
                emotion = audio_labels[np.argmax(prediction)]
                os.unlink(tmp_path)
                return JsonResponse({'emotion': emotion})
            
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

        elif 'image' in request.POST:
            # Video/image processing
            try:
                data_uri = request.POST['image']
                header, encoded = data_uri.split(",", 1)
                image_data = base64.b64decode(encoded)
                
                # Convert to numpy array
                image = Image.open(io.BytesIO(image_data))
                image_np = np.array(image.convert('RGB'))
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    return JsonResponse({'error': 'No faces detected'}, status=400)

                # Process first face
                x, y, w, h = faces[0]
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img = roi_gray.reshape(1, 48, 48, 1) / 255.0
                predictions = video_model.predict(img)
                emotion = video_labels[np.argmax(predictions)]
                
                return JsonResponse({'emotion': emotion})
            
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)