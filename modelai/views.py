import base64
import cv2
import numpy as np
import mediapipe as mp
import math
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (0, 255, 0)

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    angle = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

def is_thumbs_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    thumb_tip = hand_landmarks.landmark[tips[0]]
    index_tip = hand_landmarks.landmark[tips[1]]
    middle_tip = hand_landmarks.landmark[tips[2]]
    ring_tip = hand_landmarks.landmark[tips[3]]
    pinky_tip = hand_landmarks.landmark[tips[4]]

    index_folded = index_tip.y > hand_landmarks.landmark[tips[1]-2].y
    middle_folded = middle_tip.y > hand_landmarks.landmark[tips[2]-2].y
    ring_folded = ring_tip.y > hand_landmarks.landmark[tips[3]-2].y
    pinky_folded = pinky_tip.y > hand_landmarks.landmark[tips[4]-2].y

    thumb_up = thumb_tip.y < hand_landmarks.landmark[tips[0]-2].y

    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

def count_raised_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    count = 0
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            count += 1
    return count

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def get_frame(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            img_data = data.get('image')
            if not img_data:
                return JsonResponse({'error': 'No image data provided'}, status=400)

            # Decode base64 image
            header, encoded = img_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return JsonResponse({'error': 'Invalid image data'}, status=400)

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            messages = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
                 mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:

                results = holistic.process(rgb_image)
                hand_results = hands.process(rgb_image)

                left_thumb = right_thumb = False
                finger_count = -1

                # Check hand gestures
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        if is_thumbs_up(hand_landmarks):
                            if hand_label == "Left":
                                left_thumb = True
                            elif hand_label == "Right":
                                right_thumb = True

                        if finger_count == -1:
                            finger_count = count_raised_fingers(hand_landmarks)

                # Check for menu trigger using both thumbs up
                mode = data.get('mode', 'menu')
                squat_count = data.get('squat_count', 0)
                squat_stage = data.get('squat_stage', None)
                last_finger_trigger = data.get('last_finger_trigger', -1)
                both_thumbs_up_detected = data.get('both_thumbs_up_detected', False)

                if left_thumb and right_thumb and mode != "menu":
                    mode = "menu"
                    squat_count = 0
                    squat_stage = None
                    both_thumbs_up_detected = True
                elif not (left_thumb and right_thumb):
                    both_thumbs_up_detected = False

                # Handle finger-based mode selection
                if finger_count != -1 and finger_count != last_finger_trigger:
                    last_finger_trigger = finger_count
                    if mode == "menu":
                        if finger_count == 1:
                            mode = "body"
                        elif finger_count == 2:
                            mode = "squat"

                # BODY MOVEMENTS
                if mode == "body" and results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    if lm[mp_holistic.PoseLandmark.LEFT_WRIST].y < lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].y:
                        messages.append("Right Hand Raised")
                    if lm[mp_holistic.PoseLandmark.RIGHT_WRIST].y < lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y:
                        messages.append("Left Hand Raised")

                    # Tilt detection
                    lsy = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].y
                    rsy = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y
                    if abs(lsy - rsy) > 0.05:
                        messages.append("Body Tilted " + ("Left" if lsy < rsy else "Right"))

                    ley = lm[mp_holistic.PoseLandmark.LEFT_EAR].y
                    rey = lm[mp_holistic.PoseLandmark.RIGHT_EAR].y
                    if abs(ley - rey) > 0.03:
                        messages.append("Head Tilted " + ("Left" if ley < rey else "right"))

                    left_leg_angle = calculate_angle(
                        lm[mp_holistic.PoseLandmark.LEFT_HIP],
                        lm[mp_holistic.PoseLandmark.LEFT_KNEE],
                        lm[mp_holistic.PoseLandmark.LEFT_ANKLE]
                    )
                    right_leg_angle = calculate_angle(
                        lm[mp_holistic.PoseLandmark.RIGHT_HIP],
                        lm[mp_holistic.PoseLandmark.RIGHT_KNEE],
                        lm[mp_holistic.PoseLandmark.RIGHT_ANKLE]
                    )
                    if left_leg_angle < 160:
                        messages.append("Right Leg Raised")
                    if right_leg_angle < 160:
                        messages.append("Left Leg Raised")

                # SQUAT MODE
                elif mode == "squat" and results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    left_angle = calculate_angle(
                        lm[mp_holistic.PoseLandmark.LEFT_HIP],
                        lm[mp_holistic.PoseLandmark.LEFT_KNEE],
                        lm[mp_holistic.PoseLandmark.LEFT_ANKLE]
                    )
                    right_angle = calculate_angle(
                        lm[mp_holistic.PoseLandmark.RIGHT_HIP],
                        lm[mp_holistic.PoseLandmark.RIGHT_KNEE],
                        lm[mp_holistic.PoseLandmark.RIGHT_ANKLE]
                    )
                    avg_angle = (left_angle + right_angle) / 2

                    if avg_angle < 90:
                        squat_stage = "down"
                    if avg_angle > 160 and squat_stage == "down":
                        squat_stage = "up"
                        squat_count += 1

                    messages.append("You can do it!")
                    messages.append(f"Squat Count: {squat_count}")

                return JsonResponse({
                    'messages': messages,
                    'mode': mode,
                    'squat_count': squat_count,
                    'squat_stage': squat_stage,
                    'last_finger_trigger': last_finger_trigger,
                    'both_thumbs_up_detected': both_thumbs_up_detected,
                })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    elif request.method == 'GET':
        return JsonResponse({'message': 'GET method is supported for this endpoint.'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
