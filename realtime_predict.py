# realtime_predict.py
import cv2, joblib, threading
import mediapipe as mp
import numpy as np
from collections import deque
import pyttsx3

MODEL_PATH = "gesture_model.joblib"
LE_PATH = "label_encoder.joblib"

model = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

engine = pyttsx3.init()
def speak(text):
    def _s():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_s, daemon=True).start()

def normalize_landmarks(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    max_d = 1e-6
    for lm in landmarks:
        dx = lm.x - base_x
        dy = lm.y - base_y
        d = (dx*dx + dy*dy)**0.5
        if d > max_d: max_d = d
    feats = []
    for lm in landmarks:
        feats.append((lm.x - base_x) / max_d)
        feats.append((lm.y - base_y) / max_d)
    return np.array(feats).reshape(1, -1)

cap = cv2.VideoCapture(0)
history = deque(maxlen=9)  # smoothing window
last_spoken = None
SPEAK_ON_CHANGE = True

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        predicted_label = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)
            feats = normalize_landmarks(hand.landmark)
            # model might be a pipeline; call predict
            pred_idx = model.predict(feats)[0]
            label = le.inverse_transform([pred_idx])[0]
            history.append(label)
            # majority vote
            if len(history) == history.maxlen:
                maj = max(set(history), key=history.count)
            else:
                maj = max(set(history), key=history.count)
            predicted_label = maj
            cv2.putText(display, f"Pred: {predicted_label}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.putText(display, "No hand detected", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Realtime Gesture Prediction", display)
        # speak if changed
        if predicted_label is not None and predicted_label != last_spoken:
            if SPEAK_ON_CHANGE:
                speak(predicted_label)
            last_spoken = predicted_label

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
