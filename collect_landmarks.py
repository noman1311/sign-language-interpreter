# collect_landmarks.py
import cv2, os, csv, time
import mediapipe as mp

OUTPUT_CSV = "landmarks.csv"
FRAMES_PER_LABEL = 150  # how many samples per key press

# map keys to letters only
KEY_TO_LABEL = {
    ord('a'): "A",
    ord('b'): "B",
    ord('c'): "C",
    ord('d'): "D",
    ord('e'): "E"
    # 👉 you can add more letters here as you go
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def normalize_landmarks(landmarks):
    # landmarks: list of 21 hand landmarks
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    # compute scale = max distance from wrist
    max_d = 1e-6
    for lm in landmarks:
        dx = lm.x - base_x
        dy = lm.y - base_y
        d = (dx*dx + dy*dy)**0.5
        if d > max_d:
            max_d = d
    features = []
    for lm in landmarks:
        features.append((lm.x - base_x) / max_d)
        features.append((lm.y - base_y) / max_d)
    return features

# prepare CSV header
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}"]
        header += ["label"]
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print("Open webcam. Press a/b/c/d/e... to record samples for each letter. ESC to quit.")
print(f"Each key press records {FRAMES_PER_LABEL} valid frames (hand must be detected).")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(display, "Press a/b/c/d/e... to record. ESC to exit.",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Collect Landmarks", display)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k in KEY_TO_LABEL:
            label = KEY_TO_LABEL[k]
            print(f"Recording {FRAMES_PER_LABEL} frames for label '{label}' ...")
            collected = 0
            start = time.time()
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                while collected < FRAMES_PER_LABEL:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    display = frame.copy()
                    if res.multi_hand_landmarks:
                        hand_landmarks = res.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        features = normalize_landmarks(hand_landmarks.landmark)
                        writer.writerow(features + [label])
                        collected += 1
                        cv2.putText(display, f"{label} : {collected}/{FRAMES_PER_LABEL}",
                                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    else:
                        cv2.putText(display, "No hand detected - hold steady",
                                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    cv2.imshow("Collect Landmarks", display)
                    if cv2.waitKey(1) & 0xFF == 27:
                        collected = FRAMES_PER_LABEL
                        break
            dur = time.time() - start
            print(f"Finished recording {label} in {dur:.1f}s.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
