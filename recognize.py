import cv2
import os
import pickle
import numpy as np

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("[ERROR] Model or labels not found. Run train_model.py first.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_ids = pickle.load(f)
# invert mapping: id -> name
id_to_name = {v:k for k,v in label_ids.items()}

cap = cv2.VideoCapture(0)
print("[INFO] Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # predict: returns label id and confidence (lower = better)
        label_id, confidence = recognizer.predict(roi_gray)
        # confidence is distance measure; lower = more confident
        if confidence < 80:  # threshold — adjust empirically
            name = id_to_name.get(label_id, "Unknown")
            text = f"{name} ({int(confidence)})"
        else:
            text = f"Unknown ({int(confidence)})"

        # draw
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition - Press q to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
