import os
import cv2
import numpy as np
import pickle

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_images_and_labels(data_dir):
    images = []
    labels = []
    label_ids = {}
    current_id = 0

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in label_ids:
            label_ids[person_name] = current_id
            current_id += 1
        label_id = label_ids[person_name]

        for filename in os.listdir(person_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(person_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                images.append(image)
                labels.append(label_id)

    return images, labels, label_ids

if __name__ == "__main__":
    ensure_dir(MODEL_DIR)
    print("[INFO] Loading images...")
    images, labels, label_ids = load_images_and_labels(DATA_DIR)
    if len(images) == 0:
        print("[ERROR] No images found in data/. Run capture_faces.py first.")
        exit(1)

    print(f"[INFO] {len(images)} images found. Training LBPH recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.save(MODEL_PATH)

    # save label mapping
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_ids, f)

    print(f"[INFO] Training complete. Model saved to {MODEL_PATH}")
    print(f"[INFO] Labels saved to {LABELS_PATH}")
    print(f"[INFO] Label mapping: {label_ids}")
