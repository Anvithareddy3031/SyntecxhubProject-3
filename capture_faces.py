import cv2
import os
import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture(person_name, samples=50, cam_index=0):
    # use OpenCV's default haarcascade path
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    data_dir = "data"
    person_dir = os.path.join(data_dir, person_name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(cam_index)
    count = 0
    print(f"[INFO] Capturing {samples} samples for '{person_name}'. Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(path, face_img)

            # draw rectangle and count
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/{samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if count >= samples:
                break

        cv2.imshow("Capture - Press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} images for '{person_name}' into {person_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Person name (folder will be created)")
    parser.add_argument("--samples", type=int, default=50, help="Number of face samples to capture")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    args = parser.parse_args()
    capture(args.name, samples=args.samples, cam_index=args.cam)
