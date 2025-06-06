from FaceMesh.FaceRecognitionModule import FaceMeshDetector
from FaceMesh.DatabaseManager import Database
import cv2
import numpy as np
from datetime import datetime

def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def normalize_landmarks(landmarks):
    # landmarks: list of (x, y) pixel points
    x_vals = [p[0] for p in landmarks]
    y_vals = [p[1] for p in landmarks]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Normalize to [0,1] based on bounding box
    norm = []
    for (x, y) in landmarks:
        norm_x = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        norm_y = (y - y_min) / (y_max - y_min) if y_max != y_min else 0
        norm.append(norm_x)
        norm.append(norm_y)

    return np.array(norm)

def identify_and_show_user(db, detector):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    known_faces = db.get_all_face_landmarks()
    threshold = 0.8

    for enroll_num, emb in known_faces:
        print(f"{enroll_num}: mean={np.mean(emb):.4f}, std={np.std(emb):.4f}, len={len(emb)}")

    print("[INFO] Showing camera feed. Press 'q' to quit.")
    matched_enroll = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        frame, faces = detector.faceDetect(frame, draw=False, drawbbox=True)

        if faces:
            norm_landmarks = normalize_landmarks(faces[0])  # Just first face

            for enroll_num, known_embedding in known_faces:
                dist = euclidean_distance(norm_landmarks, known_embedding)
                print(f"Checking {enroll_num}: distance = {dist:.4f}")
                if dist < threshold:
                    matched_enroll = enroll_num
                    print(f"[✅ MATCH FOUND] Enrollment Number: {matched_enroll}")
                    break

            if matched_enroll:
                db.get_student_details(matched_enroll)
                db.show_images_by_enrollment(matched_enroll)
                break
        else:
            print("[WARN] No face detected! Ensure good lighting and that your face is visible.")

        cv2.imshow("Identify & Show", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def capture_new_student(db, detector):
    enrollment_number = input("Enter enrollment number: ")
    name = input("Enter student's name: ")
    branch = input("Enter branch: ")
    year = int(input("Enter year: "))
    email = input("Enter email: ")

    db.insert_student_details(enrollment_number, name, branch, year, email)
    print("[INFO] Student details saved successfully!")

    cap = cv2.VideoCapture(0)
    saved_faces = 0
    max_faces_to_save = 5
    button_coords = (20, 20, 170, 70)
    click_pos = None
    mouse_pos = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_pos, mouse_pos
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (x, y)

    cv2.namedWindow("Capture Faces")
    cv2.setMouseCallback("Capture Faces", mouse_callback)

    while saved_faces < max_faces_to_save:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break
        frame = cv2.flip(frame, 1)
        frame, faces = detector.faceDetect(frame, draw=False, drawbbox=True)

        cropped_face = detector.draw_capture_button_and_capture(frame, faces, click_pos, mouse_pos, button_coords)

        if cropped_face is not None and saved_faces < max_faces_to_save:

            if faces:
                landmarks = faces[0]  # list of (x,y)

                norm_landmarks = normalize_landmarks(landmarks)
                norm_landmarks_str = ','.join(map(str, norm_landmarks.tolist()))

                _, buffer = cv2.imencode('.jpg', cropped_face)
                face_bytes = buffer.tobytes()

                # Save both face image bytes and normalized landmarks string
                db.insert_face_to_database(enrollment_number, face_bytes, norm_landmarks_str)

                saved_faces += 1
                print(f"[INFO] Saved face #{saved_faces} for {enrollment_number}")
                click_pos = None

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(db, detector):
    cap = cv2.VideoCapture(0)

    print("[INFO] Loading known face embeddings from database...")

    # Fetch stored normalized landmarks with enrollment numbers
    known_faces = db.get_all_face_landmarks()  # Implement this in your DB class

    threshold = 0.7

    attendance_marked = set()

    print("[INFO] Starting attendance marking. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame, faces = detector.faceDetect(frame, draw=True, drawbbox=True)

        for landmarks in faces:
            norm_landmarks = normalize_landmarks(landmarks)

            min_dist = float('inf')
            matched_enroll = None
            for enroll_num, known_landmarks in known_faces:
                dist = euclidean_distance(norm_landmarks, known_landmarks)
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    matched_enroll = enroll_num

            if matched_enroll and matched_enroll not in attendance_marked:
                print(f"[ATTENDANCE] Marked attendance for {matched_enroll} at {datetime.now()}")
                db.mark_attendance(matched_enroll)  # Implement this method
                attendance_marked.add(matched_enroll)

        cv2.imshow("Attendance Marking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    db = Database()
    detector = FaceMeshDetector(min_detection_conf=0.3, max_faces=1)

    while True:
        print("\n--- Attendance System Menu ---")
        print("1. Input New Student Data")
        print("2. Mark Attendance")
        print("3. Identify & Show My Data")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == '1':
            capture_new_student(db, detector)
        elif choice == '2':
            mark_attendance(db, detector)
        elif choice == '3':
            identify_and_show_user(db, detector)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice, try again.")

    db.close()

if __name__ == "__main__":
    main()
