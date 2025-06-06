from FaceMesh.FaceRecognitionModule import FaceMeshDetector
from FaceMesh.DatabaseManager import Database
import cv2
import numpy as np
from datetime import datetime
from FaceMesh.attendance_gui import run_app

def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def normalize_landmarks(landmarks):
    x_vals = [p[0] for p in landmarks]
    y_vals = [p[1] for p in landmarks]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

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
        return False

    known_faces = db.get_all_face_landmarks()
    threshold = 0.8
    matched_enroll = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, faces = detector.faceDetect(frame, draw=False, drawbbox=True)

        if faces:
            norm_landmarks = normalize_landmarks(faces[0])

            for enroll_num, known_embedding in known_faces:
                dist = euclidean_distance(norm_landmarks, known_embedding)
                if dist < threshold:
                    matched_enroll = enroll_num
                    break

            if matched_enroll:
                db.get_student_details(matched_enroll)
                db.show_images_by_enrollment(matched_enroll)
                cap.release()
                cv2.destroyAllWindows()
                return True
        cv2.imshow("Identify & Show", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

def capture_new_student(db, detector, enroll=None, name=None):
    # If name or enroll is None, fallback to input() to keep console backward compatible
    if enroll is None or name is None:
        enroll = input("Enter enrollment number: ")
        name = input("Enter student's name: ")
        branch = input("Enter branch: ")
        year = int(input("Enter year: "))
        email = input("Enter email: ")
    else:
        branch = "N/A"
        year = 1
        email = "N/A"

    db.insert_student_details(enroll, name, branch, year, email)
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
            break
        frame = cv2.flip(frame, 1)
        frame, faces = detector.faceDetect(frame, draw=False, drawbbox=True)

        cropped_face = detector.draw_capture_button_and_capture(frame, faces, click_pos, mouse_pos, button_coords)

        if cropped_face is not None and saved_faces < max_faces_to_save:
            if faces:
                landmarks = faces[0]
                norm_landmarks = normalize_landmarks(landmarks)
                norm_landmarks_str = ','.join(map(str, norm_landmarks.tolist()))
                _, buffer = cv2.imencode('.jpg', cropped_face)
                face_bytes = buffer.tobytes()
                db.insert_face_to_database(enroll, face_bytes, norm_landmarks_str)
                saved_faces += 1
                click_pos = None

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

def mark_attendance(db, detector):
    cap = cv2.VideoCapture(0)
    known_faces = db.get_all_face_landmarks()
    threshold = 0.7
    attendance_marked = set()

    while True:
        ret, frame = cap.read()
        if not ret:
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
                db.mark_attendance(matched_enroll)
                attendance_marked.add(matched_enroll)

        cv2.imshow("Attendance Marking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    db = Database()
    detector = FaceMeshDetector(min_detection_conf=0.3, max_faces=1)

    callbacks = {
        'input_new_student': lambda name, enroll: capture_new_student(db, detector, enroll, name),
        'mark_attendance': lambda: mark_attendance(db, detector),
        'identify_show_data': lambda: identify_and_show_user(db, detector)
    }

    # Run the GUI app (this blocks until exit)
    run_app(db, detector, callbacks)

    db.close()


main()