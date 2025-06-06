import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=2, refine_landmarks=True,
                 min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def faceDetect(self, image, draw=True, drawbbox=True):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        faces = []

        if results.multi_face_landmarks:
            print(f"[DEBUG] Detected {len(results.multi_face_landmarks)} faces")
            for face_landmarks in results.multi_face_landmarks:
                face = []
                x_list = []
                y_list = []
                for lm in face_landmarks.landmark:
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append((cx, cy))
                    x_list.append(cx)
                    y_list.append(cy)

                if drawbbox:
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                          circle_radius=1),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                            circle_radius=1)
                    )

                faces.append(face)
        else:
            print("[DEBUG] No faces detected")

        return image, faces

    def draw_capture_button_and_capture(self, frame, faces, click_pos=None,
                                        mouse_pos=None,
                                        button_coords=(20, 20, 170, 70)):

        def rounded_rectangle(img, pt1, pt2, color, thickness, radius=15):
            # Draws a filled rounded rectangle on img from pt1 to pt2
            x1, y1 = pt1
            x2, y2 = pt2
            if thickness < 0:
                thickness = -1

            # Draw rectangles
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

            # Draw four circles
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

        x1, y1, x2, y2 = button_coords

        # Check if mouse is hovering on button
        hover = False
        if mouse_pos:
            mx, my = mouse_pos
            if x1 <= mx <= x2 and y1 <= my <= y2:
                hover = True

        # Colors
        base_color = (0, 150, 0)  # dark green
        hover_color = (0, 255, 0)  # bright green
        text_color = (255, 255, 255)  # white

        btn_color = hover_color if hover else base_color

        # Draw rounded rectangle button
        rounded_rectangle(frame, (x1, y1), (x2, y2), btn_color, thickness=-1, radius=20)

        # Put centered text
        text = "CAPTURE"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x1 + (x2 - x1 - text_w) // 2
        text_y = y1 + (y2 - y1 + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Handle click
        if click_pos:
            cx, cy = click_pos
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if not faces:
                    print("[⚠] No face detected to capture!")
                    return None

                face_landmarks = faces[0]
                x_vals = [pt[0] for pt in face_landmarks]
                y_vals = [pt[1] for pt in face_landmarks]

                h, w, _ = frame.shape
                x_min, x_max = max(0, min(x_vals)), min(w, max(x_vals))
                y_min, y_max = max(0, min(y_vals)), min(h, max(y_vals))

                cropped_face = frame[y_min:y_max, x_min:x_max]
                return cropped_face

        return None


    def match_landmarks(test_landmarks, known_landmarks_list, threshold=0.1):
        test_vec = np.array(test_landmarks)
        min_dist = float('inf')
        matched_enrollment = None

        for enrollment_number, known_vec in known_landmarks_list:
            known_vec = np.array(known_vec)
            dist = np.linalg.norm(test_vec - known_vec)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                matched_enrollment = enrollment_number

        return matched_enrollment

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
