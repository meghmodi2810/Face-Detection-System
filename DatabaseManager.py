import mysql.connector
import cv2
import numpy as np
import json

class Database:


    def __init__(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="6640",
            database="attendance_system"
        )
        self.cursor = self.conn.cursor(buffered=True)

    def get_all_face_landmarks(self):
        sql = "SELECT enrollment_number, face_embedding FROM face_images"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        data = []
        for enroll_num, embedding_str in rows:
            if embedding_str:
                embedding = np.array(list(map(float, embedding_str.split(','))))
                data.append((enroll_num, embedding))
        return data

    def insert_student_details(self, enrollment_number, name, branch, year, email):
        sql = """
            INSERT INTO student_details (enrollment_number, name, branch, year, email)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                name = VALUES(name),
                branch = VALUES(branch),
                year = VALUES(year),
                email = VALUES(email)
        """
        self.cursor.execute(sql, (enrollment_number, name, branch, year, email))
        self.conn.commit()

    def show_images_by_enrollment(self, enrollment_number):
        sql = "SELECT face_image FROM face_images WHERE enrollment_number = %s"
        self.cursor.execute(sql, (enrollment_number,))
        rows = self.cursor.fetchall()

        if not rows:
            print(f"No images found for {enrollment_number}")
            return

        for idx, (face_blob,) in enumerate(rows):
            # Convert bytes to numpy array
            nparr = np.frombuffer(face_blob, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to decode image #{idx + 1}")
                continue

            cv2.imshow(f"{enrollment_number} - Image {idx + 1}", img)
            cv2.waitKey(0)
            cv2.destroyWindow(f"{enrollment_number} - Image {idx + 1}")

    def insert_face_to_database(self, enrollment_number, face_image_bytes, face_embedding_str):
        sql = """
            INSERT INTO face_images (enrollment_number, face_image, face_embedding)
            VALUES (%s, %s, %s)
        """
        self.cursor.execute(sql, (enrollment_number, face_image_bytes, face_embedding_str))
        self.conn.commit()

    def get_student_details(self, enrollment_number):
        sql = """
            SELECT sd.enrollment_number, sd.name, sd.branch, sd.year, sd.email
            FROM student_details sd
            JOIN face_images fi ON sd.enrollment_number = fi.enrollment_number
            WHERE sd.enrollment_number = %s
        """
        self.cursor.execute(sql, (enrollment_number,))
        result = self.cursor.fetchone()
        if result:
            print(f"Enrollment: {result[0]}")
            print(f"Name      : {result[1]}")
            print(f"Branch    : {result[2]}")
            print(f"Year      : {result[3]}")
            print(f"Email     : {result[4]}")
        else:
            print(f"No student details found for {enrollment_number}")

    def get_all_face_images(self):
        """Return list of (enrollment_number, face_image_bytes)"""
        sql = "SELECT enrollment_number, face_image FROM face_images"
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def mark_attendance(self, enrollment_number):
        sql_check = """
            SELECT * FROM attendance
            WHERE enrollment_number = %s AND DATE(timestamp) = CURDATE()
        """
        self.cursor.execute(sql_check, (enrollment_number,))
        if self.cursor.fetchone():
            # Already marked today
            return False

        sql_insert = "INSERT INTO attendance (enrollment_number) VALUES (%s)"
        self.cursor.execute(sql_insert, (enrollment_number,))
        self.conn.commit()
        return True

    def insert_landmarks(self, enrollment_number, landmarks):
        landmarks_json = json.dumps(landmarks)
        sql = "INSERT INTO face_landmarks (enrollment_number, landmarks) VALUES (%s, %s)"
        self.cursor.execute(sql, (enrollment_number, landmarks_json))
        self.conn.commit()

    def get_all_landmarks(self):
        sql = "SELECT enrollment_number, landmarks FROM face_landmarks"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        data = []
        for enrollment_number, landmarks_json in rows:
            landmarks = json.loads(landmarks_json)
            data.append((enrollment_number, landmarks))
        return data


    def close(self):
        self.cursor.close()
        self.conn.close()




'''
def main():
    db = Database()
    db.get_student_details(202307100110146)
    db.show_images_by_enrollment(202307100110146)
    db.close()

main()
'''