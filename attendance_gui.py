# attendance_gui.py
import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QMessageBox, QLineEdit, QFormLayout, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt

class InputStudentDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Input New Student Data")
        self.student_data = None

        self.layout = QFormLayout(self)

        self.name_edit = QLineEdit(self)
        self.enroll_edit = QLineEdit(self)

        self.layout.addRow("Name:", self.name_edit)
        self.layout.addRow("Enrollment No:", self.enroll_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def get_data(self):
        if self.exec_() == QDialog.Accepted:
            name = self.name_edit.text().strip()
            enroll = self.enroll_edit.text().strip()
            if name and enroll:
                return name, enroll
        return None, None


class AttendanceGUI(QWidget):
    def __init__(self, db, detector, callbacks):
        """
        db: Database instance
        detector: FaceMeshDetector instance
        callbacks: dict of functions:
            {
                'input_new_student': func,
                'mark_attendance': func,
                'identify_show_data': func,
            }
        """
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.db = db
        self.detector = detector
        self.callbacks = callbacks
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Select an option:")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.btn_input = QPushButton("Input New Student Data")
        self.btn_input.clicked.connect(self.handle_input_new_student)
        layout.addWidget(self.btn_input)

        self.btn_mark = QPushButton("Mark Attendance")
        self.btn_mark.clicked.connect(self.handle_mark_attendance)
        layout.addWidget(self.btn_mark)

        self.btn_identify = QPushButton("Identify & Show My Data")
        self.btn_identify.clicked.connect(self.handle_identify_show_data)
        layout.addWidget(self.btn_identify)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close_app)
        layout.addWidget(self.btn_exit)

        self.setLayout(layout)

    def set_buttons_enabled(self, enabled: bool):
        self.btn_input.setEnabled(enabled)
        self.btn_mark.setEnabled(enabled)
        self.btn_identify.setEnabled(enabled)
        self.btn_exit.setEnabled(enabled)

    def handle_input_new_student(self):
        dialog = InputStudentDialog()
        name, enroll = dialog.get_data()
        if not name or not enroll:
            self.show_message("Please enter valid Name and Enrollment No.")
            return

        def task():
            try:
                self.set_label("Starting new student data capture...")
                self.set_buttons_enabled(False)
                success = self.callbacks['input_new_student'](name, enroll)
                if success:
                    self.show_message(f"Student {name} enrolled successfully!")
                else:
                    self.show_message("Failed to enroll student. Try again.")
            except Exception as e:
                self.show_message(f"Error: {str(e)}")
            finally:
                self.set_label("Select an option:")
                self.set_buttons_enabled(True)

        threading.Thread(target=task, daemon=True).start()

    def handle_mark_attendance(self):
        def task():
            try:
                self.set_label("Marking attendance...")
                self.set_buttons_enabled(False)
                self.callbacks['mark_attendance']()
                self.show_message("Attendance marked!")
            except Exception as e:
                self.show_message(f"Error: {str(e)}")
            finally:
                self.set_label("Select an option:")
                self.set_buttons_enabled(True)
        threading.Thread(target=task, daemon=True).start()

    def handle_identify_show_data(self):
        def task():
            try:
                self.set_label("Identifying user...")
                self.set_buttons_enabled(False)
                identified = self.callbacks['identify_show_data']()
                if not identified:
                    self.show_message("User not recognized.")
            except Exception as e:
                self.show_message(f"Error: {str(e)}")
            finally:
                self.set_label("Select an option:")
                self.set_buttons_enabled(True)
        threading.Thread(target=task, daemon=True).start()

    def show_message(self, message):
        QMessageBox.information(self, "Info", message)

    def set_label(self, text):
        self.label.setText(text)

    def close_app(self):
        self.db.close()
        self.close()


def run_app(db, detector, callbacks):
    app = QApplication(sys.argv)
    window = AttendanceGUI(db, detector, callbacks)
    window.show()
    sys.exit(app.exec_())
