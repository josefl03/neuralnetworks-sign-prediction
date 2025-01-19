import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget
from PyQt5.QtMultimedia import QCamera, QCameraInfo, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QUrl

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QCamera Frame Capture")

        # Create the main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Camera viewfinder
        self.viewfinder = QCameraViewfinder()
        layout.addWidget(self.viewfinder)

        # Start camera button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)

        # Capture frame button
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.clicked.connect(self.capture_frame)
        layout.addWidget(self.capture_button)

        # Display label for captured images
        self.image_label = QLabel("Captured Frame will appear here")
        layout.addWidget(self.image_label)

        # Initialize camera and image capture objects
        self.camera = None
        self.image_capture = None

    def start_camera(self):
        # Select the default camera
        available_cameras = QCameraInfo.availableCameras()
        if not available_cameras:
            self.image_label.setText("No cameras found.")
            return

        # Initialize the camera
        self.camera = QCamera(available_cameras[0])
        self.camera.setViewfinder(self.viewfinder)

        # Initialize the image capture
        self.image_capture = QCameraImageCapture(self.camera)

        # Start the camera
        self.camera.start()

    def capture_frame(self):
        if self.image_capture is None:
            self.image_label.setText("Camera not started.")
            return

        # Capture the frame
        self.image_capture.imageCaptured.connect(self.display_captured_image)
        self.image_capture.capture()

    def display_captured_image(self, id, image):
        # Convert QImage to QPixmap and display it in the QLabel
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
