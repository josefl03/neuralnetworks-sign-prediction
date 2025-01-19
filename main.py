import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.uic import loadUi
from PyQt5.QtMultimedia import QCameraInfo, QCamera
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
import logging
from PyQt5.QtCore import QThread
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtMultimedia import QCameraImageCapture
from PyQt5.QtGui import QPixmap
import numpy as np
from PyQt5.QtCore import QTimer

# Categories mapping
categories = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "G", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"
}

TIMER_INTERVAL = 1500

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("layout.ui", self)

        # ------------------------ MODEL ------------------------
        v2_base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

        def resnet_with_cnn(base_model, cnn_layers=2):
            base_model.trainable = False # we are freezing the base_model
            inputs = Input(shape=(64, 64, 3))
            x = base_model(inputs, training=False)
            for _ in range(cnn_layers - 1):
                x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = MaxPooling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = GlobalAveragePooling2D()(x)
            outputs = Dense(29, activation='softmax')(x)  # Updated for 29 classes
            model = Model(inputs, outputs)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        self.model = resnet_with_cnn(v2_base_model)
        self.model.load_weights("r50v2_cnn.weights.h5")
        # ------------------------       ------------------------

        # Variables
        self.predicting = False

        # Connect buttons
        self.startButton.clicked.connect(self.start_predicting)

        self.cameraCombobox.currentIndexChanged.connect(self.change_camera)

        # Get a list of the connected cameras
        self.cameras = QCameraInfo.availableCameras()

        logger.info("Cameras found:")
        for camera in self.cameras:
            logger.info(f"- '{camera.deviceName()}': {camera.description()}")

            self.cameraCombobox.addItem(f"{camera.description()} @ {camera.deviceName()}")

        # Show the main camera
        self.cameraViewFinder = QCameraViewfinder(self)
        self.findChild(QWidget, "cameraWidget").layout().addWidget(self.cameraViewFinder)

        self.camera = QCamera(self.cameras[0])
        self.camera.setViewfinder(self.cameraViewFinder)

        # Setup the image capture
        self.imageCapture = QCameraImageCapture(self.camera)

        self.camera.start()

        # Set up the timer
        self.imageCapture.imageCaptured.connect(self.image_captured)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_image)  # Connect timeout to update_counter

    def change_camera(self, index):
        if not hasattr(self, "camera"):
            return 
        
        # Stop the current camera
        self.camera.stop()

        # Start the new camera
        self.camera = QCamera(self.cameras[index])
        self.camera.setViewfinder(self.cameraViewFinder)
        self.camera.start()

    def capture_image(self):
        self.imageCapture.capture()
        logger.info("Captured.")

    def image_captured(self, id, image: QImage):
        logger.info("Image received")
        
        # Get the dimensions of the original image
        width = image.width()
        height = image.height()

        # Determine the size of the square (smallest dimension)
        square_size = min(width, height)

        # Calculate the top-left point to center the crop
        x_offset = (width - square_size) // 2
        y_offset = (height - square_size) // 2

        # Crop the image to a centered square
        cropped_image = image.copy(x_offset, y_offset, square_size, square_size)

        self.predict_sign(cropped_image)
        self.set_preview(cropped_image)

    def set_preview(self, cropped_image): 
        # Scale the cropped image to 80x80 pixels and 64x64 pixels
        image80 = cropped_image.scaled(80, 80)

        pixmap = QPixmap.fromImage(image80)
        self.previewLabel.setPixmap(pixmap)
        self.previewLabel.setScaledContents(True)

    def predict_sign(self, cropped_image):
        logger.info("Predicting...")

        image64 = cropped_image.scaled(64, 64)

        # Convert QImage to bytes and reshape it to (64, 64, 3)
        image64 = image64.convertToFormat(QImage.Format_RGB888)
        width = image64.width()
        height = image64.height()

        # Extract pixel data from QImage
        ptr = image64.bits()
        ptr.setsize(width * height * 3)
        img_array = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

        # Normalize the pixel values to range [0, 1]
        img_array = img_array.astype('float32') / 255.0
        exp_img = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = self.model.predict(exp_img)

        # Extract probabilities from the prediction (1D array for the batch)
        probabilities = predictions[0]

        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]
        predicted_label = categories[predicted_class_index]

        logger.info(f"Predicted sign: '{predicted_label}' with probability {predicted_probability}")
        
        # Set the sign (markdown)
        self.signLabel.setText(f'"**{predicted_label}**"')
        self.probLabel.setText(f"{int(predicted_probability*100):}%")

        # Map probabilities to their corresponding labels
        category_probabilities = {categories[i]: prob for i, prob in enumerate(probabilities)}

        # Sort the probabilities
        sorted_probabilities = sorted(category_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Get the top 10 probabilities
        top_10_probabilities = sorted_probabilities[:10]

        # Create a string with the top 10 probabilities
        probabilitiesText = ""
        for i, (label, prob) in enumerate(top_10_probabilities):
            probabilitiesText += f"{i+1}. **{label}**: {int(prob*100)}%\n"

        self.probabilitiesTextbox.setMarkdown(probabilitiesText)


    def start_predicting(self):
        if not self.predicting:
            self.predicting = True

            self.startButton.setText("Stop")
            self.cameraCombobox.setEnabled(False)

            self.timer.start(TIMER_INTERVAL)
        else:
            self.predicting = False

            self.startButton.setText("Start")
            self.cameraCombobox.setEnabled(True)

            self.timer.stop()

class PredictWorker(QThread):
    requestSignal = pyqtSignal()

    def __init__(self, parent=None,
                 q=None):
        super(PredictWorker, self).__init__(parent)

        self.q = q

    def run(self):
        while True:
            # Request an image
            #self.requestSignal.emit()

            # Obtain the image
            image = self.q.get()

            logger.info("Received image:", image)

            logger.info("Predicting...")
            
            # Get the image from the camera
            image = self.cameraViewFinder.grab()

            # Resize the image to 64x64
            image = image.scaled(64, 64)

            self.sleep(1)


if __name__ == "__main__":
    # Setup logger
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Start application
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
