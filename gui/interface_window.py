from PyQt5.QtWidgets import QMainWindow, QFileDialog, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt

from .display import DisplayWindow

from handlers import FileFramesHandler

class InterfaceWindow(QMainWindow):
    def __init__(self, filepath=None):
        super().__init__()
        self.setWindowTitle("PyQt Image Viewer")

        if filepath is None:
            filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        
        self.image_handler = FileFramesHandler()
        self.image_handler.load(filepath)

        self.display = DisplayWindow(self.image_handler)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.image_handler.frame_count - 1)
        self.slider.valueChanged.connect(self.change_frame)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.display.show()

    def change_frame(self, value):
        self.image_handler.set_frame(value)
        self.display.update_display()
