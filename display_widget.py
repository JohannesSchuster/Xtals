import numpy as np
import cv2
from PyQt5.QtWidgets import QLabel, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

class DisplayWidget(QLabel):
    def __init__(self, image_handler):
        super().__init__()
        self.image_handler = image_handler
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.dragging = False
        self.last_pos = QPoint()
        self.setStyleSheet("background-color: rgb(82, 82, 82);")
        self.setScaledContents(False)
        self.setMouseTracking(True)
        self.update_display()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y() // 120
            self.image_handler.set_frame(self.image_handler.current_frame + delta)
            self.update_display()
        else:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.zoom *= factor
            self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.last_pos
            self.pan += delta
            self.last_pos = event.pos()
            self.update_display()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def resizeEvent(self, event):
        self.update_display()

    def update_display(self):
        img = self.image_handler.get_frame()
        img = self.prepare_image(img)
        h, w, _ = img.shape

        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.setPixmap(pixmap)
        self.repaint()

    def prepare_image(self, img):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1 if img.dtype == np.float32 else 255)
            img = (img * 255) if img.dtype == np.float32 else img
            img = img.astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w = img.shape[:2]
        scaled = cv2.resize(img, (int(w * self.zoom), int(h * self.zoom)), interpolation=cv2.INTER_AREA)

        # Create canvas with background
        canvas = np.full((self.height(), self.width(), 3), 82, dtype=np.uint8)
        y, x = scaled.shape[:2]
        oy = max(0, (canvas.shape[0] - y) // 2 + self.pan.y())
        ox = max(0, (canvas.shape[1] - x) // 2 + self.pan.x())
        canvas[oy:oy+y, ox:ox+x] = scaled[:canvas.shape[0]-oy, :canvas.shape[1]-ox]
        return canvas
