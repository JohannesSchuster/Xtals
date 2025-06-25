import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

class DisplayWidget(QLabel):
    def __init__(self, image_handler, fft: bool = False):
        super().__init__()
        self.image_handler = image_handler
        self.fft = fft
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
        if self.fft: img = self.compute_fft_gpu_torch(img)
        img = self.prepare_image_torch(img)
        h, w, _ = img.shape

        qimg = QImage(img.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.setPixmap(pixmap)
        self.repaint()

    def prepare_image_torch(self, img):
        """
        img: np.ndarray or torch.Tensor, shape (H, W) or (H, W, C)
             C expected to be 1 or 3 channels.

        Returns:
            np.ndarray (H, W, 3) uint8 RGB image ready for display.
        """

        # Convert numpy array to torch tensor if needed
        if isinstance(img, np.ndarray):
            # If shape is (H,W,C), convert to (C,H,W)
            if img.ndim == 3:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            elif img.ndim == 2:
                img_tensor = torch.from_numpy(img).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported image shape {img.shape}")
        elif torch.is_tensor(img):
            img_tensor = img
        else:
            raise TypeError("Input must be numpy.ndarray or torch.Tensor")

        # Ensure float tensor in [0,1]
        img_tensor = img_tensor.float()
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # Convert RGB to grayscale if needed
        if img_tensor.shape[0] == 3:
            img_tensor = rgb_to_grayscale_torch(img_tensor)

        _, h, w = img_tensor.shape
        new_h, new_w = int(h * self.zoom), int(w * self.zoom)

        # Resize using bilinear interpolation
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
        img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        img_resized = img_resized.squeeze(0)

        # Convert to uint8
        img_resized = (img_resized.clamp(0, 1) * 255).byte()

        # Repeat grayscale to RGB
        if img_resized.shape[0] == 1:
            img_rgb = img_resized.repeat(3, 1, 1)
        else:
            img_rgb = img_resized

        # Prepare gray canvas
        canvas_h, canvas_w = self.height(), self.width()
        canvas = torch.full((3, canvas_h, canvas_w), int(0.32 * 255), dtype=torch.uint8)

        # Calculate offsets
        oy = max(0, (canvas_h - new_h) // 2 + self.pan.y())
        ox = max(0, (canvas_w - new_w) // 2 + self.pan.x())

        paste_h = min(new_h, canvas_h - oy)
        paste_w = min(new_w, canvas_w - ox)

        canvas[:, oy:oy + paste_h, ox:ox + paste_w] = img_rgb[:, :paste_h, :paste_w]

        # Convert to numpy HWC
        canvas_np = canvas.permute(1, 2, 0).cpu().numpy()

        return canvas_np
    
#    def prepare_image(self, img):
#        if img.dtype != np.uint8:
#            img = np.clip(img, 0, 1 if img.dtype == np.float32 else 255)
#            img = (img * 255) if img.dtype == np.float32 else img
#            img = img.astype(np.uint8)
#
#        if img.ndim == 2:
#            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#        elif img.shape[2] == 4:
#            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#
#        h, w = img.shape[:2]
#        scaled = cv2.resize(img, (int(w * self.zoom), int(h * self.zoom)), interpolation=cv2.INTER_AREA)
#
#        # Create canvas with background
#        canvas = np.full((self.height(), self.width(), 3), 82, dtype=np.uint8)
#        y, x = scaled.shape[:2]
#        oy = max(0, (canvas.shape[0] - y) // 2 + self.pan.y())
#        ox = max(0, (canvas.shape[1] - x) // 2 + self.pan.x())
#        canvas[oy:oy+y, ox:ox+x] = scaled[:canvas.shape[0]-oy, :canvas.shape[1]-ox]
#        return canvas
