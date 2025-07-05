from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QResizeEvent, QMouseEvent, QWheelEvent, QPixmap

from renderer import Renderer, RenderQueue
import renderer.jobs as rj

from tiffhandler import FrameHandler

from util.qt_torch_glue import tensor_to_qpixmap


class DisplayWidget(QLabel):
    def __init__(self, image_handler: FrameHandler, renderer: Renderer):
        super().__init__()
        self.image_handler = image_handler
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.dragging = False
        self.last_pos = QPoint()
        self.background_color = (82, 82, 82)
        self.renderer = renderer
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(1, 1)
        self.setMouseTracking(True)
        #self.update_display()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y() // 120
            self.image_handler.set_frame(self.image_handler.current_frame + delta)
            self.update_display()
        else:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            new_zoom = self.zoom * factor

            img = self.image_handler.get_frame()
            c, h, w = img.shape
            disp_w, disp_h = self.width(), self.height()

            # Prevent zooming in beyond minimum crop area (10x10 pixels)
            min_zoom = max(disp_w / w, disp_h / h, 10 / w, 10 / h)
            # Prevent zooming out so that visible region becomes smaller than 10x10
            max_zoom = min(w / 10, h / 10)

            # Clamp zoom
            self.zoom = max(min_zoom, min(max_zoom, new_zoom))
            self.update_display()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            delta = event.pos() - self.last_pos
            self.pan += delta
            self.last_pos = event.pos()
            self.update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.dragging = False

    def resizeEvent(self, event: QResizeEvent):
        self.clear()
        new_size = event.size()
        new_width = new_size.width()
        new_height = new_size.height()
        self.update_display(new_size=(new_width, new_height))

    def update_display(self, new_size: tuple[int, int] | None = None):
        """
        Updates the QLabel pixmap using torch.Tensor image rendering on the GPU.
        Applies zoom, pan, FFT (if enabled), and embeds into canvas.
        """
        img = self.image_handler.get_frame()
        c, h, w = img.shape
        if new_size:
            disp_w, disp_h = new_size
        else:
            disp_w, disp_h = self.width(), self.height()

        # Determine crop size in image space
        crop_w = int(disp_w / self.zoom)
        crop_h = int(disp_h / self.zoom)

        # Pan is in screen space; determine top-left in image coords
        x = int(self.pan.x() / self.zoom)
        y = int(self.pan.y() / self.zoom)

        # Clamp crop origin
        x = max(0, min(w - crop_w, x))
        y = max(0, min(h - crop_h, y))

        # Compute offset on canvas
        embed_ox = int(self.pan.x())
        embed_oy = int(self.pan.y())

        # Build queue
        queue: RenderQueue = self.renderer.begin()
        img_upload = queue.submit(rj.Upload(slot=0, img=img))
        canvas_create = queue.submit(rj.CreateFlat(slot=1, size=(disp_w, disp_h), color=(42, 42, 42)))
        img_crop = queue.submit(rj.Crop(offset=(x, y), size=(crop_w, crop_h)), wait_for=[img_upload])
        img_resize = queue.submit(rj.Resize(new_size=(disp_w, disp_h), mode="bilinear"), wait_for=[img_crop])
        blend = queue.submit(rj.Blend(slots=[1, 0], offsets=[(0,0), (embed_ox, embed_oy)], mode="override"), wait_for=[canvas_create, img_resize])
        output = self.renderer.render(queue, output=[blend])[blend]


        # Convert back to QPixmap and display
        pixmap: QPixmap = tensor_to_qpixmap(output)
        
        self.setPixmap(pixmap)
        #self.repaint()

class DisplayWindow(QWidget):
    def __init__(self, image_handler: BaseFramesHandler):
        super().__init__()
        self.setWindowTitle(f"Display - {image_handler.id}")
        self.display_widget = DisplayWidget(image_handler)
        
        layout = QVBoxLayout()
        layout.addWidget(self.display_widget)
        self.setLayout(layout)
        self.resize(800, 600)

    def update_display(self):
        self.display_widget.update_display()