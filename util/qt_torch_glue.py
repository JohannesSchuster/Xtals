import torch
import numpy as np
from util.np_torch_glue import tensor_to_np
from PyQt5.QtGui import QPixmap, QImage

__all__ = ['tensor_to_qpixmap', 'qpixmap_to_tensor']

def qpixmap_to_tensor(pixmap: QPixmap, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Converts a QPixmap to a torch.Tensor (C, H, W) in uint8 on specified device.

    Args:
        pixmap (QPixmap): The input pixmap.
        device (torch.device): Target device for the tensor.

    Returns:
        torch.Tensor: (3, H, W) uint8 RGB image tensor.
    """
    if pixmap.isNull():
        raise ValueError("Received null QPixmap.")

    # Convert QPixmap to QImage (32-bit RGB)
    qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    width = qimage.width()
    height = qimage.height()

    # Extract raw data and reshape into NumPy array
    ptr = qimage.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))

    # Convert to torch tensor and permute to (C, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return tensor


def tensor_to_qpixmap(tensor: torch.Tensor) -> QPixmap:
    """
    Convert a torch.Tensor (C, H, W) or (1, H, W) in uint8 or float [0..1]
    to QPixmap.

    Assumes:
    - tensor dtype is uint8 or float32/float64
    - tensor channels are first (C,H,W) or (1,H,W)
    - supports 1 (grayscale) or 3 (RGB) channels
    
    Returns QPixmap ready for Qt display.
    """

    img: np.ndarray = tensor_to_np(tensor).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    if channels == 1:
        # Grayscale image
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    elif channels == 3:
        # RGB image, QImage expects RGB888
        qimg = QImage(img.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    return QPixmap.fromImage(qimg)