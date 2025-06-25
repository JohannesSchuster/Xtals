import torch
import numpy as np
from PyQt5.QtGui import QPixmap, QImage

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
    Convert a torch.Tensor (C,H,W) or (H,W,C) or (H,W) in uint8 or float [0..1]
    to QPixmap.

    Assumes:
    - tensor is on CPU
    - tensor dtype is uint8 or float32/float64
    - tensor channels are last (H,W,C) or first (C,H,W)
    - supports 1 (grayscale) or 3 (RGB) channels
    
    Returns QPixmap ready for Qt display.
    """

    # Move channels last if needed and convert to CPU & numpy
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):  # (C,H,W)
        img = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img = tensor.cpu().numpy()

    # Normalize float image to uint8 if needed
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    if channels == 1:
        # Grayscale image
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif channels == 3:
        # RGB image, QImage expects RGB888
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    return QPixmap.fromImage(qimg)