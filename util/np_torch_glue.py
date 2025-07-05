import numpy as np
import torch

def np_to_tensor(img: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a np.ndarray.")

    if img.ndim == 3:
        img_tensor = torch.from_numpy(img).to(device)
    elif img.ndim == 2:
        img_tensor = torch.from_numpy(img).to(device).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")

    img_tensor = img_tensor.float()
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    return img_tensor


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a torch.Tensor (C, H, W) or (1, H, W) to a numpy array (C, H, W, C) or (1, H, W).
    Assumes the input is float in [0, 1] or in [0, 255].

    Returns:
        np.ndarray with dtype=uint8
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with 3 dimensions (C, H, W), got shape {tensor.shape}")

    # break any connection to torch's gradient engine for conversion to numpy
    tensor = tensor.detach()
    
    # Unnormalize to 0–255 if needed
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        tensor = tensor.clamp(0, 255).byte()

    # If it's on GPU, move it to CPU
    arr = tensor.cpu().numpy()
    return arr