import numpy as np
import torch

def np_to_tensor(img: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    if img.ndim == 3:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    elif img.ndim == 2:
        img_tensor = torch.from_numpy(img).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")

    img_tensor = img_tensor.float().to(device)
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    return img_tensor