import torch
import torch.nn.functional as F

def rgb_to_grayscale(img_tensor: torch.Tensor):
    """
    img_tensor: torch.Tensor of shape (C, H, W) with C=3, values in [0,1] or [0,255]
    Returns grayscale image tensor of shape (1, H, W)
    """
    # If input is in [0,255], normalize to [0,1]
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # Luminosity method
    
    return gray.unsqueeze(0)  # Add channel dimension (1, H, W)


def resize_image_tensor(img_tensor: torch.Tensor, new_size: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    """
    Resizes a given torch.Tensor of size (C, H, W) to be of size (C, H_new, W_new)
    using pixel interpolation. The supported modes are:
        - "nearest" (default)
        - "bilinear"
        - "bicubic"
    """
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.ndim == 3 and img_tensor.shape[0] in (1, 3):
        pass
    else:
        raise ValueError("Expected image shape (C, H, W)")
    
    if mode not in ["nearest", "bilinear", "bicubic"]:
        raise RuntimeError("The interpolation mode needs to be 'nearest', 'bilinear' or 'bicubic'")

    img_tensor = img_tensor.unsqueeze(0)  # Add batch
    resized: torch.Tensor = F.interpolate(img_tensor, size=new_size, mode=mode, align_corners=False)
    return resized.squeeze(0)


def embed_image(background: torch.Tensor, image: torch.Tensor, offset: tuple[int, int]) -> torch.Tensor:
    """
    Embeds `image` into `background` at the given (oy, ox) offset.
    Ensures all tensors are on the same device and handles size clipping.

    Args:
        background (torch.Tensor): (C, H_bg, W_bg) tensor.
        image (torch.Tensor): (C, H_im, W_im) tensor.
        offset (tuple[int, int]): (oy, ox) offset for top-left placement.

    Returns:
        torch.Tensor: background with the image embedded at the offset.
    """
    # Ensure both tensors are on the same device
    if image.device != background.device:
        image = image.to(background.device)

    # Check shape consistency
    assert background.dim() == 3 and image.dim() == 3, "Both tensors must be (C, H, W)"
    assert background.shape[0] == image.shape[0], "Channel counts must match"

    _, h_bg, w_bg = background.shape
    _, h_im, w_im = image.shape
    oy, ox = offset

    # Calculate clipping bounds
    paste_h = min(h_im, h_bg - oy)
    paste_w = min(w_im, w_bg - ox)

    if paste_h <= 0 or paste_w <= 0:
        return background  # Nothing to paste

    # Embed the image
    background[:, oy:oy + paste_h, ox:ox + paste_w] = image[:, :paste_h, :paste_w]
    return background


def compute_fft_torch(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2x2 shifted fft of a torch.Tensor

    Returns:
        torch.Tensor: log magnitude of the fft (real numbers)
    """
    # Compute FFT on GPU
    fft = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    log_mag = torch.log1p(magnitude)

    # Normalize and bring back to CPU
    log_mag = (log_mag / log_mag.max()) * 255
    return log_mag

