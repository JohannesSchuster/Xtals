import torch
import torch.nn.functional as F

import abc
import numpy as np
from typing import Callable


class RenderContext:
    """Global context for shared state across jobs."""
    storage: dict = {}

# This is abstract and never to be used
class RenderOp:
    """Base class for all render operations."""

    @abc.abstractmethod
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        pass

# A incomplete list of render-commands
# TODO(Johannes): expand if needed

class Load(RenderOp):
    def __init__(self, source: Callable[[], torch.Tensor | np.ndarray]):
        self.source = source

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        new_img = self.source()
        if isinstance(new_img, np.ndarray):
            if new_img.ndim == 2:
                new_img = np.expand_dims(new_img, axis=2)
            new_img = torch.from_numpy(new_img).permute(2, 0, 1).float()
            if new_img.max() > 1:
                new_img /= 255.0
        elif isinstance(new_img, torch.Tensor):
            new_img = new_img.float()
            if new_img.max() > 1:
                new_img /= 255.0
        else:
            raise TypeError("Unsupported image source type")
        return new_img.to(img.device)


class Store(RenderOp):
    def __init__(self, name: str):
        self.name = name

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        RenderContext.storage[self.name] = img.clone()
        return img


class Recall(RenderOp):
    def __init__(self, name: str):
        self.name = name

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return RenderContext.storage.get(self.name, img)
    

class Clear(RenderOp):
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        RenderContext.storage.clear()
        return img


class Resize(RenderOp):
    def __init__(self, new_size: tuple[int, int], mode: str = "nearest"):
        self.new_size = new_size  # (height, width)
        self.mode = mode

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        # img shape: (C, H, W)
        img = img.unsqueeze(0)  # add batch dim
        resized: torch.Tensor = F.interpolate(img, size=self.new_size, mode=self.mode, align_corners=False)
        return resized.squeeze(0)

class Embed(RenderOp):
    def __init__(self, new_size: tuple[int, int], offset: tuple[int, int], blank=(82, 82, 82)):
        self.new_size = new_size  # (height, width)
        self.offset = offset      # (oy, ox)
        self.blank = blank        # background color RGB

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        # img shape: (C, H, W)
        c, h, w = img.shape
        canvas = torch.full((c, *self.new_size), 0, dtype=img.dtype, device=img.device)
        for i in range(c):
            canvas[i].fill_(self.blank[i])
        oy, ox = self.offset

        paste_h = min(h, self.new_size[0] - oy)
        paste_w = min(w, self.new_size[1] - ox)
        if paste_h <= 0 or paste_w <= 0:
            return canvas

        canvas[:, oy:oy+paste_h, ox:ox+paste_w] = img[:, :paste_h, :paste_w]
        return canvas

class Crop(RenderOp):
    def __init__(self, offset: tuple[int, int], size: tuple[int, int]):
        self.offset = offset  # (oy, ox)
        self.size = size      # (height, width)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        oy, ox = self.offset
        ch, cw = self.size

        # Clip to bounds
        oy = max(0, oy)
        ox = max(0, ox)
        ch = min(ch, h - oy)
        cw = min(cw, w - ox)

        return img[:, oy:oy+ch, ox:ox+cw]

class ToGrayscale(RenderOp):
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == 3:
            r, g, b = img[0], img[1], img[2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray.unsqueeze(0)
        return img


class Normalize(RenderOp):
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return (img - img.min()) / (img.max() - img.min() + self.tolerance)


class Clip(RenderOp):
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return img.clamp(self.min_val, self.max_val)
    

class RemapIntensity(RenderOp):
    """
    Remaps pixel values from a given input range to an output range,
    for grayscale or color images. Values outside the input range are clamped.

    Args:
        old_range (Tuple[float, float]): Input range (min, max).
        new_range (Tuple[float, float]): Output range (min, max).
    """

    def __init__(self, old_range: tuple[float, float], new_range: tuple[float, float] = (0, 255)):
        self.old_min, self.old_max = old_range
        self.new_min, self.new_max = new_range

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        assert image.ndim == 3, "Expected image with shape (C, H, W)"
        # Clamp to old input range
        img = image.clamp(self.old_min, self.old_max)
        # Normalize to [0, 1]
        img = (img - self.old_min) / (self.old_max - self.old_min)
        # Scale to new output range
        img = img * (self.new_max - self.new_min) + self.new_min
        return img.clamp(self.new_min, self.new_max).round().byte()


class Compose(RenderOp):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.fn = fn

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return self.fn(img)

