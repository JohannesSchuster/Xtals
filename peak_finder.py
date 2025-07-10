import torch
from matplotlib.path import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
from timer import timed, Timer

class Mask:
    def apply(self, arr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def as_mask(self, shape: tuple, device=None) -> torch.Tensor:
        # Default: create a dummy array and call apply
        dummy = torch.zeros(shape, dtype=torch.bool, device=device)
        return self.apply(dummy)

class RectMask(Mask):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def apply(self, arr: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(arr.shape, dtype=torch.bool, device=arr.device)
        mask[self.y:self.y+self.height, self.x:self.x+self.width] = True
        return mask
    def as_mask(self, shape: tuple, device=None) -> torch.Tensor:
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        mask[self.y:self.y+self.height, self.x:self.x+self.width] = True
        return mask

class CircleMask(Mask):
    def __init__(self, x: int, y: int, r: int):
        self.x = x
        self.y = y
        self.r = r
    def apply(self, arr: torch.Tensor) -> torch.Tensor:
        Y, X = torch.meshgrid(torch.arange(arr.shape[0], device=arr.device), torch.arange(arr.shape[1], device=arr.device), indexing='ij')
        mask = (X - self.x)**2 + (Y - self.y)**2 <= self.r**2
        return mask
    def as_mask(self, shape: tuple, device=None) -> torch.Tensor:
        Y, X = torch.meshgrid(torch.arange(shape[0], device=device), torch.arange(shape[1], device=device), indexing='ij')
        mask = (X - self.x)**2 + (Y - self.y)**2 <= self.r**2
        return mask

class PolyMask(Mask):
    def __init__(self, vertices: Sequence[Tuple[float, float]]):
        self.vertices = torch.tensor(vertices, dtype=torch.float32)
    def apply(self, arr: torch.Tensor) -> torch.Tensor:
        # PolyMask is not easily torch-native due to Path.contains_points, so fallback to numpy for mask creation
        import numpy as np
        Y, X = np.mgrid[:arr.shape[0], :arr.shape[1]]
        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(self.vertices.cpu().numpy())
        mask = path.contains_points(points).reshape(arr.shape)
        return torch.from_numpy(mask).to(arr.device)
    def as_mask(self, shape: tuple, device=None) -> torch.Tensor:
        import numpy as np
        Y, X = np.mgrid[:shape[0], :shape[1]]
        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(self.vertices.cpu().numpy())
        mask = path.contains_points(points).reshape(shape)
        return torch.from_numpy(mask).to(device)

class PeakFinder:
    @dataclass
    class Cache:
        coordinates: torch.Tensor = field(default_factory=lambda: torch.empty((0, 2), dtype=torch.float32))
        
    def __init__(self, min_distance: int = 10, threshold_abs: float = 128):
        self.min_distance: int = min_distance
        self.threshold_abs: float = threshold_abs
        self.masks: List[Mask] = []
        self.cache = PeakFinder.Cache()

    def add_mask(self, mask: Mask) -> "PeakFinder":
        self.masks.append(mask)
        return self

    def clear_masks(self) -> "PeakFinder":
        self.masks = []
        return self

    def find_peaks(self, image: torch.Tensor) -> torch.Tensor:
        """
        Fast local maxima detection using max pooling (like skimage.feature.peak_local_max).
        """
        device = image.device
        timer: Timer = Timer()
        timer.start()
        # 1. Max pooling for local maxima
        window = 2 * self.min_distance + 1
        image_max = torch.nn.functional.max_pool2d(
            image.unsqueeze(0).unsqueeze(0),
            kernel_size=window,
            stride=1,
            padding=self.min_distance
        )[0, 0]
        print(f"[PeakFinder] Max pooling: {timer.elapsed*1000:.1f} ms")

        # 2. Find local maxima
        timer.start()
        is_peak = (image == image_max) & (image > self.threshold_abs)
        print(f"[PeakFinder] Find local maxima: {timer.elapsed*1000:.1f} ms")

        # 3. Masking (if any masks are present)
        timer.start()
        if self.masks:
            mask_total = torch.zeros(image.shape, dtype=torch.bool, device=device)
            for m in self.masks:
                mask_total |= m.apply(image)
            valid_mask = ~mask_total
            is_peak &= valid_mask
        print(f"[PeakFinder] Apply masks: {timer.elapsed*1000:.1f} ms")

        # 4. Get coordinates
        timer.start()
        coords = torch.nonzero(is_peak, as_tuple=False)
        print(f"[PeakFinder] Get coordinates: {timer.elapsed*1000:.1f} ms")

        # 5. Optionally sort by intensity (descending)
        timer.start()
        if coords.shape[0] > 0:
            intensities = image[coords[:,0], coords[:,1]]
            sorted_idx = torch.argsort(intensities, descending=True)
            coords = coords[sorted_idx]
        print(f"[PeakFinder] Sort coordinates: {timer.elapsed*1000:.1f} ms")
        self.cache.coordinates = coords
