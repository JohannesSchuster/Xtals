import torch
from matplotlib.path import Path
from typing import List, Optional, Tuple, Sequence

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
    def __init__(self, min_distance: int = 10, threshold_abs: float = 128):
        self.min_distance: int = min_distance
        self.threshold_abs: float = threshold_abs
        self.masks: List[Mask] = []

    def add_mask(self, mask: Mask) -> "PeakFinder":
        self.masks.append(mask)
        return self

    def clear_masks(self) -> "PeakFinder":
        self.masks = []
        return self

    def find_peaks(self, image: torch.Tensor, window: int = 3) -> torch.Tensor:
        # Combine all masks (logical OR)
        device = image.device
        if self.masks:
            mask_total = torch.zeros(image.shape, dtype=torch.bool, device=device)
            for m in self.masks:
                mask_total |= m.apply(image)
            valid_mask = ~mask_total
        else:
            valid_mask = torch.ones(image.shape, dtype=torch.bool, device=device)
        # Find local maxima using max pooling
        pad = window // 2
        image_padded = torch.nn.functional.pad(image, (pad, pad, pad, pad), mode='constant', value=float('-inf'))
        unfolded = image_padded.unfold(0, window, 1).unfold(1, window, 1)
        local_max = unfolded.max(dim=-1)[0].max(dim=-1)[0]
        is_peak = (image == local_max) & (image > self.threshold_abs) & valid_mask
        # Enforce min_distance by iterative suppression
        coords = torch.nonzero(is_peak, as_tuple=False)
        if coords.shape[0] == 0:
            return coords
        # Sort by intensity
        intensities = image[coords[:,0], coords[:,1]]
        sorted_idx = torch.argsort(intensities, descending=True)
        coords = coords[sorted_idx]
        # Non-maximum suppression for min_distance
        keep = []
        taken = torch.zeros(coords.shape[0], dtype=torch.bool, device=device)
        for i in range(coords.shape[0]):
            if taken[i]:
                continue
            y, x = coords[i].tolist()
            keep.append([y, x])
            dist = torch.sqrt((coords[:,0] - y)**2 + (coords[:,1] - x)**2)
            taken |= dist < self.min_distance
        coords_out = torch.tensor(keep, dtype=torch.int64, device=device)
        return coords_out
