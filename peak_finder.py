import numpy as np
from skimage.feature import peak_local_max
from matplotlib.path import Path
from typing import List, Optional, Tuple, Sequence

#from loggging import Logging

class Mask:
    def apply(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def as_mask(self, shape: tuple) -> np.ndarray:
        # Default: create a dummy array and call apply
        dummy = np.zeros(shape, dtype=bool)
        return self.apply(dummy)

class RectMask(Mask):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def apply(self, arr: np.ndarray) -> np.ndarray:
        mask = np.zeros(arr.shape, dtype=bool)
        mask[self.y:self.y+self.height, self.x:self.x+self.width] = True
        return mask
    def as_mask(self, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        mask[self.y:self.y+self.height, self.x:self.x+self.width] = True
        return mask

class CircleMask(Mask):
    def __init__(self, x: int, y: int, r: int):
        self.x = x
        self.y = y
        self.r = r
    def apply(self, arr: np.ndarray) -> np.ndarray:
        Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
        mask = (X - self.x)**2 + (Y - self.y)**2 <= self.r**2
        return mask
    def as_mask(self, shape: tuple) -> np.ndarray:
        Y, X = np.ogrid[:shape[0], :shape[1]]
        mask = (X - self.x)**2 + (Y - self.y)**2 <= self.r**2
        return mask

class PolyMask(Mask):
    def __init__(self, vertices: Sequence[Tuple[float, float]]):
        self.vertices = np.array(vertices)
    def apply(self, arr: np.ndarray) -> np.ndarray:
        Y, X = np.mgrid[:arr.shape[0], :arr.shape[1]]
        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(self.vertices)
        mask = path.contains_points(points).reshape(arr.shape)
        return mask
    def as_mask(self, shape: tuple) -> np.ndarray:
        Y, X = np.mgrid[:shape[0], :shape[1]]
        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(self.vertices)
        mask = path.contains_points(points).reshape(shape)
        return mask

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

    def find_peaks(self, image: np.ndarray, window: int = 3) -> np.ndarray:
        # Combine all masks (logical OR)
        if self.masks:
            mask_total = np.zeros(image.shape, dtype=bool)
            for m in self.masks:
                mask_total |= m.apply(image)
            labels: Optional[np.ndarray] = (~mask_total).astype(int)
        else:
            labels = None
        print(f"PeakFinder: {len(self.masks)} masks applied, labels shape: {labels.shape if labels is not None else 'None'}")
        print(f"PeakFinder: Finding peaks with min_distance={self.min_distance}, threshold_abs={self.threshold_abs}")
        coordinates = peak_local_max(
            image,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            exclude_border=False,
            labels=labels
        )
        print(f"PeakFinder: Found {len(coordinates)} peaks")
        # Sort by average brightness in a window around each peak
        if len(coordinates) > 0:
            pad = window // 2
            avgs = []
            h, w = image.shape
            for y, x in coordinates:
                y0, x0 = y, x
                y1, y2 = max(0, y0-pad), min(h, y0+pad+1)
                x1, x2 = max(0, x0-pad), min(w, x0+pad+1)
                region = image[y1:y2, x1:x2]
                avgs.append(region.mean())
            avgs = np.array(avgs)
            sort_idx = np.argsort(avgs)[::-1]  # descending
            coordinates = coordinates[sort_idx]
        return coordinates
