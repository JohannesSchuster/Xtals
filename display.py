import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

class ImageHandler:
    def __init__(self):
        self.handle: Optional[np.ndarray] = None
        self.last_result: Optional[np.ndarray] = None
        self.last_settings: Optional[tuple] = None

    def set_handle(self, handle: np.ndarray):
        self.handle = handle
        self.last_result = None
        self.last_settings = None

    def get_image(self, show_sum: bool, do_fft: bool, frame_idx: int) -> Optional[np.ndarray]:
        settings = (show_sum, do_fft, frame_idx)
        if self.last_result is not None and self.last_settings == settings:
            return self.last_result
        if self.handle is None:
            return None
        if show_sum:
            arr = np.sum(self.handle, axis=0)
        else:
            arr = self.handle[frame_idx]
        ## Always scale to 0-1 before further processing
        #arr = arr.astype(np.float32)
        #arr_min, arr_max = arr.min(), arr.max()
        #if arr_max > 1.0 or arr_min < 0.0:
        #    arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
        if do_fft:
            arr = self.fft(arr)
        self.last_result = arr
        self.last_settings = settings
        return arr

    @staticmethod
    def fft(arr: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift) + 1e-8)

class BaseDisplay:
    def __init__(self):
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def set_window_title(self, title: str):
        if self.fig is not None and hasattr(self.fig.canvas, 'manager'):
            self.fig.canvas.manager.set_window_title(title)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

class ImageDisplay(BaseDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.img_obj: Optional[plt.AxesImage] = None
        self.overlay: Optional[plt.AxesImage] = None

    def display_image(self, 
                      image: np.ndarray, 
                      coordinates: np.ndarray | None = None, 
                      mode: str = 'Circle', color: str = 'red', size: int=10, 
                      title: str = '', 
                      black: float = 0, white:float = 255, 
                      overlay: np.ndarray | None = None):
        # If the window was closed, reset all figure/axes/img_obj
        if self.fig is not None and (not plt.fignum_exists(self.fig.number)):
            self.fig = None
            self.ax = None
            self.img_obj = None
            self.overlay = None
        if self.fig is None or self.ax is None or self.img_obj is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.img_obj = self.ax.imshow(image, cmap='gray', vmin=black, vmax=white)
            self.overlay = None
            self.set_window_title(title )
            self.ax.axis('off')
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Make image fill the window
        else:
            self.img_obj.set_data(image)
            self.img_obj.set_clim(black, white)
            self.set_window_title(title)
            # Remove all previous overlays (patches, lines, and overlay image)
            for artist in list(self.ax.artists) + list(self.ax.patches) + list(self.ax.lines):
                artist.remove()
            if self.overlay is not None:
                self.overlay.remove()
                self.overlay = None
            self.fig.canvas.draw_idle()
        # Draw overlay as RGBA image if provided
        if overlay is not None:
            self.overlay = self.ax.imshow(overlay, interpolation='none')
        if coordinates is not None:
            for y, x in coordinates:
                if mode == 'Circle':
                    circ = plt.Circle((x, y), radius=size, linewidth=1, edgecolor=color, facecolor='none')
                    self.ax.add_patch(circ)
                elif mode == 'Extraction Box':
                    box = plt.Rectangle((x-size, y-size), 2*size, 2*size, linewidth=1, edgecolor=color, facecolor='none')
                    self.ax.add_patch(box)
                elif mode == 'Point':
                    self.ax.plot(x, y, 'o', color=color, markersize=size)
        plt.show(block=False)

class HistogramDisplay(BaseDisplay):
    def __init__(self):
        super().__init__()
        self.bar_container = None
        self.cutoff_line = None

    def display_histogram(self, image: np.ndarray, cutoff: float, title: str = "", black: float = 0, white: float = 255):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots()
            self.set_window_title(f"Histogram: {title}")
        else:
            self.ax.clear()
            self.set_window_title(f"Histogram: {title}")
        # Compute histogram and normalize counts
        bins = int(white - black)
        if bins < 1:
            bins = 1
        hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(black, white))
        hist = hist / hist.sum() if hist.sum() > 0 else hist  # relative counts
        self.bar_container = self.ax.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), color='gray', alpha=0.8, align='edge')
        # Draw cutoff line
        self.cutoff_line = self.ax.axvline(cutoff, color='red', linestyle='--', label=f'Cutoff ({cutoff:.2f})')
        self.ax.set_xlabel('Pixel Value')
        self.ax.set_ylabel('Relative Count')
        self.ax.set_xlim(black, white)
        self.ax.legend()
        self.ax.set_title("")
        self.fig.tight_layout()
        self.fig.show()
