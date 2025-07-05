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
        print(arr)
        return arr

    @staticmethod
    def fft(arr: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift) + 1e-8)

class ImageDisplay:
    def __init__(self, window_title: str = "") -> None:
        self.window_title: str = window_title
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.img_obj: Optional[plt.AxesImage] = None

    def display_image(self, image: np.ndarray, coordinates=None, mode='Circle', color='red', size=10, title=None):
        if self.fig is None or self.ax is None or self.img_obj is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.img_obj = self.ax.imshow(image, cmap='gray')
            self.fig.canvas.manager.set_window_title(title or self.window_title)
            self.ax.axis('off')
        else:
            self.img_obj.set_data(image)
            self.fig.canvas.manager.set_window_title(title or self.window_title)
            self.fig.canvas.draw_idle()
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

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.img_obj = None
