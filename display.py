import matplotlib.pyplot as plt
import torch
from typing import Optional

class Image:
    def __init__(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Image must be initialized with a torch.Tensor")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tensor = tensor.detach().to(device).float()

    def cpu(self):
        return self._tensor.cpu()

    def numpy(self):
        return self.cpu().numpy()

    @property
    def shape(self):
        return self._tensor.shape

    def clone(self):
        return Image(self._tensor.clone())

    def to(self, device):
        return Image(self._tensor.to(device))

    def mean(self):
        return self._tensor.mean()

    def std(self):
        return self._tensor.std()

    def clamp(self, min_val=0.0, max_val=1.0):
        return Image(torch.clamp(self._tensor, min_val, max_val))

    def sigma(self, sigma=1.0):
        mean = self.mean()
        std = self.std()
        vmin = float(mean - sigma * std)
        vmax = float(mean + sigma * std)
        return Image(torch.clamp((self._tensor - vmin) / (vmax - vmin), 0, 1))

    def gamma(self, gamma=1.0):
        if gamma == 1.0:
            return self.clone()
        return Image(self._tensor ** gamma)

    def rescale(self, sigma: float = 1.0, gamma: float = 1.0, clamp_min: float = 0.0, clamp_max: float = 1.0):
        img = self._tensor
        # Sigma contrast stretch
        if sigma > 0:
            mean = img.mean()
            std = img.std()
            vmin = float(mean - sigma * std)
            vmax = float(mean + sigma * std)
            img = (img - vmin) / (vmax - vmin)
        # Clamp
        img = torch.clamp(img, clamp_min, clamp_max)
        # Gamma correction
        if gamma != 1.0:
            img = img ** gamma
        return Image(img)

    def remap(self, min_val: float = 0.0, max_val: float = 1.0):
        """Remaps the given input scale to be between 0 and 1"""
        img = self._tensor
        # Avoid division by zero
        if max_val == min_val:
            return Image(torch.zeros_like(img))
        img = (img - min_val) / (max_val - min_val)
        img = torch.clamp(img, 0.0, 1.0)
        return Image(img)

    def flatten(self):
        return self._tensor.flatten()
       
    def __getitem__(self, idx):
        return Image(self._tensor[idx])

    def __getattr__(self, name):
        # Forward attribute access to the underlying tensor
        return getattr(self._tensor, name)

class ImageHandler:
    def __init__(self):
        self.handle: Optional[torch.Tensor] = None
        self.sum: Optional[Image] = None

    @property
    def device(self):
        return self.handle.device if self.handle is not None else 'cpu'

    def set_handle(self, handle: torch.Tensor):
        self.handle = handle

    def precompute(self):
        if self.handle is not None:
            self.sum = Image(torch.sum(self.handle, dim=0))
        else:
            self.sum = None
        self.last_result = None
        self.last_settings = None

    def get_sum(self) -> Optional[Image]:
        if self.sum is None:
            self.precompute()
        return self.sum

    def get_frame(self, frame_idx: int) -> Optional[Image]:
        idx = self._get_valid_frame_idx(frame_idx)
        return Image(self.handle[idx])
    
    def _get_valid_frame_idx(self, frame_idx: int) -> int:
        if self.handle is None or frame_idx < 0 or frame_idx >= self.handle.shape[0]:
            return 0
        return frame_idx

class FFImageHandler(ImageHandler):
    def __init__(self):
        super().__init__()
        self.frames: Optional[list[Image]] = None
    
    def precompute(self):
        # Multi-frame: compute FFT for each frame
        self.frames = [Image(self.fft(frame)) for frame in self.handle]
        self.sum = Image(self.fft(torch.sum(self.handle, dim=0)))

    def get_frame(self, frame_idx):
        idx = self._get_valid_frame_idx(frame_idx)
        if self.frames is None:
            self.precompute()
        return self.frames[idx]

    @staticmethod
    def fft(arr: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fft2(arr)
        fshift = torch.fft.fftshift(f)
        return torch.log(torch.abs(fshift) + 1e-12)
    
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
                      image: Image, 
                      coordinates: torch.Tensor | None = None, 
                      overlay: Image | None = None,
                      mode: str = 'Circle', color: str = 'red', size: int=10, 
                      title: str = ''):    
        # If the window was closed, reset all figure/axes/img_obj
        if self.fig is not None and (not plt.fignum_exists(self.fig.number)):
            self.fig = None
            self.ax = None
            self.img_obj = None
            self.overlay = None
        if self.fig is None or self.ax is None or self.img_obj is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.img_obj = self.ax.imshow(image.numpy(), cmap='gray')
            self.overlay = None
            self.set_window_title(title)
            self.ax.axis('off')
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Make image fill the window
        else:
            self.img_obj.set_data(image.numpy())
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
            self.overlay = self.ax.imshow(overlay.numpy(), interpolation='none')
        if coordinates is not None:
            coordinates = coordinates.cpu().numpy()
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

    def display_histogram(self, image: Image, cutoff: float, title: str = "", black: float = 0, white: float = 1, xscale: str = 'linear', yscale: str = 'linear'):
        # Use torch for histogram computation, convert to numpy for plotting
        bins = int((white - black) * 256)
        if bins < 1:
            bins = 1
        # Compute histogram with torch
        hist, bin_edges = torch.histogram(image.cpu().flatten(), bins=bins, range=(black, white))
        hist = hist.float()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        # Convert to numpy for matplotlib
        hist_np = hist.numpy()
        bin_edges_np = bin_edges.numpy()
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots()
            self.set_window_title(f"Histogram: {title}")
        else:
            self.ax.clear()
            self.set_window_title(f"Histogram: {title}")
        self.bar_container = self.ax.bar(bin_edges_np[:-1], hist_np, width=(bin_edges_np[1]-bin_edges_np[0]), color='gray', alpha=0.8, align='edge')
        self.cutoff_line = self.ax.axvline(cutoff, color='red', linestyle='--', label=f'Cutoff ({cutoff:.2f})')
        self.ax.set_xlabel('Pixel Value')
        self.ax.set_ylabel('Relative Count')
        self.ax.set_xlim(black, white)
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        self.ax.legend()
        self.ax.set_title("")
        self.fig.tight_layout()
        self.fig.show()
