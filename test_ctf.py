import sys
import tifffile
import torch
import matplotlib.pyplot as plt
from ctf import CTF

### DEBUG ###
# For debug plotting
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt

# Lightweight debug visualizer for 2D torch tensors
def debug_tensor_plot(tensor: torch.Tensor, title: str = None, cmap: str = 'viridis'):
    """
    Lightweight debug visualizer for 2D torch tensors using matplotlib.
    Usage: debug_tensor_plot(tensor)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError("Only 2D tensors can be visualized")
    arr = tensor.detach().cpu().numpy()
    fig = plt.figure()
    plt.imshow(arr, cmap=cmap, aspect='equal')
    plt.colorbar()
    if title:
        plt.title(title)
    return fig

# Helper for 2D Gaussian decay
def get_gauss_decay(shape, sigma_x, sigma_y, device):
    h, w = shape
    y = torch.arange(h, device=device) - (h - 1) / 2
    x = torch.arange(w, device=device) - (w - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    gauss = torch.exp(-((xx**2) / (2 * sigma_x**2) + (yy**2) / (2 * sigma_y**2)))
    return gauss

def get_test_data(shape, noise: float, defocus1: float, defocus2: float, defocus_angle: float, device: str|None = None, sigma_x: float = 1.0, sigma_y: float = 1.0) -> torch.Tensor:
    """
    Generate a test tensor with a simple CTF pattern.
    This is for debugging purposes only.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_ctf = CTF(pixel_size=0.8791, voltage=200.0, Cs=2.7, phase_shift=0.0, amplitude_contrast=0.07)
    test_ctf.set_initial_params(defocus1, defocus2, defocus_angle/180*torch.pi)  # Initial defocus parameters
    test_data = test_ctf.get(shape)
    gauss_decay = get_gauss_decay(shape, sigma_x, sigma_y, device)
    test_data *= gauss_decay  # Apply Gaussian decay to the CTF pattern
    noise_tensor = torch.randn_like(test_data) * noise  # Add some noise
    test_data += noise_tensor  # Add noise to the CTF pattern
    test_data = torch.clamp(test_data, -1, 1)  # Clamp values to [-1, 1]
    return test_data

if len(sys.argv) < 2:
    print("Usage: python ctf.py <input.tiff>")
    sys.exit(1)
filename = sys.argv[1]
print(f"Reading: {filename}")
arr = tifffile.imread(filename)
# Convert to torch tensor, float32
data = torch.from_numpy(arr).float()
if data.ndim == 3:
    # Take sum over frames if multi-frame
    data = data.sum(dim=0)
elif data.ndim != 2:
    raise ValueError("Input TIFF must be 2D or 3D (frames, H, W)")
print(f"Data shape: {data.shape}")
# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
h, w = data.shape
crop_size = min(h, w)
cy, cx = h // 2, w // 2
half = crop_size // 2
croped = data[cy-half:cy+half, cx-half:cx+half]
test_data = get_test_data(croped.shape, noise=0.75,
                          defocus1=0.2, defocus2=0.1, defocus_angle=30.0, 
                          sigma_x=200.0, sigma_y=200.0)

ctf = CTF(pixel_size=0.8791, voltage=200.0, Cs=2.7, phase_shift=0.0, amplitude_contrast=0.07)
ctf.set_debug_level(CTF.DebugLevel.TIMING)
ctf.set_fit_params(N_d=1024, g_range=(5, 20.0),
                   defocus_range=(0.1, 1), defocus_step=0.02, defocus_restraint=0.01)
fft_data = ctf.get_fft_data(croped, N_frames=-1, eps=1e-16) # Use all frames for fitting

data = fft_data

def test_fit(data, ctf, method: str):
    ctf.defocus_estimation = method
    print(f"Testing CTF fit with method: {method}")
    params = ctf.fit(data, data_is_fourier=True, tol=1e-8, max_iter=1000)
    print("Fitted CTF parameters:", params)
    # Diaplay the CTF pattern
    alpha = 3  # Adjust contrast
    sigma = 2  # Standard deviation for contrast adjustment
    data = (data - data.min()) / (data.max() - data.min() + 1e-12)
    mean = data.mean()
    std = data.std()
    vmin = float(mean - sigma * std)
    vmax = float(mean + sigma * std)
    data = torch.clamp((data - vmin) / (vmax - vmin), 0, 1)
    data = data**alpha
    # Visualize input and CTF
    ctf_img = 0.5 * ctf.get(data.shape) + 0.5
    #mask = ctf.get_g_mask(fft_data.shape) * ctf.get_ray_mask(fft_data.shape)
    #debug_tensor_plot(mask, title="Complete Mask")
    #plt.show()
    g_min, g_max = torch.tensor(data.shape) / (2 * ctf.get_fit_params()[1:3] * ctf.pixel_size)
    h, w = data.shape
    data[h//2:h,0:w//2] = ctf_img[h//2:h,0:w//2]
    # fft_data[0:h//2, 0:w//2] = g_mask[0:h//2, 0:w//2]
    fig = debug_tensor_plot(data.cpu(), title="Diagnostic Image")
    ax = fig.gca()
    origin = data.shape[1] // 2, data.shape[0] // 2
    circ1 = plt.Circle(origin, radius=g_min, linewidth=1, edgecolor="orange", facecolor='none')
    circ2 = plt.Circle(origin, radius=g_max, linewidth=1, edgecolor="red", facecolor='none')
    
    # Add rectangle showing the crop box size (N_d)
    N_d = ctf.get_fit_params()[0].item()  # Get crop size from fit params
    half_box = N_d // 2
    center_x, center_y = origin
    rect = plt.Rectangle((center_x - half_box, center_y - half_box), N_d, N_d, linewidth=1, edgecolor="black", facecolor='none')

    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.add_patch(rect)
    plt.show()

#test_fit(data, ctf, "linear")
test_fit(data, ctf, "gradient")
test_fit(data, ctf, "annealing")