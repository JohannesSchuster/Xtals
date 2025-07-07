import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from tifffile import imread

# --- 2D Gaussian model ---
def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    g = offset + amplitude * np.exp(- (a*(x - xo)**2 + 2*b*(x - xo)*(y - yo) + c*(y - yo)**2))
    return g.ravel()

# --- Load Image ---
file_path = r'C:\Users\ziegler\Desktop\Xtals\Data\X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif'
image_stack = imread(file_path)
image = image_stack[0]

# --- Find brightest point near center ---
center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
radius = 2
x_min = max(center_x - radius, 0)
x_max = min(center_x + radius + 1, image.shape[1])
y_min = max(center_y - radius, 0)
y_max = min(center_y + radius + 1, image.shape[0])
search_region = image[y_min:y_max, x_min:x_max]
max_pos = np.unravel_index(np.argmax(search_region), search_region.shape)
max_y = y_min + max_pos[0]
max_x = x_min + max_pos[1]

print(f"Brightest spot near center: (x={max_x}, y={max_y}), intensity={image[max_y, max_x]}")

# --- Extract 2D ROI around bright spot ---
roi_size = 30
half = roi_size // 2
x1 = max(max_x - half, 0)
x2 = min(max_x + half, image.shape[1])
y1 = max(max_y - half, 0)
y2 = min(max_y + half, image.shape[0])
roi = image[y1:y2, x1:x2]

# --- Prepare meshgrid ---
x = np.linspace(0, roi.shape[1] - 1, roi.shape[1])
y = np.linspace(0, roi.shape[0] - 1, roi.shape[0])
x, y = np.meshgrid(x, y)

# --- Fit 2D Gaussian ---
initial_guess = (
    roi.max(),  # amplitude
    roi.shape[1]/2,  # xo
    roi.shape[0]/2,  # yo
    3, 3,            # sigma_x, sigma_y
    0,               # theta
    np.min(roi)      # offset
)

try:
    popt, _ = curve_fit(twoD_Gaussian, (x, y), roi.ravel(), p0=initial_guess)
    print("\n2D Gaussian fit succeeded.")
except Exception as e:
    popt = None
    print("2D Gaussian fitting failed:", e)

# --- Plotting ---
if popt is not None:
    # Generate smooth surface
    x_fit = np.linspace(0, roi.shape[1] - 1, 100)
    y_fit = np.linspace(0, roi.shape[0] - 1, 100)
    x_fit, y_fit = np.meshgrid(x_fit, y_fit)
    z_fit = twoD_Gaussian((x_fit, y_fit), *popt).reshape(x_fit.shape)

    # Original ROI plot
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(roi, cmap='viridis')
    ax1.set_title("Original ROI")
    ax1.axis('off')

    # 3D surface plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(x_fit, y_fit, z_fit, cmap='plasma', edgecolor='none')
    ax2.set_title("2D Gaussian Fit Surface")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Intensity')
    ax2.view_init(elev=30, azim=230)

    # Residuals
    z_data = roi
    z_model = twoD_Gaussian((x, y), *popt).reshape(roi.shape)
    residual = z_data - z_model

    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(residual, cmap='coolwarm')
    ax3.set_title("Residual (Data - Fit)")
    fig.colorbar(im, ax=ax3, shrink=0.6)
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Save fitted parameters ---
    with open('2D_gaussian_fit.txt', 'w') as f:
        f.write("2D Gaussian fit parameters:\n")
        f.write(f"Amplitude: {popt[0]:.4f}\n")
        f.write(f"X center:  {popt[1]:.4f} px\n")
        f.write(f"Y center:  {popt[2]:.4f} px\n")
        f.write(f"Sigma X:   {popt[3]:.4f} px\n")
        f.write(f"Sigma Y:   {popt[4]:.4f} px\n")
        f.write(f"Theta:     {popt[5]:.4f} rad\n")
        f.write(f"Offset:    {popt[6]:.4f}\n")

    print("2D Gaussian parameters saved to '2D_gaussian_fit.txt'")
else:
    print("Fit failed; no output generated.")
