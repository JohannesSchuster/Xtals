import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tifffile import imread

# ---- 1D Gaussian model ----
def oneD_Gaussian(x, amplitude, mean, sigma, offset):
    return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# ---- Load Image ----
file_path = r'C:\Users\ziegler\Desktop\Xtals\Data\X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif'
image_stack = imread(file_path)
image = image_stack[0]  # take the first slice/frame

# ---- Auto find max intensity near center ----
center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
radius = 2 # radius in pixels for searching max near center, tweak this to explore
x_min = max(center_x - radius, 0)
x_max = min(center_x + radius + 1, image.shape[1])
y_min = max(center_y - radius, 0)
y_max = min(center_y + radius + 1, image.shape[0])
search_region = image[y_min:y_max, x_min:x_max]
max_pos = np.unravel_index(np.argmax(search_region), search_region.shape)
max_y = y_min + max_pos[0]
max_x = x_min + max_pos[1]
print(f"Max intensity spot near center at (x={max_x}, y={max_y}), intensity={image[max_y, max_x]}")

# ---- Extract ROI ----
roi_size = 30
half_size = roi_size // 2
x1 = max(max_x - half_size, 0)
x2 = min(max_x + half_size, image.shape[1])
y1 = max(max_y - half_size, 0)
y2 = min(max_y + half_size, image.shape[0])
roi = image[y1:y2, x1:x2]

# ---- Extract horizontal profile line at vertical center of ROI ----
profile_y = roi.shape[0] // 2
x_data_full = np.arange(roi.shape[1])
y_data_full = roi[profile_y, :]

# ---- Define fitting window within ROI ----
x_min_fit = 7  # start pixel of fitting window (within ROI)
x_max_fit = 11 # end pixel of fitting window (within ROI)

# Select data within fitting window
mask = (x_data_full >= x_min_fit) & (x_data_full <= x_max_fit)
x_data = x_data_full[mask]
y_data = y_data_full[mask]

# ---- Initial guess for Gaussian parameters: amplitude, mean, sigma, offset ----
initial_guess_1d = [y_data.max(), (x_min_fit + x_max_fit) / 2, 1.0, y_data.min()]

# ---- Fit 1D Gaussian ----
try:
    popt_1d, _ = curve_fit(oneD_Gaussian, x_data, y_data, p0=initial_guess_1d)
    print("1D Gaussian fit succeeded.")
except Exception as e:
    popt_1d = None
    print("1D Gaussian fit failed:", e)

# ---- Plot horizontal profile and fitted curve ----
plt.figure(figsize=(8, 5))
plt.plot(x_data_full, y_data_full, 'b.', label='Data (horizontal profile)')

if popt_1d is not None:
    # Full ROI width curve for smooth display
    x_fit = np.linspace(0, roi.shape[1] - 1, 500)
    y_fit = oneD_Gaussian(x_fit, *popt_1d)
    plt.plot(x_fit, y_fit, 'r-', label='1D Gaussian fit (full ROI width)')
    
    # Show fitting window as shaded area
    plt.axvspan(x_min_fit, x_max_fit, color='green', alpha=0.2, label='Fit window')

plt.title("1D Gaussian Fit on Horizontal Profile")
plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Save fitted parameters to text file ----
if popt_1d is not None:
    with open('spot.txt', 'w') as f:
        f.write("1D Gaussian fit parameters:\n")
        f.write(f"Amplitude: {popt_1d[0]:.6f}\n")
        f.write(f"Mean (center): {popt_1d[1]:.6f} px\n")
        f.write(f"Sigma (width): {popt_1d[2]:.6f} px\n")
        f.write(f"Offset (baseline): {popt_1d[3]:.6f}\n")
    print("Parameters saved to spot.txt")
else:
    print("1D Gaussian fit failed; no parameters saved.")
