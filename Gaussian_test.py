import os
import numpy as np
import tifffile
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


tiff_path = "data/X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif"

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop, "Microcrystals")
os.makedirs(output_folder, exist_ok=True)
output_txt_file = os.path.join(output_folder, "grace1.txt")
plot_file = os.path.join(output_folder, "grace1_gaussian_fit.png")

handle = tifffile.imread(tiff_path)
frames, height, width = handle.shape

region_size = 5
cx, cy = width // 2, height // 2
half = region_size // 2

abs_intensities = []
for i in range(frames):
    region = handle[i,
                    max(0, cy - half):min(height, cy + half + 1),
                    max(0, cx - half):min(width, cx + half + 1)]
    abs_i = np.mean(region)
    abs_intensities.append(abs_i)

abs_intensities = np.array(abs_intensities)
x_data = np.arange(frames)

def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

a_init = abs_intensities.max()
x0_init = x_data[np.argmax(abs_intensities)]
sigma_init = np.std(x_data)
offset_init = abs_intensities.min()
p0 = [a_init, x0_init, sigma_init, offset_init]

params, covariance = curve_fit(gaussian, x_data, abs_intensities, p0=p0)
a_fit, x0_fit, sigma_fit, offset_fit = params

with open(output_txt_file, "w") as f:
    f.write("Frame, Absolute_Intensity\n")
    for i, val in enumerate(abs_intensities):
        f.write(f"{i}, {val:.6f}\n")
    f.write("\nGaussian Fit Parameters:\n")
    f.write(f"Amplitude (a): {a_fit:.6f}\n")
    f.write(f"Center (x0): {x0_fit:.6f}\n")
    f.write(f"Sigma: {sigma_fit:.6f}\n")
    f.write(f"Offset: {offset_fit:.6f}\n")

print(f"Results saved to: {output_txt_file}")

x_fit = np.linspace(x_data.min(), x_data.max(), 500)
y_fit = gaussian(x_fit, *params)

plt.figure(figsize=(8, 5))
plt.plot(x_data, abs_intensities, 'bo', label="Data")
plt.plot(x_fit, y_fit, 'r--', label="Gaussian Fit", linewidth=2)
plt.xlabel("Frame")
plt.ylabel("Absolute Intensity")
plt.title("Gaussian Fit of Intensity Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.show()

print(f"Gaussian plot saved to: {plot_file}")
