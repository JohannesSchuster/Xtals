import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Gaussian function for fitting ===
def gaussian(x, amp, mean, sigma, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset

# === Path to your TIFF stack ===
tiff_path = r"C:\Users\ziegler\Desktop\Xtals\Data\2025-06-19_21.53.27_Xlzso-NO3-3_r1_77_000.tif"

# Load the TIFF stack (frames x height x width)
stack = tifffile.imread(tiff_path)

# Get number of frames
num_frames = stack.shape[0]

# Find the brightest pixel coordinates over the whole stack
max_pos = np.unravel_index(np.argmax(stack), stack.shape)
frame_max, y_max, x_max = max_pos
print(f"Brightest pixel found at (x={x_max}, y={y_max})")

# Extract intensity of that pixel over all frames
intensity_over_time = stack[:, y_max, x_max]

# Frame indices as x-axis
frames = np.arange(num_frames)

# Initial guess for Gaussian parameters: amp, mean, sigma, offset
initial_guess = [intensity_over_time.max(), num_frames / 2, num_frames / 5, intensity_over_time.min()]

# Fit Gaussian
params, _ = curve_fit(gaussian, frames, intensity_over_time, p0=initial_guess)
amp, mean, sigma, offset = params

# Create fitted Gaussian curve
fitted_curve = gaussian(frames, *params)

# === Save results to a text file ===
output_path = r"C:\Users\ziegler\Desktop\Microcrystals\exercice.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== Crystal Brightness Analysis ===\n")
    f.write(f"TIFF File: {os.path.basename(tiff_path)}\n")
    f.write(f"Brightest Pixel Coordinates: x={x_max}, y={y_max}, (max at frame {frame_max})\n")
    f.write("\nGaussian Fit Parameters:\n")
    f.write(f"Amplitude (A): {amp:.2f}\n")
    f.write(f"Mean Frame (μ): {mean:.2f}\n")
    f.write(f"Sigma (σ): {sigma:.2f}\n")
    f.write(f"Offset: {offset:.2f}\n")
    f.write("\nNote: Coordinates are in image space (pixels).\n")

    f.write("\nGaussian Curve Data (Frame, Intensity):\n")
    for frame, intensity in zip(frames, fitted_curve):
        f.write(f"{frame}\t{intensity:.2f}\n")

print(f"Results written to {output_path}")

# === Plot original intensity data and Gaussian fit ===
plt.figure(figsize=(8,5))
plt.plot(frames, intensity_over_time, 'bo', label='Intensity (data)')
plt.plot(frames, fitted_curve, 'r-', label='Gaussian fit', linewidth=2)
plt.xlabel('Frame Number')
plt.ylabel('Intensity')
plt.title('Gaussian Fit of Brightest Pixel Intensity over Frames')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
