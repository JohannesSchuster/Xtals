import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tifffile import imread

# ---- 3D Gaussian function ----
def threeD_Gaussian(coords, amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset):
    x, y, z = coords
    exp_part = (
        ((x - x0) ** 2) / (2 * sigma_x ** 2) +
        ((y - y0) ** 2) / (2 * sigma_y ** 2) +
        ((z - z0) ** 2) / (2 * sigma_z ** 2)
    )
    return offset + amplitude * np.exp(-exp_part)

# ---- Load your 3D TIFF image ----
file_path = r'C:\Users\ziegler\Desktop\Xtals\Data\X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif'
image_stack = imread(file_path)  # shape = (Z, Y, X) or (frames, height, width)

print(f"Image stack shape: {image_stack.shape}")

# Confirm shape ordering, assume (Z, Y, X) or (frames, height, width)
# Let's assign as (Z, Y, X) for clarity
Z, Y, X = image_stack.shape

# Create coordinate grids
x = np.arange(X)
y = np.arange(Y)
z = np.arange(Z)
x, y, z = np.meshgrid(x, y, z, indexing='xy')  # x,y,z shape = (Y,X,Z) if indexing='xy'
# But we want consistent shape to image_stack (Z,Y,X) - so swap axes accordingly:
x = x.transpose(2,0,1)  # shape to (Z, Y, X)
y = y.transpose(2,0,1)
z = z.transpose(2,0,1)

# Flatten data for fitting
coords = np.vstack((x.ravel(), y.ravel(), z.ravel()))
data_flat = image_stack.ravel()

# ---- Initial guess ----
amplitude_guess = data_flat.max() - data_flat.min()
x0_guess = X / 2
y0_guess = Y / 2
z0_guess = Z / 2
sigma_guess = min(X, Y, Z) / 6  # rough guess
offset_guess = data_flat.min()

initial_guess = [amplitude_guess, x0_guess, y0_guess, z0_guess, sigma_guess, sigma_guess, sigma_guess, offset_guess]

print("Starting 3D Gaussian fit — this might take a while...")

# ---- Fit the 3D Gaussian ----
popt, pcov = curve_fit(threeD_Gaussian, coords, data_flat, p0=initial_guess, maxfev=5000)

print("Fit completed!")
print("Fitted parameters:")
param_names = ['Amplitude', 'x0', 'y0', 'z0', 'sigma_x', 'sigma_y', 'sigma_z', 'Offset']
for name, val in zip(param_names, popt):
    print(f"  {name}: {val:.3f}")

# ---- Compute fitted Gaussian volume ----
fitted_volume = threeD_Gaussian(coords, *popt).reshape((Z, Y, X))

# ---- Extract center indices ----
x0_i = int(round(popt[1]))
y0_i = int(round(popt[2]))
z0_i = int(round(popt[3]))

# ---- Plot slices at fitted center ----
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# Slice along X-axis (Y-Z plane)
axes[0, 0].imshow(image_stack[:, :, x0_i], cmap='viridis', origin='lower')
axes[0, 0].set_title(f'Raw data slice X={x0_i}')
axes[0, 1].imshow(fitted_volume[:, :, x0_i], cmap='viridis', origin='lower')
axes[0, 1].set_title(f'Fitted Gaussian slice X={x0_i}')

# Slice along Y-axis (Z-X plane)
axes[1, 0].imshow(image_stack[:, y0_i, :], cmap='viridis', origin='lower')
axes[1, 0].set_title(f'Raw data slice Y={y0_i}')
axes[1, 1].imshow(fitted_volume[:, y0_i, :], cmap='viridis', origin='lower')
axes[1, 1].set_title(f'Fitted Gaussian slice Y={y0_i}')

# Slice along Z-axis (Y-X plane)
axes[2, 0].imshow(image_stack[z0_i, :, :], cmap='viridis', origin='lower')
axes[2, 0].set_title(f'Raw data slice Z={z0_i}')
axes[2, 1].imshow(fitted_volume[z0_i, :, :], cmap='viridis', origin='lower')
axes[2, 1].set_title(f'Fitted Gaussian slice Z={z0_i}')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
