import os
import numpy as np
import tifffile
from scipy.ndimage import maximum_filter, label, find_objects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


tiff_path = "data/X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif" 

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop, "Microcrystals")
os.makedirs(output_folder, exist_ok=True)
output_txt_file = os.path.join(output_folder, "gloire4.txt")

handle = tifffile.imread(tiff_path)
frames, height, width = handle.shape

summed_image = np.sum(handle, axis=0)
f = np.fft.fft2(summed_image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

neighborhood_size = 5
max_filtered = maximum_filter(magnitude_spectrum, size=neighborhood_size)
local_max = (magnitude_spectrum == max_filtered)
threshold = np.percentile(magnitude_spectrum, 90)
detected_maxima = local_max & (magnitude_spectrum > threshold)
labeled, num_features = label(detected_maxima)
slices = find_objects(labeled)

maxima_coords = []
for dy, dx in slices:
    x_center = (dx.start + dx.stop - 1) // 2
    y_center = (dy.start + dy.stop - 1) // 2
    maxima_coords.append((x_center, y_center))

cx_px, cy_px = width / 2, height / 2
distances = [np.sqrt((x - cx_px)**2 + (y - cy_px)**2) for x, y in maxima_coords]
closest_index = np.argmin(distances)
x_px, y_px = maxima_coords[closest_index]

region_size = 5
half = region_size // 2

abs_intensities = []
for i in range(frames):
    region = handle[i,
                    max(0, y_px - half):min(height, y_px + half + 1),
                    max(0, x_px - half):min(width, x_px + half + 1)]
    abs_i = np.mean(region)
    abs_intensities.append(abs_i)

max_intensity = max(abs_intensities)
rel_intensities = [val / max_intensity if max_intensity else 0 for val in abs_intensities]

with open(output_txt_file, "w") as f:
    f.write("Frame, Absolute_Intensity, Relative_Intensity\n")
    for i in range(frames):
        f.write(f"{i}, {abs_intensities[i]:.6f}, {rel_intensities[i]:.6f}\n")

print(f"Data saved to: {output_txt_file}")


start_frame = 0
end_frame = frames - 1

x = np.arange(start_frame, end_frame + 1)
abs_y = np.array(abs_intensities[start_frame:end_frame + 1])
rel_y = np.array(rel_intensities[start_frame:end_frame + 1])

norm = Normalize(vmin=x.min(), vmax=x.max())
cmap = cm.plasma

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(len(x) - 1):
    xs = [x[i], x[i + 1]]
    ys = [abs_y[i], abs_y[i + 1]]
    color = cmap(norm(x[i]))
    ax.plot(xs, ys, color=color, linewidth=2)
    ax.fill_between(xs, 0, ys, color=color, alpha=0.3)

for i in range(len(x) - 1):
    xs = [x[i], x[i + 1]]
    ys = [rel_y[i], rel_y[i + 1]]
    color = cmap(norm(x[i]))
    ax.plot(xs, ys, color=color, linewidth=2, linestyle='--')
    ax.fill_between(xs, 0, ys, color=color, alpha=0.15)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, max(abs_y.max(), rel_y.max()) * 1.1)
ax.set_xlabel("Frame Number")
ax.set_ylabel("Intensity")
ax.set_title("Spectrum with Color Gradient and Filled Area")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Frame Number (Wavelength color gradient)")

plt.tight_layout()

plot_file = output_txt_file.replace(".txt", "_colored_filled_spectrum.png")
plt.savefig(plot_file, dpi=300)
plt.show()

print(f"Colored filled spectrum plot saved to: {plot_file}")
