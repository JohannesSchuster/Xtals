import os
import numpy as np
import tifffile
from scipy.ndimage import maximum_filter, label, find_objects
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# --- Your 2D Gaussian model ---
def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()

# --- Peak detection ---
def detect_peaks(image, neighborhood_size=5, threshold_rel=0.5):
    max_filtered = maximum_filter(image, size=neighborhood_size)
    local_max = (image == max_filtered)
    threshold = image.max() * threshold_rel
    detected_peaks = local_max & (image > threshold)
    labeled, num_features = label(detected_peaks)
    slices = find_objects(labeled)

    peak_coords = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) // 2
        y_center = (dy.start + dy.stop - 1) // 2
        peak_coords.append((x_center, y_center))

    return peak_coords

# --- Fit function ---
def fit_2d_gaussian(region):
    height, width = region.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)

    initial_guess = (
        region.max(),  # amplitude
        width // 2,    # xo
        height // 2,   # yo
        3,             # sigma_x
        3,             # sigma_y
        np.median(region)  # offset (background)
    )
    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), region.ravel(), p0=initial_guess)
        return popt, pcov
    except RuntimeError:
        # Fit failed
        return None, None

# --- Diameter calculation ---
def calculate_diameter(region, rel_threshold=0.67):
    threshold = region.max() * rel_threshold
    mask = region > threshold
    area_px = np.sum(mask)
    diameter = 2 * np.sqrt(area_px / np.pi)
    return diameter

# --- Placeholder for resolution calculation ---
def calculate_resolution(x, y):
    # TODO: Replace this function with actual resolution calculation logic
    # For example, resolution might be related to peak position or metadata
    # Here we assign dummy resolution based on distance from center (just as example)
    center_x, center_y = 512, 512  # Adjust according to your image size
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Inverse relationship for demo: resolution in Angstrom^-1
    resolution = 1 / (1 + dist / 100)  # example function, change as needed
    return resolution

def main():
    tiff_path = "data/2025-06-19_21.53.27_Xlzso-NO3-3_r1_77_000.tif"

    handle = tifffile.imread(tiff_path)
    frames, height, width = handle.shape

    box_size = 15
    half_box = box_size // 2

    results = []

    for frame_index in range(frames):
        frame_image = handle[frame_index]

        # Detect peaks automatically
        peaks = detect_peaks(frame_image, neighborhood_size=5, threshold_rel=0.5)

        for (x_peak, y_peak) in peaks:
            # Define box boundaries with edge checking
            x_start = max(x_peak - half_box, 0)
            x_end = min(x_peak + half_box + 1, width)
            y_start = max(y_peak - half_box, 0)
            y_end = min(y_peak + half_box + 1, height)
            region = frame_image[y_start:y_end, x_start:x_end]

            diameter = calculate_diameter(region, rel_threshold=0.67)
            popt, pcov = fit_2d_gaussian(region)
            if popt is not None:
                amplitude = popt[0]
                fitted_x = x_start + popt[1]
                fitted_y = y_start + popt[2]
                resolution = calculate_resolution(fitted_x, fitted_y)

                results.append({
                    "frame": frame_index,
                    "x": fitted_x,
                    "y": fitted_y,
                    "amplitude": amplitude,
                    "sigma_x": popt[3],
                    "sigma_y": popt[4],
                    "offset": popt[5],
                    "diameter": diameter,
                    "resolution": resolution
                })

    # Save results to CSV
    df = pd.DataFrame(results)
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Microcrystals")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "joie.csv")
    df.to_csv(output_file, index=False)
    print(f"Fit joie saved to: {output_file}")

    # Bin resolution and calculate statistics per frame
    bins = [0, 1/4, 1/3.5, 1/3, 1/2.5, 1/2, 10]  # Adjust bins to your needs
    labels = ["<1/4", "1/4-1/3.5", "1/3.5-1/3", "1/3-1/2.5", "1/2.5-1/2", ">1/2"]
    df['resolution_bin'] = pd.cut(df['resolution'], bins=bins, labels=labels, include_lowest=True)

    stats = df.groupby(['frame', 'resolution_bin'])['amplitude'].agg(['mean', 'std']).reset_index()

    # Save stats
    stats_file = os.path.join(output_folder, "resolution_stats.csv")
    stats.to_csv(stats_file, index=False)
    print(f"Resolution binned statistics saved to: {stats_file}")

    # Plot stats for a specific frame (example: frame 0)
    plot_resolution_stats(stats, frame_number=0)

def plot_resolution_stats(stats_df, frame_number):
    df_frame = stats_df[stats_df['frame'] == frame_number]
    plt.errorbar(df_frame['resolution_bin'], df_frame['mean'], yerr=df_frame['std'], fmt='o-', capsize=5)
    plt.title(f"Average Amplitude vs Resolution Bin (Frame {frame_number})")
    plt.xlabel("Resolution bin (Å⁻¹)")
    plt.ylabel("Average amplitude")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
