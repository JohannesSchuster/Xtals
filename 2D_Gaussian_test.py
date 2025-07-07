import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# Define the region I want to analyse 

def extract_peak_region(image, x, y, size=30):
    half = size // 2
    y_min = max(0, y - half)
    y_max = min(image.shape[0], y + half + 1)
    x_min = max(0, x - half)
    x_max = min(image.shape[1], x + half + 1)
    return image[y_min:y_max, x_min:x_max]

def find_brightest_near(image, x0, y0, search_radius=5):
    y0, x0 = int(y0), int(x0)
    y_min = max(0, y0 - search_radius)
    y_max = min(image.shape[0], y0 + search_radius + 1)
    x_min = max(0, x0 - search_radius)
    x_max = min(image.shape[1], x0 + search_radius + 1)
    sub_image = image[y_min:y_max, x_min:x_max]
    max_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)
    brightest_y = y_min + max_idx[0]
    brightest_x = x_min + max_idx[1]
    return brightest_x, brightest_y

# to fit the 2D-Gaussian 

def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo, yo = float(xo), float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp(- (a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2)))
    return g.ravel()

def fit_gaussian_to_region(region):
    x = np.linspace(0, region.shape[1] - 1, region.shape[1])
    y = np.linspace(0, region.shape[0] - 1, region.shape[0])
    x, y = np.meshgrid(x, y)
    initial_guess = (region.max(), region.shape[1]/2, region.shape[0]/2, 3, 3, 0, np.min(region))
    try:
        popt, _ = curve_fit(twoD_Gaussian, (x, y), region.ravel(), p0=initial_guess)
        return popt 
    except RuntimeError:
        return None

# To analyse the data 

def analyze_brightest_region_over_frames(image_stack, picked_coord, box_size=30, search_radius=5):
    results = []
    frames = image_stack.shape[0]
    for frame_idx in range(frames):
        image = image_stack[frame_idx]
        x_bright, y_bright = find_brightest_near(image, *picked_coord, search_radius)
        region = extract_peak_region(image, x_bright, y_bright, box_size)
        if region.shape[0] < box_size or region.shape[1] < box_size:
            continue 
        popt = fit_gaussian_to_region(region)
        if popt is not None:
            amplitude = popt[0]
            results.append((frame_idx, x_bright, y_bright, amplitude))
    return pd.DataFrame(results, columns=["frame", "x", "y", "amplitude"])

def plot_intensity_over_time(df, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(df["frame"], df["amplitude"], marker='o', linewidth=2)
    plt.title("Gaussian Amplitude (Intensity) Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude (Real Intensity)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# Here I can change the data (image, coordinates, even the radius)

def main():
    # ---- INPUTS ----
    tiff_path = "data/2025-06-19_21.58.23_Xlzso-NO3-3_r1_78_000.tif"
    picked_coord = (750, 800)  
    box_size = 30
    search_radius = 1

    
    image_stack = tifffile.imread(tiff_path)

    
    df = analyze_brightest_region_over_frames(image_stack, picked_coord, box_size, search_radius)

    # ---- OUTPUT ----
    desktop = os.path.join(os.path.expanduser("~"), "Desktop", "Microcrystals")
    os.makedirs(desktop, exist_ok=True)
    data_file = os.path.join(desktop, "intensity_over_time1.txt")
    plot_file = os.path.join(desktop, "intensity_over_time1.png")

    df.to_csv(data_file, index=False)
    plot_intensity_over_time(df, plot_file)

    print(f"Results saved to:\n{data_file}\n{plot_file}")

if __name__ == "__main__":
    main()
