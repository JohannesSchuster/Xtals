import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz  # for Voigt function

# ---- Voigt profile model ----
def voigt(x, amplitude, x0, sigma, gamma, offset):
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    return offset + amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# ---- Load TIFF ----
tiff_path = "data/X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif"
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop, "Microcrystals")
os.makedirs(output_folder, exist_ok=True)
output_txt_file = os.path.join(output_folder, "grace1_voigt.txt")
plot_file = os.path.join(output_folder, "grace1_voigt_fit.png")

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

# ---- Initial guess for Voigt parameters ----
amplitude_init = abs_intensities.max() - abs_intensities.min()
x0_init = x_data[np.argmax(abs_intensities)]
sigma_init = 3.0
gamma_init = 1.0
offset_init = abs_intensities.min()
p0 = [amplitude_init, x0_init, sigma_init, gamma_init, offset_init]

# ---- Fit Voigt model ----
try:
    params, _ = curve_fit(voigt, x_data, abs_intensities, p0=p0)
    a_fit, x0_fit, sigma_fit, gamma_fit, offset_fit = params
    fit_success = True
except Exception as e:
    print("Voigt fitting failed:", e)
    fit_success = False

# ---- Save results ----
with open(output_txt_file, "w") as f:
    f.write("Frame, Absolute_Intensity\n")
    for i, val in enumerate(abs_intensities):
        f.write(f"{i}, {val:.6f}\n")

    if fit_success:
        f.write("\nVoigt Fit Parameters:\n")
        f.write(f"Amplitude: {a_fit:.6f}\n")
        f.write(f"Center (x0): {x0_fit:.6f}\n")
        f.write(f"Sigma (Gaussian width): {sigma_fit:.6f}\n")
        f.write(f"Gamma (Lorentzian width): {gamma_fit:.6f}\n")
        f.write(f"Offset: {offset_fit:.6f}\n")

print(f"Results saved to: {output_txt_file}")

# ---- Plotting ----
x_fit = np.linspace(x_data.min(), x_data.max(), 500)
y_fit = voigt(x_fit, *params) if fit_success else None

plt.figure(figsize=(8, 5))
plt.plot(x_data, abs_intensities, 'bo', label="Data")

if fit_success:
    plt.plot(x_fit, y_fit, 'r--', linewidth=2, label="Voigt Fit")
    plt.title("Voigt Fit of Intensity Over Time")
else:
    plt.title("Data (Voigt fit failed)")

plt.xlabel("Frame")
plt.ylabel("Absolute Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.show()

print(f"Plot saved to: {plot_file}")
