import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import IdentityTransform


import tifffile
import numpy as np

handle = tifffile.imread("data/data/2025-06-19_21.20.41_Xlzso-NO3-3_r1_70_000.tif.tif")

frames, height, width = handle.shape

arr = np.zeros((height, width))
for frame in handle:
    arr = np.add(arr, frame)

f = np.fft.fft2(arr)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 40*np.log(np.abs(fshift))

width, height = magnitude_spectrum.shape
ammount = 2000

flat_indices = np.argsort(magnitude_spectrum.ravel())[-ammount:]
coords = np.column_stack(np.unravel_index(flat_indices, magnitude_spectrum.shape))

fig, ax = plt.subplots()
ax.imshow(magnitude_spectrum, cmap='gray')

# Define the callback for mouse clicks
def onclick(event):
    # Capture the x, y coordinates of the click
    x, y = int(event.xdata), int(event.ydata)
    print(f"Clicked at: x={x}, y={y}")
    
    # Store the clicked coordinates
    selected_coords.append((x, y))
    
    # Draw a circle at the clicked location
    circle = patches.Circle((x, y), radius=20, transform=ax.transData,
                            linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(circle)
    plt.draw()  # Redraw the plot to update the circle

# Connect the click event to the function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Show the plot
plt.show()

# After plotting, use selected_coords in your calculation
print("Coordinates selected:", selected_coords)

for y, x in coords:
    circle = patches.Circle((x, y), radius=20,
                            transform=ax.transData,  # Position in data coords
                            linewidth=2, edgecolor='red', facecolor='none')
    # Adjust for fixed pixel radius using blended transform
    circle.set_transform(ax.transData + plt.matplotlib.transforms.ScaledTranslation(
        0, 0, ax.figure.dpi_scale_trans))  # To keep center in data coords
    #ax.add_patch(circle)
cx = width / 2
cy = height / 2
y, x = coords[1918.2,1854.8]

R = np.sqrt((x-cx)**2 + (y-cx)**2) 

pixel_scale = 0.7891 # Angs/pix

file = open("results.txt", "w")

for n, frame in enumerate(handle):
    f = np.fft.fft2(frame)    
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 40*np.log(np.abs(fshift))
    value = magnitude_spectrum[x][y]
    file.write(f"{n}, {x}, {y}, {value}\n")


print("did something else")





