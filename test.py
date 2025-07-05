import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import IdentityTransform
from skimage.feature import peak_local_max

import tifffile
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import os
import threading
from display import ImageDisplay
from display import ImageHandler
from peak_finder import PeakFinder, RectMask, CircleMask, PolyMask
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class State:
    handle: Optional[Any] = None
    filename: Optional[str] = None
    fig: Optional[Any] = None

@dataclass
class PeakFinderParams:
    ammount: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=600))
    cutoff: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.74))
    R: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=150))
    LINE: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=20))

# --- Widget logic ---
class PeakFinderWidget:
    def __init__(self, root):
        self.root = root
        self.root.title('Peak Finder Widget')
        self.peak_finder_params = PeakFinderParams()
        self.state = State()
        # removed cached_image, cached_coordinates, cached_ammount
        self.image_handler = ImageHandler()
        self.image_display = ImageDisplay(window_title='Image Display')
        self.peak_finder = PeakFinder()
        self.masks = []  # type: list
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _menubar(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load", command=self.load_file)
        filemenu.add_command(label="Save", command=self.save_file)
        filemenu.add_command(label="Save As", command=self.save_file_as)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=filemenu)
        aboutmenu = tk.Menu(menubar, tearoff=0)
        aboutmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="About", menu=aboutmenu)
        return menubar
    
    def _image_section(self, parent):
        # File input and browse button
        self.file_path_var = tk.StringVar()
        self.file_input = ttk.Entry(parent, textvariable=self.file_path_var, width=40)
        self.file_input.grid(row=0, column=0, columnspan=2, sticky='ew', padx=(2,2))
        self.file_input.bind('<Return>', lambda e: self.on_file_input())
        self.file_input.bind('<FocusOut>', lambda e: self.on_file_input())
        self.browse_btn = ttk.Button(parent, text='Browse', command=self.browse_file)
        self.browse_btn.grid(row=0, column=2, sticky='ew')
        
        # Image info labels
        ttk.Label(parent, text='Info:').grid(row=1, column=0, sticky='e')
        self.info_display = ttk.Label(parent, text='No image loaded')
        self.info_display.grid(row=1, column=1, sticky='w', columnspan=2)
        # A/px
        ttk.Label(parent, text='A/px:').grid(row=2, column=0, sticky='e')
        self.apx_var = tk.DoubleVar(value=1.0)
        self.apx_entry = ttk.Entry(parent, textvariable=self.apx_var)
        self.apx_entry.grid(row=2, column=1, sticky='nsew')
        # Sum checkbox
        self.display_sum = tk.BooleanVar(value=True)
        self.sum_checkbox = ttk.Checkbutton(parent, text='Sum', variable=self.display_sum, command=self.on_sum_toggle)
        self.sum_checkbox.grid(row=3, column=0, sticky='w', columnspan=3)
        # Frame selection (hidden if sum is checked)
        self.frame_section = ttk.Frame(parent)
        ttk.Label(self.frame_section, text='Frame:').grid(row=0, column=0, sticky='e')
        self.display_frame_idx = tk.IntVar(value=0)
        self.frame_idx_entry = ttk.Entry(self.frame_section, textvariable=self.display_frame_idx, width=6)
        self.frame_idx_entry.grid(row=0, column=1, sticky='nsew')
        self.frame_section.grid(row=4, column=0, columnspan=3, sticky='ew')
        # FFT checkbox
        self.display_fft = tk.BooleanVar(value=True)
        self.norm_checkbox = ttk.Checkbutton(parent, text='Calculate FFT', variable=self.display_fft)
        self.norm_checkbox.grid(row=5, column=1, sticky='w')
        # Show button
        self.show_image_btn = ttk.Button(parent, text='Show', command=self.update_image_display)
        self.show_image_btn.grid(row=6, column=0, columnspan=3, sticky='nsew', padx=(2,2))
        self.update_frame_section_visibility()

    def update_frame_section_visibility(self):
        if self.display_sum.get():
            self.frame_section.grid_remove()
        else:
            self.frame_section.grid()

    def on_sum_toggle(self):
        self.update_frame_section_visibility()
        # Optionally reset frame index
        if self.display_sum.get():
            self.display_frame_idx.set(0)

    def _peak_finder_controls(self, parent):
        # Settings group
        settings_frame = ttk.LabelFrame(parent, text='Settings', padding=(5, 5))
        settings_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        ttk.Label(settings_frame, text='Ammount').grid(row=0, column=0)
        self.ammount_slider = ttk.Scale(settings_frame, from_=1, to=2000, orient='horizontal', variable=self.peak_finder_params.ammount)
        self.ammount_slider.grid(row=0, column=1, sticky='ew')
        self.ammount_label = ttk.Label(settings_frame, textvariable=self.peak_finder_params.ammount)
        self.ammount_label.grid(row=0, column=2)
        self.ammount_slider.config(command=lambda val: self.peak_finder_params.ammount.set(int(float(val))))
        ttk.Label(settings_frame, text='Cutoff').grid(row=1, column=0)
        cutoff_entry = ttk.Entry(settings_frame, textvariable=self.peak_finder_params.cutoff)
        cutoff_entry.grid(row=1, column=1)
        ttk.Label(settings_frame, text='1.000').grid(row=1, column=2)
        ttk.Label(settings_frame, text='Mask Radius').grid(row=2, column=0)
        r_entry = ttk.Entry(settings_frame, textvariable=self.peak_finder_params.R)
        r_entry.grid(row=2, column=1)
        ttk.Label(settings_frame, text='Mask Line').grid(row=3, column=0)
        line_entry = ttk.Entry(settings_frame, textvariable=self.peak_finder_params.LINE)
        line_entry.grid(row=3, column=1)
        calc_btn = ttk.Button(settings_frame, text='Calculate', command=self.find_peaks)
        calc_btn.grid(row=4, column=0, columnspan=2, pady=5)
        cont_btn = ttk.Button(settings_frame, text='Continue', command=self.continue_fn)
        cont_btn.grid(row=5, column=0, columnspan=2, pady=5)
        # Display group
        display_frame = ttk.LabelFrame(parent, text='Display', padding=(5, 5))
        display_frame.grid(row=1, column=0, columnspan=3, sticky='ew')
        ttk.Label(display_frame, text='Mode:').grid(row=0, column=0)
        self.display_mode = tk.StringVar(value='Circle')
        display_choices = ['Circle', 'Extraction Box', 'Point']
        display_menu = ttk.OptionMenu(display_frame, self.display_mode, display_choices[0], *display_choices)
        display_menu.grid(row=0, column=1, sticky='ew')
        # Color selector
        ttk.Label(display_frame, text='Color:').grid(row=1, column=0)
        self.display_color = tk.StringVar(value='red')
        color_choices = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'black', 'white']
        color_menu = ttk.OptionMenu(display_frame, self.display_color, color_choices[0], *color_choices)
        color_menu.grid(row=1, column=1, sticky='ew')
        # Size slider
        ttk.Label(display_frame, text='Size:').grid(row=2, column=0)
        self.display_size = tk.IntVar(value=10)
        size_slider = ttk.Scale(display_frame, from_=2, to=50, orient='horizontal', variable=self.display_size)
        size_slider.grid(row=2, column=1, sticky='ew')
        size_slider.config(command=lambda val: self.display_size.set(int(float(val))))
        size_label = ttk.Label(display_frame, textvariable=self.display_size)
        size_label.grid(row=2, column=2, sticky='w')

    def _build_layout(self):
        # Menu bar
        self.root.config(menu=self._menubar())

        # Left column: image info
        left_outer = ttk.LabelFrame(self.root, text='Image', padding=(10, 5))
        left_outer.grid(row=0, column=0, sticky='n', padx=10, pady=10)
        self._image_section(left_outer)

        # Right column: peak finder controls
        right_outer = ttk.LabelFrame(self.root, text='Peak Finder', padding=(10, 5))
        right_outer.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        self._peak_finder_controls(right_outer)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('TIFF files', '*.tif;*.tiff')])
        if file_path:
            self.file_path_var.set(file_path)
            self.on_file_input()

    def on_file_input(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.isfile(file_path):
            return
        self.info_display.config(text='Loading...')
        self.load_file(file_path)

    def load_file(self, file_path=None):
        def do_load(file_path):
            handle = tifffile.imread(file_path)
            self.state.handle = handle
            self.state.filename = os.path.basename(file_path)
            self.image_handler.set_handle(handle)
            height, width = handle.shape[-2], handle.shape[-1]
            frames = handle.shape[0] if handle.ndim == 3 else 1
            def update_gui():
                self.file_path_var.set(file_path)
                self.info_display.config(text=f"{width}x{height} @ {frames} Frames")
                self.display_frame_idx.set(0)
                self.update_frame_section_visibility()
            self.root.after(0, update_gui)
        if file_path is None:
            file_path = self.file_path_var.get()
            if not file_path or not os.path.isfile(file_path):
                return
        self.file_path_var.set(file_path)
        threading.Thread(target=do_load, args=(file_path,), daemon=True).start()

    def find_peaks(self):
        def do_find_peaks():
            if self.state.handle is None:
                return
            show_sum = self.display_sum.get()
            do_fft = self.display_fft.get()
            idx = self.display_frame_idx.get()
            image = self.image_handler.get_image(show_sum, do_fft, idx)
            ammount = self.peak_finder_params.ammount.get()
            cutoff = self.peak_finder_params.cutoff.get()
            R = self.peak_finder_params.R.get()
            LINE = self.peak_finder_params.LINE.get()
            # Build masks
            height, width = image.shape
            center_x = width // 2
            center_y = height // 2
            # Circle mask for R
            circle_mask = CircleMask(center_x, center_y, R)
            # Cross mask for LINE
            poly_points = [
                (center_x - LINE//2, center_y), (center_x + LINE//2, center_y),
                (center_x, center_y - LINE//2), (center_x, center_y + LINE//2)
            ]
            poly_mask = PolyMask(poly_points)
            self.masks = [circle_mask, poly_mask]
            # Use PeakFinder
            self.peak_finder.clear_masks()
            for mask in self.masks:
                self.peak_finder.add_mask(mask)
            self.peak_finder.min_distance = 3  # or expose as UI param
            self.peak_finder.threshold_rel = cutoff
            coordinates = self.peak_finder.find_peaks(image=image, window=3)
            if len(coordinates) > ammount:
                coordinates = coordinates[:ammount]
            self.peak_finder.clear_masks()
            self.root.after(0, lambda: self.update_plot(coordinates))

        threading.Thread(target=do_find_peaks, daemon=True).start()

    def update_plot(self, coordinates=[]):
        show_sum = self.display_sum.get()
        do_fft = self.display_fft.get()
        idx = self.display_frame_idx.get()
        image = self.image_handler.get_image(show_sum, do_fft, idx)
        if image is None:
            return
        mode = self.display_mode.get()
        color = self.display_color.get()
        size = self.display_size.get()
        title = f"{'Sum' if self.display_sum.get() else f'Frame {self.display_frame_idx.get()}'} - {self.state.filename or 'Untitled'}"
        self.image_display.display_image(
            image=image,
            mode=mode,
            color=color,
            size=size,
            title=title,
            coordinates=coordinates,
        )
        if len(coordinates) != 0:
            self.ammount_label.config(text=f"{len(coordinates)}")

    def update_image_display(self):
        self.update_plot([])

    def save_file(): pass
    def save_file_as(self): pass

    def continue_fn(self): pass

    def show_about(self):
        tk.messagebox.showinfo(
            "About",
            "Peak Finder Widget\n\nby Johannes Schuster\nUniversity of Regensburg\n\n04.07.2025"
        )

    def on_closing(self):
        self.image_display.close()
        self.root.destroy()

def run_gui():
    root = tk.Tk()
    PeakFinderWidget(root)
    root.mainloop()

if __name__ == '__main__':
    run_gui()




