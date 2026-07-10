import tifffile
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import threading
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import concurrent.futures
import torch

# Importing custom modules
from display import Image, ImageDisplay, ImageHandler, FFTImageHandler
from display import HistogramDisplay
from peak_finder import PeakFinder, Mask, RectMask, CircleMask, PolyMask
from timer import Timer, timed
from gaussfitter import gaussfit



@dataclass
class State:
    filename: Optional[str] = None
    black_level: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.0))
    white_level: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=1.0))
    sigma_var: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=1.0))
    gamma_var: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=1.0))

@dataclass
class PeakFinderParams:
    ammount: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=50))
    cutoff: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.95))
    R: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=150))
    LINE: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=20))

class PeakFinderWidget:
    def __init__(self, root):
        self.root = root
        self.root.title('Peak Finder Widget')
        self.state = State()
        self.image_handler = ImageHandler()
        self.fft_image_cache = None
        self.image_display = ImageDisplay()
        self.hist_display = HistogramDisplay()
        self.peak_finder = PeakFinder()
        self.peak_finder_params = PeakFinderParams()
        self._current_file_path = None  # Track currently loaded file
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.image_display.callback("WM_DELETE_WINDOW", self.on_closing_display)
        self.hist_display.callback("WM_DELETE_WINDOW", self.on_closing_hist)

    def _menubar(self):
        menubar = tk.Menu(self.root)
        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load", command=self.browse_file)
        filemenu.add_command(label="Save", command=self.save_file)
        filemenu.add_command(label="Save As", command=self.save_file_as)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=filemenu)

        # Image menu
        imagemenu = tk.Menu(menubar, tearoff=0)
        self.precalc_sum_var = tk.BooleanVar(value=False)
        self.precalc_fft_var = tk.BooleanVar(value=False)
        imagemenu.add_checkbutton(label="Precalculate Sum", variable=self.precalc_sum_var, command=self.on_precalc_sum)
        imagemenu.add_checkbutton(label="Precalculate FFT", variable=self.precalc_fft_var, command=self.on_precalc_ffts)
        menubar.add_cascade(label="Image", menu=imagemenu)

        # About menu
        aboutmenu = tk.Menu(menubar, tearoff=0)
        aboutmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="About", menu=aboutmenu)
        return menubar

    def _image_section(self, parent):
        # Use vertical layout for the whole section
        outer = ttk.Frame(parent)
        outer.pack(fill='both', expand=True)

        # File input and browse button (horizontal)
        file_row = ttk.Frame(outer)
        file_row.pack(fill='x', pady=2)
        file_label = ttk.Label(file_row, text='')  # Empty label for alignment
        file_label.pack(side='left', padx=(0, 4))
        self.file_path_var = tk.StringVar()
        self.file_input = ttk.Entry(file_row, textvariable=self.file_path_var, width=40)
        self.file_input.pack(side='left', fill='x', expand=True, padx=(0, 2))
        self.file_input.bind('<Return>', lambda e: self.load_file())
        self.file_input.bind('<FocusOut>', lambda e: self.load_file())
        self.browse_btn = ttk.Button(file_row, text='Browse', command=self.browse_file)
        self.browse_btn.pack(side='left')

        # Info row (horizontal)
        info_row = ttk.Frame(outer)
        info_row.pack(fill='x', pady=2)
        info_label = ttk.Label(info_row, text='Info:', anchor='e', width=9)
        info_label.pack(side='left')
        self.info_display = ttk.Label(info_row, text='No image loaded')
        self.info_display.pack(side='left', padx=(4,0))

        # A/px row (horizontal)
        apx_row = ttk.Frame(outer)
        apx_row.pack(fill='x', pady=2)
        apx_label = ttk.Label(apx_row, text='A/px:', anchor='e', width=9)
        apx_label.pack(side='left')
        self.apx_var = tk.DoubleVar(value=1.0)
        self.apx_entry = ttk.Entry(apx_row, textvariable=self.apx_var, width=9)
        self.apx_entry.pack(side='left', padx=(4,0))

        # Frame selector and sum checkbox (horizontal)
        frame_row = ttk.Frame(outer)
        frame_row.pack(fill='x', pady=2)
        frame_label = ttk.Label(frame_row, text='Frame:', anchor='e', width=9)
        frame_label.pack(side='left')
        self.display_frame_idx = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(frame_row, from_=0, to=0, orient='horizontal', variable=self.display_frame_idx)
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=(4,2))
        self.frame_slider.config(command=lambda val: self.display_frame_idx.set(int(float(val))))
        self.frame_idx_label = ttk.Label(frame_row, textvariable=self.display_frame_idx, width=4)
        self.frame_idx_label.pack(side='left', padx=(2,2))
        self.display_sum = tk.BooleanVar(value=True)
        self.sum_checkbox = ttk.Checkbutton(frame_row, text='Sum', variable=self.display_sum, command=self.on_sum_toggle)
        self.sum_checkbox.pack(side='left')
        self.update_frame_slider_state()

        # Calculate/FFT row (horizontal)
        calc_row = ttk.Frame(outer)
        calc_row.pack(fill='x', pady=2)
        calc_label = ttk.Label(calc_row, text='Calculate:', anchor='e', width=9)
        calc_label.pack(side='left')
        
        self.display_fft = tk.BooleanVar(value=True)
        self.norm_checkbox = ttk.Checkbutton(calc_row, text='FFT', variable=self.display_fft)
        self.norm_checkbox.pack(side='left', padx=(4,0))

        # Image display frame
        self.display_img_frame = ttk.LabelFrame(parent, text='Display')
        self.display_img_frame.pack(fill='x', pady=(10, 0))
        self.display_img_frame.grid_columnconfigure(2, weight=1)
        # black level
        ttk.Label(self.display_img_frame, text="Black [0-1] =").grid(row=0, column=0, sticky='e')
        self.black_level_entry = ttk.Entry(self.display_img_frame, textvariable=self.state.black_level, width=10) 
        self.black_level_entry.grid(row=0, column=1, sticky='e')
        # white level
        ttk.Label(self.display_img_frame, text="White [0-1] =").grid(row=1, column=0, sticky='e')
        self.white_level_entry = ttk.Entry(self.display_img_frame, textvariable=self.state.white_level, width=10) 
        self.white_level_entry.grid(row=1, column=1, sticky='e')
        # sigma
        ttk.Label(self.display_img_frame, text="Sigma =").grid(row=0, column=2, sticky='e')
        self.sigma_entry = ttk.Entry(self.display_img_frame, textvariable=self.state.sigma_var, width=10)
        self.sigma_entry.grid(row=0, column=3, sticky='e')
        # gamma
        ttk.Label(self.display_img_frame, text="Gamma =").grid(row=1, column=2, sticky='e')
        self.gamma_entry = ttk.Entry(self.display_img_frame, textvariable=self.state.gamma_var, width=10)
        self.gamma_entry.grid(row=1, column=3, sticky='e')
        # Refresh button
        self.show_image_display = False
        def display_button_command():
            if self.show_image_display:
                self.display_refresh_button.config(text='Refresh...')
            else:
                self.display_refresh_button.config(text='Show...')
            self.show_image_display = True
            self.refresh_display()
        self.display_refresh_button = ttk.Button(self.display_img_frame, text='Show', command=display_button_command)
        self.display_refresh_button.grid(row=2, column=1, columnspan=2, sticky='e', pady=(4, 0))
        self.auto_refresh_image_display = tk.BooleanVar(value=True)
        self.auto_refresh_image_checkbox = ttk.Checkbutton(self.display_img_frame, text='Auto Refresh', variable=self.auto_refresh_image_display)
        self.auto_refresh_image_checkbox.grid(row=2, column=3, sticky='w', pady=(4, 0))

        # Histogram display frame
        self.display_hist_frame = ttk.LabelFrame(parent, text='Histogram')
        self.display_hist_frame.pack(fill='x', pady=(10, 0))
        self.display_hist_frame.grid_columnconfigure(2, weight=1)
        # X Axis selection
        ttk.Label(self.display_hist_frame, text='X Axis:').grid(row=0, column=0, sticky='w')
        self.hist_x_axis = tk.StringVar(value='linear')
        x_axis_menu = ttk.OptionMenu(self.display_hist_frame, self.hist_x_axis, 'linear', 'linear', 'log')
        x_axis_menu.grid(row=0, column=1, sticky='w')
        # Y Axis selection
        ttk.Label(self.display_hist_frame, text='Y Axis:').grid(row=1, column=0, sticky='w')
        self.hist_y_axis = tk.StringVar(value='linear')
        y_axis_menu = ttk.OptionMenu(self.display_hist_frame, self.hist_y_axis, 'linear', 'linear', 'log')
        y_axis_menu.grid(row=1, column=1, sticky='w')
        # Update histogram on change
        self.hist_x_axis.trace_add('write', self.auto_update_histogram_display)
        self.hist_y_axis.trace_add('write', self.auto_update_histogram_display)

        # Refresh button
        self.show_hist_display = False
        def hist_button_command():
            if self.show_hist_display:
                self.hist_refresh_button.config(text='Refresh...')
            else:
                self.hist_refresh_button.config(text='Show...')
            self.show_hist_display = True
            self.refresh_histogram()
        self.hist_refresh_button = ttk.Button(self.display_hist_frame, text='Show', command=hist_button_command)
        self.hist_refresh_button.grid(row=2, column=1, columnspan=2, sticky='e', pady=(4, 0))
        self.auto_refresh_hist_display = tk.BooleanVar(value=True)
        self.auto_refresh_image_checkbox = ttk.Checkbutton(self.display_hist_frame, text='Auto Refresh', variable=self.auto_refresh_hist_display)
        self.auto_refresh_image_checkbox.grid(row=2, column=3, sticky='w', pady=(4, 0))

        # Bind events for auto-redraw
        self.frame_slider.bind('<ButtonRelease-1>', self.auto_update_displays)
        self.frame_slider.bind('<KeyRelease>', self.auto_update_displays)
        self.black_level_entry.bind('<KeyRelease>', self.auto_update_displays)
        self.white_level_entry.bind('<KeyRelease>', self.auto_update_displays)
        self.sigma_entry.bind('<KeyRelease>', self.auto_update_displays)
        self.gamma_entry.bind('<KeyRelease>', self.auto_update_displays)

    def update_frame_slider_state(self):
        # Disable slider if sum is checked, enable otherwise
        if self.display_sum.get():
            self.frame_slider.state(['disabled'])
        else:
            self.frame_slider.state(['!disabled'])

    def on_sum_toggle(self):
        self.update_frame_slider_state()
        # Optionally reset frame index
        if self.display_sum.get():
            self.display_frame_idx.set(0)
        self.auto_update_displays()

    def _peak_finder_controls(self, parent):
        # Settings group
        settings_frame = ttk.LabelFrame(parent, text='Settings', padding=(5, 5))
        settings_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))

        # Ammount row
        ammount_row = ttk.Frame(settings_frame)
        ammount_row.pack(fill='x', pady=2)
        # Label for current value (left)
        self.ammount_label = ttk.Label(ammount_row, text='Ammount:', width=12, anchor='e')
        self.ammount_label.pack(side='left', padx=(0, 4))
        # Slider for selected amount (center)
        self.ammount_slider = ttk.Scale(ammount_row, from_=1, to=100, orient='horizontal', variable=self.peak_finder_params.ammount)
        self.ammount_slider.pack(side='left', fill='x', expand=True)
        # Label for slider value (right of slider)
        self.ammount_slider_value = ttk.Label(ammount_row, textvariable=self.peak_finder_params.ammount, width=3)
        self.ammount_slider_value.pack(side='left', padx=(4,0))
        # Entry for slider maximum (rightmost)
        self.ammount_slider_max = tk.IntVar(value=100)
        max_entry = ttk.Entry(ammount_row, textvariable=self.ammount_slider_max, width=4)
        max_entry.pack(side='left', padx=(4, 0))
        # Update slider max only on defocus or Enter
        max_entry.bind('<FocusOut>', self.update_slider_max)
        max_entry.bind('<Return>', self.update_slider_max)
        # Update label when slider changes
        self.ammount_slider.config(command=lambda val: self.peak_finder_params.ammount.set(int(float(val))))

        # Cutoff row
        cutoff_row = ttk.Frame(settings_frame)
        cutoff_row.pack(fill='x', pady=2)
        ttk.Label(cutoff_row, text='Cutoff:', anchor='e', width=12).pack(side='left')
        self.cutoff_entry = ttk.Entry(cutoff_row, textvariable=self.peak_finder_params.cutoff)
        self.cutoff_entry.pack(side='left', fill='x', expand=True)

        # Mask Radius row
        r_row = ttk.Frame(settings_frame)
        r_row.pack(fill='x', pady=2)
        ttk.Label(r_row, text='Mask Radius:', anchor='e', width=12).pack(side='left')
        r_entry = ttk.Entry(r_row, textvariable=self.peak_finder_params.R)
        r_entry.pack(side='left', fill='x', expand=True)

        # Mask Line row
        line_row = ttk.Frame(settings_frame)
        line_row.pack(fill='x', pady=2)
        ttk.Label(line_row, text='Mask Line:', anchor='e', width=12).pack(side='left')
        line_entry = ttk.Entry(line_row, textvariable=self.peak_finder_params.LINE)
        line_entry.pack(side='left', fill='x', expand=True)

        # Buttons row (horizontal)
        btn_row = ttk.Frame(settings_frame)
        btn_row.pack(fill='x', pady=(6,0))
        self.calc_btn = ttk.Button(btn_row, text='Calculate', command=self.find_peaks)
        self.calc_btn.pack(side='left', fill='x', expand=True, padx=(0,4))
        self.cont_btn = ttk.Button(btn_row, text='Continue', command=self.continue_fn)
        self.cont_btn.pack(side='left', fill='x', expand=True)

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
        self.size_slider = ttk.Scale(display_frame, from_=2, to=50, orient='horizontal', variable=self.display_size)
        self.size_slider.grid(row=2, column=1, sticky='ew')
        self.size_slider.config(command=self.set_coordinate_size)
        size_label = ttk.Label(display_frame, textvariable=self.display_size)
        size_label.grid(row=2, column=2, sticky='w')

        # Add checkboxes for toggling coordinates and masks display
        self.show_coordinates = tk.BooleanVar(value=True)
        self.show_masks = tk.BooleanVar(value=True)
        coords_checkbox = ttk.Checkbutton(display_frame, text='Show Coordinates', variable=self.show_coordinates)
        coords_checkbox.grid(row=0, column=3, sticky='w', pady=(4,0))
        masks_checkbox = ttk.Checkbutton(display_frame, text='Show Masks', variable=self.show_masks)
        masks_checkbox.grid(row=1, column=3, sticky='w', pady=(4,0))

        # Trace changes to update image display
        self.cutoff_entry.bind('<KeyRelease>', self.auto_update_histogram_display)
        self.display_mode.trace_add('write', self.auto_update_image_display)
        self.display_color.trace_add('write', self.auto_update_image_display)
        self.show_coordinates.trace_add('write', self.auto_update_image_display)
        self.show_masks.trace_add('write', self.auto_update_image_display)
        self.ammount_slider.bind('<ButtonRelease-1>', self.auto_update_image_display)
        self.ammount_slider.bind('<KeyRelease>', self.auto_update_image_display)
        self.size_slider.bind('<ButtonRelease-1>', self.auto_update_image_display)
        self.size_slider.bind('<KeyRelease>', self.auto_update_image_display)

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
            self.load_file()

    def load_file(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.isfile(file_path):
            return
        
        # Check if this is a different file than currently loaded
        current_file = getattr(self, '_current_file_path', None)
        is_new_file = current_file != file_path
        
        if is_new_file:
            # Clear peak finder cache for new file
            old_peak_count = len(self.peak_finder.cache.coordinates) if hasattr(self.peak_finder.cache, 'coordinates') else 0
            self.peak_finder.cache.coordinates = torch.empty((0, 2))
            self.peak_finder.clear_masks()
            
            # Handle display windows - mark as inactive and create new ones
            try:
                if hasattr(self, 'image_display') and self.image_display:
                    # Just set the flag to false and recreate - let the old one cleanup naturally
                    self.show_image_display = False
                    old_display = self.image_display
                    self.image_display = ImageDisplay()
                    self.image_display.callback("WM_DELETE_WINDOW", self.on_closing_display)
                    self.display_refresh_button.config(text='Show')
                    # Try to close the old one, but don't fail if it errors
                    try:
                        old_display.close()
                    except Exception:
                        pass  # Ignore errors from closing old display
            except Exception as e:
                print(f"Warning: Error handling image display: {e}")
                # Ensure we have a working display
                self.image_display = ImageDisplay()
                self.image_display.callback("WM_DELETE_WINDOW", self.on_closing_display)
                self.show_image_display = False
                self.display_refresh_button.config(text='Show')
                
            try:
                if hasattr(self, 'hist_display') and self.hist_display:
                    # Just set the flag to false and recreate - let the old one cleanup naturally
                    self.show_hist_display = False
                    old_display = self.hist_display
                    self.hist_display = HistogramDisplay()
                    self.hist_display.callback("WM_DELETE_WINDOW", self.on_closing_hist)
                    self.hist_refresh_button.config(text='Show')
                    # Try to close the old one, but don't fail if it errors
                    try:
                        old_display.close()
                    except Exception:
                        pass  # Ignore errors from closing old display
            except Exception as e:
                print(f"Warning: Error handling histogram display: {e}")
                # Ensure we have a working display
                self.hist_display = HistogramDisplay()
                self.hist_display.callback("WM_DELETE_WINDOW", self.on_closing_hist)
                self.show_hist_display = False
                self.hist_refresh_button.config(text='Show')
            
            # Clear FFT cache
            self.fft_image_cache = None
            
            if old_peak_count > 0:
                print(f"Loading new file: {os.path.basename(file_path)} - Cleared {old_peak_count} peaks from cache")
            else:
                print(f"Loading new file: {os.path.basename(file_path)}")
            
        self._current_file_path = file_path
        self.info_display.config(text='Loading...')

        def do_load(file_path):
            handle = torch.from_numpy(tifffile.imread(file_path))
            self.state.filename = os.path.basename(file_path)
            self.image_handler.set_handle(handle)
            height, width = handle.shape[-2], handle.shape[-1]
            frames = handle.shape[0] if handle.ndim == 3 else 1
            def update_gui():
                self.file_path_var.set(file_path)
                self.info_display.config(text=f"{width}x{height} @ {frames} Frames")
                self.display_frame_idx.set(0)
                self.frame_slider.config(from_=0, to=max(frames-1, 0))
                self.update_frame_slider_state()
            self.root.after(0, update_gui)
        if file_path is None:
            file_path = self.file_path_var.get()
            if not file_path or not os.path.isfile(file_path):
                return
        self.file_path_var.set(file_path)
        threading.Thread(target=do_load, args=(file_path,), daemon=True).start()

    def find_peaks(self):
        @timed
        def target():
            image = self.get_image()
            if image is not None:
                
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
                rectMask1 = RectMask(center_x - LINE//2, 0, LINE, height)
                rectMask2 = RectMask(0, center_y - LINE//2, width, LINE)
                # Use PeakFinder
                self.peak_finder.clear_masks()
                self.peak_finder.add_mask(circle_mask)
                self.peak_finder.add_mask(rectMask1)
                self.peak_finder.add_mask(rectMask2)
                self.peak_finder.threshold_abs = cutoff
                self.peak_finder.find_peaks(image=image.torch())
                self.root.after(0, self.auto_update_image_display, image, self.get_title())

            self.root.after(0, self.reset_peak_finder_buttons)

        self.calc_btn.config(state='disabled')
        self.calc_btn.config(text='Calculating...')
        self.cont_btn.config(state='disabled')
        threading.Thread(target=target, daemon=True).start()

    def get_image(self) -> Image:
        if self.image_handler.handle is None:
            return None
        show_sum = self.display_sum.get()
        do_fft = self.display_fft.get()
        idx = self.display_frame_idx.get()
        img = None
        if do_fft and self.fft_image_cache is None:
            self.fft_image_cache = FFTImageHandler()
            self.fft_image_cache.set_handle(self.image_handler.handle)
        if do_fft:
            if show_sum: img = self.fft_image_cache.get_sum()
            else: img = self.fft_image_cache.get_frame(idx)
        else:
            if show_sum: img = self.image_handler.get_sum()
            else: img = self.image_handler.get_frame(idx)
        return img

    def apply_image_transformations(self, image: Image) -> Image:
        # Get sigma and gamma
        sigma = self.state.sigma_var.get() 
        gamma = self.state.gamma_var.get() 
        black = self.state.black_level.get()
        white = self.state.white_level.get()
        
        # Apply transformations
        image = image.rescale(sigma=sigma, gamma=gamma)
        image = image.remap(min_val=black, max_val=white)
        return image
    
    def get_title(self) -> str:
        fft = "FFT: " if self.display_fft.get() else ""
        filename = self.state.filename or 'Untitled'
        idx = "Sum" if self.display_sum.get() else f"Frame {self.display_frame_idx.get()}"
        return f"{fft}{idx} - {filename}"

    def refresh_display(self, image: Optional[Image] = None, title: Optional[str] = None):
        def target(image: Image, title: str):
            if image is None:
                image = self.apply_image_transformations(self.get_image())

            if title is None:
                title = self.get_title()

            # Only show coordinates if checkbox is checked
            coordinates = None
            if self.show_coordinates.get() and self.peak_finder.cache.coordinates.numel() > 0:
                # Limit the number of coordinates to the selected amount    
                coordinates = self.peak_finder.cache.coordinates[0: self.peak_finder_params.ammount.get()]

            # Only show overlay if checkbox is checked
            overlay = None
            if self.show_masks.get() and self.peak_finder.masks:
                height, width = image.shape
                overlay = torch.zeros((height, width, 4), dtype=torch.float32, device=image.device)
                for mask in self.peak_finder.masks:
                    mask_arr = mask.as_mask((height, width), device=image.device)
                    overlay[..., 0] += mask_arr  # Red channel
                    overlay[..., 3] += mask_arr * 0.3  # Alpha channel
                overlay[..., 0] = torch.clamp(overlay[..., 0], 0, 1)
                overlay[..., 1:3] = 0  # No green/blue
                overlay[..., 3] = torch.clamp(overlay[..., 3], 0, 0.3)  # Max alpha

            def callback():
                mode = self.display_mode.get()
                color = self.display_color.get()
                size = self.display_size.get()
                self.image_display.display(image, coordinates, overlay, mode, color, size, title)

            self.root.after(0, callback)
            
        if self.show_image_display:
            threading.Thread(target=target, args=(image, title), daemon=True).start()

    def refresh_histogram(self, image: Optional[Image] = None, title: Optional[str] = None):
        def target(image: Image, title: str):
            if image is None:
                image = self.apply_image_transformations(self.get_image())

            if title is None:
                title = self.get_title()

            def callback():
                cutoff=self.peak_finder_params.cutoff.get()
                black=self.state.black_level.get()
                white=self.state.white_level.get()
                xscale=self.hist_x_axis.get()
                yscale=self.hist_y_axis.get()
                self.hist_display.display(image, cutoff, title, black, white, xscale, yscale)

            self.root.after(0, callback)
            
        if self.show_hist_display:
            threading.Thread(target=target, args=(image, title), daemon=True).start()

    def auto_update_image_display(self, *args):
        if self.auto_refresh_image_display.get():
            self.refresh_display()
    
    def auto_update_histogram_display(self, *args):
        if self.auto_refresh_hist_display.get():
            self.refresh_histogram()

    def auto_update_displays(self, *args):
        self.auto_update_image_display()
        self.auto_update_histogram_display()  

    def update_displays(self):
        def target():
            if self.show_image_display.get() or self.show_hist_display.get():
                return
            image = self.apply_image_transformations(self.get_image())
            
            def callback():
                title = self.get_title()
                self.refresh_display(image, title)
                self.refresh_histogram(image, title)

            self.root.after(0, callback)
        threading.Thread(target=target, daemon=True).start()

    def update_slider_max(self, event=None):
        new_max = int(self.ammount_slider_max.get())
        new_max = min(max(new_max, 1), 999)  # Clamp to a reasonable range
        self.ammount_slider.config(to=new_max)
        self.ammount_slider_max.set(new_max)  # Update the entry value
        # Clamp slider value if needed
        if self.peak_finder_params.ammount.get() > new_max:
            self.peak_finder_params.ammount.set(new_max)

    def reset_peak_finder_buttons(self):
        # Reset the state of the buttons after calculation
        self.calc_btn.config(state='normal')
        self.calc_btn.config(text='Calculate')
        self.cont_btn.config(state='normal')
        self.cont_btn.config(text='Continue')

    def set_coordinate_size(self, value: float):
        self.display_size.set(int(float(value)))
        self.auto_update_image_display()

    def save_file(): pass
    def save_file_as(self): pass

    def continue_fn_old(self): 
        # TODO(Deogratias):
        #   1. Test what is fast and what is slow here O(#coordinates * #frames * (w*h)*log(w*h))
        #   2. Implement Threding tor the estractions and ffts
        #   3. Calculate the distance from the center to the center of the peaks (resolution)
        #   4. Have a look at the output format???
        #   5. seprate this whole file into sensible classes
        # TODO(Johannes):
        #   1. Make this work wit CTF estimation
        #
        
        # Ask user for output filename
        output_filename = filedialog.asksaveasfilename(
            title="Choose Output File for Complete Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.state.filename) if self.state.filename else os.getcwd(),
            initialfile=os.path.splitext(self.state.filename)[0] + "_analysis.txt" if self.state.filename else "analysis.txt"
        )
        
        if not output_filename:  # User cancelled
            return
        
        class FitResult:
            def __init__(self, amplitude: float, sigma_x: float, sigma_y: float):
                self.amplitude = amplitude
                self.sigma_x = sigma_x
                self.sigma_y = sigma_y

            def __str__(self):
                return f"{self.amplitude:.3f}, {self.sigma_x:.3f}, {self.sigma_y:.3f}"
            
            def __repr__(self): return self.__str__()

        def fit_gaussian_2d(data: np.ndarray) -> FitResult:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_torch = torch.tensor(data, dtype=torch.float32, device=device)
            height, width = data.shape
            amplitude = data_torch.max()
            x0 = width / 2
            y0 = height / 2
            sigma_x = sigma_y = min(height, width) / 4
            offset = data_torch.min()
            params = torch.tensor([amplitude, x0, y0, sigma_x, sigma_y, offset], dtype=torch.float32, device=device, requires_grad=True)
            yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
            def gaussian2d(params):
                A, x0, y0, sx, sy, off = params
                return A * torch.exp(-(((xx - x0) ** 2) / (2 * sx ** 2) + ((yy - y0) ** 2) / (2 * sy ** 2))) + off
            optimizer = torch.optim.Adam([params], lr=0.05)
            for _ in range(100):
                optimizer.zero_grad()
                fit = gaussian2d(params)
                loss = torch.mean((fit - data_torch) ** 2)
                loss.backward()
                optimizer.step()
            A, x0, y0, sx, sy, off = params.detach().cpu().numpy()
            return FitResult(A, sx, sy)

        def extraction_worker(frame, idx) -> tuple[int, list[FitResult]]:
            timer = Timer()
            timer.start()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            frame_torch = torch.tensor(frame, dtype=torch.float32, device=device)
            frame_fft = torch.fft.fft2(frame_torch)
            frame_fft = torch.abs(torch.fft.fftshift(frame_fft)).cpu().numpy()
            fft_time = timer.stop()
            timer.start()
            results = []
            for x, y in self.peak_finder.cache.coordinates:
                half_size = self.peak_finder_params.R.get() // 2
                x_start = max(0, int(x) - half_size)
                x_end = min(frame_fft.shape[1], int(x) + half_size)
                y_start = max(0, int(y) - half_size)
                y_end = min(frame_fft.shape[0], int(y) + half_size)
                extracted_region = frame_fft[y_start:y_end, x_start:x_end]
                result = fit_gaussian_2d(extracted_region)
                results.append(result)
            print(f"Frame {idx}: FFT: {fft_time:.3f} s, Fitting: {timer.stop():.3f} s.")
            return idx, results

        def run_extraction():
            images = self.image_handler.handle
            timer = Timer()
            timer.start()
            results = []
            for i, image in enumerate(images):
                result = extraction_worker(image, i)
                results.append(result)
            elapsed = timer.stop()
            print(f"Extraction, fft and fitting took {elapsed:.3f} seconds.")

            # Write comprehensive results to single file
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    # Write main header
                    f.write("# Peak Analysis Results - Gaussian Fitting\n")
                    f.write(f"# Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("#" + "="*70 + "\n")
                    f.write(f"# Source file: {self.state.filename or 'Unknown'}\n")
                    f.write(f"# Pixel size: {self.apx_var.get():.6f} A/px\n")
                    f.write(f"# Frame count: {len(images)}\n")
                    f.write(f"# Peak count: {len(self.peak_finder.cache.coordinates)}\n")
                    
                    # Get image dimensions for coordinate calculations
                    image = self.get_image()
                    if image is not None:
                        height, width = image.shape
                        center_x = width / 2.0
                        center_y = height / 2.0
                        pixel_size = self.apx_var.get()
                        
                        f.write(f"# Image dimensions: {width}x{height} pixels\n")
                        f.write("#" + "="*70 + "\n\n")
                        
                        # Section 1: Peak Coordinates and Reciprocal Space
                        f.write("# SECTION 1: PEAK COORDINATES AND RECIPROCAL SPACE\n")
                        f.write("#" + "-"*50 + "\n")
                        f.write("# Peak_ID, X_Pixel, Y_Pixel, Gx_1_per_A, Gy_1_per_A, G_Magnitude_1_per_A, Resolution_Angstrom\n")
                        
                        ammount = len(self.peak_finder.cache.coordinates)
                        for i, (x, y) in enumerate(self.peak_finder.cache.coordinates[0:ammount]):
                            x_pixel = float(x)
                            y_pixel = float(y)
                            
                            # Convert to reciprocal space
                            dx_pixels = x_pixel - center_x
                            dy_pixels = y_pixel - center_y
                            gx = dx_pixels / (width * pixel_size)
                            gy = dy_pixels / (height * pixel_size)
                            g_magnitude = (gx**2 + gy**2)**0.5
                            resolution = 1 / g_magnitude if g_magnitude != 0 else float('inf')
                            
                            f.write(f"{i+1}, {x_pixel:.2f}, {y_pixel:.2f}, {gx:.6f}, {gy:.6f}, {g_magnitude:.6f}, {resolution:.6f}\n")
                        
                        f.write("\n")
                        
                        # Section 2: Gaussian Fitting Results
                        f.write("# SECTION 2: GAUSSIAN FITTING RESULTS\n")
                        f.write("#" + "-"*50 + "\n")
                        f.write("# Frame, " + ", ".join([f"Amplitude_{i+1}, Sigma_x_{i+1}, Sigma_y_{i+1}" for i in range(ammount)]) + "\n")
                        
                        for frame_idx, fit_results in results:
                            line_data = [str(frame_idx)]
                            for result in fit_results:
                                line_data.extend([f"{result.amplitude:.6f}", f"{result.sigma_x:.6f}", f"{result.sigma_y:.6f}"])
                            f.write(", ".join(line_data) + "\n")
                    
                print(f"Complete analysis results written to {output_filename}")
            except Exception as e:
                print(f"Error writing results: {e}")
            
            # Schedule GUI update on main thread
            self.root.after(0, lambda: self._on_continue_done())

        # Disable buttons and give feedback
        self.cont_btn.config(state='disabled', text='Working...')
        self.calc_btn.config(state='disabled')
        threading.Thread(target=run_extraction, daemon=True).start()

    def continue_fn(self): 
        # Simple brightness analysis - find the brightest pixel in each region for every frame
        
        # Ask user for output filename
        output_filename = filedialog.asksaveasfilename(
            title="Choose Output File for Complete Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.state.filename) if self.state.filename else os.getcwd(),
            initialfile=os.path.splitext(self.state.filename)[0] + "_analysis.txt" if self.state.filename else "analysis.txt"
        )
        
        if not output_filename:  # User cancelled
            return
        
        def extraction_worker(frame, idx, ammount) -> tuple[int, list[float]]:
            timer = Timer()
            timer.start()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            frame_torch = torch.tensor(frame, dtype=torch.float32, device=device)
            frame_fft = torch.fft.fft2(frame_torch)
            frame_fft = torch.abs(torch.fft.fftshift(frame_fft)).cpu().numpy()
            fft_time = timer.stop()
            
            timer.start()
            max_values = []
            for x, y in self.peak_finder.cache.coordinates[0:ammount]:
                #print(int(x), int(y))
                half_size = self.peak_finder_params.R.get() // 2
                x_start = max(0, int(x) - half_size)
                x_end = min(frame_fft.shape[1], int(x) + half_size)
                y_start = max(0, int(y) - half_size)
                y_end = min(frame_fft.shape[0], int(y) + half_size)
                if x_start >= x_end or y_start >= y_end:
                    #print(f"Skipping invalid region for peak at ({x}, {y}) in frame {idx}.")
                    max_values.append(0.0)
                    continue
                extracted_region = frame_fft[y_start:y_end, x_start:x_end]
                max_value = float(np.max(extracted_region))
                max_values.append(max_value)
            
            analysis_time = timer.stop()
            print(f"Frame {idx}: FFT: {fft_time:.3f} s, Max value analysis: {analysis_time:.3f} s.")
            return idx, max_values

        def run_extraction():
            images = self.image_handler.handle
            timer = Timer()
            timer.start()
            ammount = min(self.peak_finder_params.ammount.get(), len(self.peak_finder.cache.coordinates))
            if ammount == 0:    
                print("No peaks found to analyze.")
                self._on_continue_done()
                return  
            
            results = []
            for i, image in enumerate(images):
                result = extraction_worker(image, i, ammount)
                results.append(result)
            
            elapsed = timer.stop()
            print(f"Brightness analysis took {elapsed:.3f} seconds.")

            # Write comprehensive results to single file
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    # Write main header
                    f.write("# Peak Analysis Results - Brightness Analysis\n")
                    f.write(f"# Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("#" + "="*70 + "\n")
                    f.write(f"# Source file: {self.state.filename or 'Unknown'}\n")
                    f.write(f"# Pixel size: {self.apx_var.get():.6f} A/px\n")
                    f.write(f"# Frame count: {len(results)}\n")
                    f.write(f"# Peak count: {ammount}\n")
                    
                    # Get image dimensions for coordinate calculations
                    image = self.get_image()
                    if image is not None:
                        height, width = image.shape
                        center_x = width / 2.0
                        center_y = height / 2.0
                        pixel_size = self.apx_var.get()
                        
                        f.write(f"# Image dimensions: {width}x{height} pixels\n")
                        f.write("#" + "="*70 + "\n\n")
                        
                        # Section 1: Peak Coordinates and Reciprocal Space
                        f.write("# SECTION 1: PEAK COORDINATES AND RECIPROCAL SPACE\n")
                        f.write("#" + "-"*50 + "\n")
                        f.write("# Peak_ID, X_Pixel, Y_Pixel, Gx_1_per_A, Gy_1_per_A, G_Magnitude_1_per_A, Resolution_Angstrom\n")
                        
                        for i, (x, y) in enumerate(self.peak_finder.cache.coordinates[0:ammount]):
                            x_pixel = float(x)
                            y_pixel = float(y)
                            
                            # Convert to reciprocal space
                            dx_pixels = x_pixel - center_x
                            dy_pixels = y_pixel - center_y
                            gx = dx_pixels / (width * pixel_size)
                            gy = dy_pixels / (height * pixel_size)
                            g_magnitude = (gx**2 + gy**2)**0.5
                            resolution = 1 / g_magnitude if g_magnitude != 0 else float('inf')
                            
                            f.write(f"{i+1}, {x_pixel:.2f}, {y_pixel:.2f}, {gx:.6f}, {gy:.6f}, {g_magnitude:.6f}, {resolution:.6f}\n")
                        
                        f.write("\n")
                        
                        # Section 2: Brightness Analysis Results
                        f.write("# SECTION 2: BRIGHTNESS ANALYSIS RESULTS\n")
                        f.write("#" + "-"*50 + "\n")
                        f.write("# Frame, " + ", ".join([f"Peak_{i+1}" for i in range(ammount)]) + "\n")
                        
                        for frame_idx, max_values in results:
                            line = f"{frame_idx}, " + ", ".join([f"{val:.6f}" for val in max_values]) + "\n"
                            f.write(line)
                    
                print(f"Complete brightness analysis results written to {output_filename}")
            except Exception as e:
                print(f"Error writing results: {e}")
            
            # Schedule GUI update on main thread
            self.root.after(0, lambda: self._on_continue_done())

        # Disable buttons and give feedback
        self.cont_btn.config(state='disabled', text='Analyzing...')
        self.calc_btn.config(state='disabled')
        threading.Thread(target=run_extraction, daemon=True).start()

    def _on_continue_done(self):
        self.cont_btn.config(state='normal', text='Continue')
        self.calc_btn.config(state='normal')
        #self.info_display.config(text=msg)

    def show_about(self):
        tk.messagebox.showinfo(
            "About",
            "Peak Finder Widget\n\nby Johannes Schuster\nUniversity of Regensburg\n\n04.07.2025"
        )

    def on_closing(self):
        self.image_display.close()
        self.hist_display.close()
        self.root.destroy()
    
    def on_closing_display(self):
        self.show_image_display = False
        self.display_refresh_button.config(text='Show')
        self.image_display.close()
        self.image_display = ImageDisplay()
        self.image_display.callback("WM_DELETE_WINDOW", self.on_closing_display)

    def on_closing_hist(self):
        self.show_hist_display = False
        self.hist_refresh_button.config(text='Show')
        self.hist_display.close()
        self.hist_display = HistogramDisplay()
        self.hist_display.callback("WM_DELETE_WINDOW", self.on_closing_hist)

    def on_precalc_sum(self):
        if self.image_handler.handle is None:
            return
        if self.precalc_sum_var.get():
            threading.Thread(target=self.image_handler.precompute, daemon=True).start()

    def on_precalc_ffts(self):
        if self.image_handler.handle is None:
            return
        if self.precalc_fft_var.get():
            if self.fft_image_cache is None:
                self.fft_image_cache = FFTImageHandler()
                self.fft_image_cache.set_handle(self.image_handler.handle)
            threading.Thread(target=self.fft_image_cache.precompute, daemon=True).start()


def run_gui():
    root = tk.Tk()
    root.resizable(False, False)
    app = PeakFinderWidget(root)
    # Check for file path argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            app.file_path_var.set(file_path)
            app.load_file()
    root.mainloop()

if __name__ == '__main__':
    run_gui()




