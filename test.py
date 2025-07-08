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
from typing import Optional, Any
import concurrent.futures
import torch

# Importing custom modules
from display import Image, ImageDisplay, ImageHandler, FFImageHandler
from display import HistogramDisplay
from peak_finder import PeakFinder, Mask, RectMask, CircleMask, PolyMask
from timer import Timer
from gaussfitter import gaussfit

@dataclass
class State:
    handle: Optional[torch.Tensor] = None
    filename: Optional[str] = None
    black_level: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.25))
    white_level: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.9))
    sigma_var: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=3.0))
    gamma_var: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=6.0))

@dataclass
class PeakFinderParams:
    ammount: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=50))
    cutoff: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=220))
    R: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=150))
    LINE: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=20))

@dataclass
class PeakFinderCache:
    coordinates: torch.Tensor = field(default_factory=lambda: torch.empty((0, 2), dtype=torch.float32))

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
        self.peak_finder_cache = PeakFinderCache()
        self.masks: list[Mask] = []  # type: list
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _menubar(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load", command=self.browse_file)
        filemenu.add_command(label="Save", command=self.save_file)
        filemenu.add_command(label="Save As", command=self.save_file_as)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=filemenu)
        # Image menu
        imagemenu = tk.Menu(menubar, tearoff=0)
        imagemenu.add_command(label="Precalculate Sum", command=self.on_precalc_sum)
        imagemenu.add_command(label="Precalculate FFT", command=self.on_precalc_ffts)
        menubar.add_cascade(label="Image", menu=imagemenu)
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
        self.file_input.bind('<Return>', self.load_file)
        self.file_input.bind('<FocusOut>', self.load_file)
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

        # Show button (full width)
        self.show_image_btn = ttk.Button(outer, text='Show', command=self.update_image_display)
        self.show_image_btn.pack(fill='x', pady=(6,0))

        # Image display frame
        img_frame = ttk.LabelFrame(parent, text='Display', padding=(5, 5))
        img_frame.pack(fill='x', pady=(10, 0))
        # black level
        ttk.Label(img_frame, text="Black [0-1]=").grid(row=0, column=0)
        self.black_level_entry = ttk.Entry(img_frame, textvariable=self.state.black_level) 
        self.black_level_entry.grid(row=0, column=1)
        # white level
        ttk.Label(img_frame, text="White [0-1]=").grid(row=1, column=0)
        self.white_level_entry = ttk.Entry(img_frame, textvariable=self.state.white_level) 
        self.white_level_entry.grid(row=1, column=1)
        # sigma
        ttk.Label(img_frame, text="Sigma=").grid(row=0, column=2)
        self.sigma_entry = ttk.Entry(img_frame, textvariable=self.state.sigma_var, width=6)
        self.sigma_entry.grid(row=0, column=3)
        # gamma
        ttk.Label(img_frame, text="Gamma=").grid(row=1, column=2)
        self.gamma_entry = ttk.Entry(img_frame, textvariable=self.state.gamma_var, width=6)
        self.gamma_entry.grid(row=1, column=3)

        # Histogram display frame
        hist_frame = ttk.LabelFrame(parent, text='Histogram', padding=(5, 5))
        hist_frame.pack(fill='x', pady=(10, 0))
        # X Axis selection
        ttk.Label(hist_frame, text='X Axis:').grid(row=0, column=0, sticky='w')
        self.hist_x_axis = tk.StringVar(value='linear')
        x_axis_menu = ttk.OptionMenu(hist_frame, self.hist_x_axis, 'linear', 'linear', 'log')
        x_axis_menu.grid(row=0, column=1, sticky='w')
        # Y Axis selection
        ttk.Label(hist_frame, text='Y Axis:').grid(row=1, column=0, sticky='w')
        self.hist_y_axis = tk.StringVar(value='linear')
        y_axis_menu = ttk.OptionMenu(hist_frame, self.hist_y_axis, 'linear', 'linear', 'log')
        y_axis_menu.grid(row=1, column=1, sticky='w')
        # Update histogram on change
        self.hist_x_axis.trace_add('write', lambda *args: self.update_plot())
        self.hist_y_axis.trace_add('write', lambda *args: self.update_plot())

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
        def update_slider_max(event=None):
            try:
                new_max = int(self.ammount_slider_max.get())
                new_max = torch.clamp(new_max, 1, 999)  # Clamp to a reasonable range
                self.ammount_slider.config(to=new_max)
                self.ammount_slider_max.set(new_max)  # Update the entry value
                # Clamp slider value if needed
                if self.peak_finder_params.ammount.get() > new_max:
                    self.peak_finder_params.ammount.set(new_max)
            except Exception:
                pass
        max_entry.bind('<FocusOut>', update_slider_max)
        max_entry.bind('<Return>', update_slider_max)
        # Update label when slider changes
        self.ammount_slider.config(command=lambda val: self.peak_finder_params.ammount.set(int(float(val))))

        # Cutoff row
        cutoff_row = ttk.Frame(settings_frame)
        cutoff_row.pack(fill='x', pady=2)
        ttk.Label(cutoff_row, text='Cutoff:', anchor='e', width=12).pack(side='left')
        cutoff_entry = ttk.Entry(cutoff_row, textvariable=self.peak_finder_params.cutoff)
        cutoff_entry.pack(side='left', fill='x', expand=True)

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
        size_slider = ttk.Scale(display_frame, from_=2, to=50, orient='horizontal', variable=self.display_size)
        size_slider.grid(row=2, column=1, sticky='ew')
        size_slider.config(command=lambda val: self.display_size.set(int(float(val))))
        size_label = ttk.Label(display_frame, textvariable=self.display_size)
        size_label.grid(row=2, column=2, sticky='w')

        # Add checkboxes for toggling coordinates and masks display
        self.show_coordinates = tk.BooleanVar(value=True)
        self.show_masks = tk.BooleanVar(value=True)
        coords_checkbox = ttk.Checkbutton(display_frame, text='Show Coordinates', variable=self.show_coordinates)
        coords_checkbox.grid(row=0, column=3, sticky='w', pady=(4,0))
        masks_checkbox = ttk.Checkbutton(display_frame, text='Show Masks', variable=self.show_masks)
        masks_checkbox.grid(row=1, column=3, sticky='w', pady=(4,0))

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
        self.info_display.config(text='Loading...')

        def do_load(file_path):
            handle = torch.from_numpy(tifffile.imread(file_path))
            self.state.handle = handle
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
        def do_find_peaks():
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
                self.masks = [circle_mask, rectMask1, rectMask2]
                # Use PeakFinder
                self.peak_finder.clear_masks()
                for mask in self.masks:
                    self.peak_finder.add_mask(mask)

                self.peak_finder.threshold_abs = cutoff
                coordinates = self.peak_finder.find_peaks(image=image, window=3)
                if len(coordinates) > ammount:
                    coordinates = coordinates[:ammount]
                self.peak_finder_cache.coordinates = coordinates
                self.peak_finder.clear_masks()
                self.root.after(0, self.update_plot)

            self.root.after(0, self.update_ammount_slider)
            self.root.after(0, self.reset_peak_finder_buttons)

        self.calc_btn.config(state='disabled')
        self.calc_btn.config(text='Calculating...')
        self.cont_btn.config(state='disabled')
        threading.Thread(target=do_find_peaks, daemon=True).start()

    def get_image(self) -> Image:
        if self.image_handler.handle is None:
            return None
        show_sum = self.display_sum.get()
        do_fft = self.display_fft.get()
        idx = self.display_frame_idx.get()
        if do_fft and self.fft_image_cache is None:
            self.fft_image_cache = FFImageHandler()
            self.fft_image_cache.set_handle(self.image_handler.handle)
        if do_fft:
            if show_sum: return self.fft_image_cache.get_sum()
            else: return self.fft_image_cache.get_frame(idx)
        else:
            if show_sum: return self.image_handler.get_sum()
            else: return self.image_handler.get_frame(idx)

    def update_plot(self):
        image = self.get_image()
        if image is None:
            return
        mode = self.display_mode.get()
        color = self.display_color.get()
        size = self.display_size.get()
        title = f"{'Sum' if self.display_sum.get() else f'Frame {self.display_frame_idx.get()}'} - {self.state.filename or 'Untitled'}"
        
        # Only show coordinates if checkbox is checked
        coordinates = None
        if self.show_coordinates.get() and self.peak_finder_cache.coordinates.numel() > 0:
            # Limit the number of coordinates to the selected amount    
            coordinates = self.peak_finder_cache.coordinates[0: self.peak_finder_params.ammount.get()]

        # Only show overlay if checkbox is checked
        overlay = None
        if self.show_masks.get() and self.masks:
            height, width = image.shape
            overlay = torch.zeros((height, width, 4), dtype=torch.float32, device=image.device)
            for mask in self.masks:
                mask_arr = mask.as_mask((height, width), device=image.device)
                overlay[..., 0] += mask_arr  # Red channel
                overlay[..., 3] += mask_arr * 0.3  # Alpha channel
            overlay[..., 0] = torch.clamp(overlay[..., 0], 0, 1)
            overlay[..., 1:3] = 0  # No green/blue
            overlay[..., 3] = torch.clamp(overlay[..., 3], 0, 0.3)  # Max alpha
        # Get sigma and gamma
        sigma = self.state.sigma_var.get() 
        gamma = self.state.gamma_var.get() 
        black = max(0.0, min(1.0, self.state.black_level.get()))
        white = max(0.0, min(1.0, self.state.white_level.get()))
        image = image.rescale(sigma=sigma, gamma=gamma)
        image = image.remap(min_val=black, max_val=white)
        self.image_display.display_image(
            image=image,
            mode=mode,
            color=color,
            size=size,
            title=title,
            coordinates=coordinates,
            overlay=overlay
        )
        self.hist_display.display_histogram(
            image=image,
            cutoff=self.peak_finder_params.cutoff.get(),
            title=title,
            black=black,
            white=white,
            xscale=self.hist_x_axis.get(),
            yscale=self.hist_y_axis.get(),
        )

    def update_image_display(self):
        self.update_plot()

    def update_ammount_slider(self):
        # Update the ammount slider range based on current coordinates
        coords = self.peak_finder_cache.coordinates
        found = coords.size(dim=0)
        if found > 0:
            max_ammount = min(2000, found)
            self.ammount_slider.config(to=max_ammount)
            # Set slider to selected (if not already)
            selected = self.peak_finder_params.ammount.get()
            if selected > max_ammount or selected == 0:
                self.peak_finder_params.ammount.set(max_ammount)
        else:
            self.ammount_slider.config(to=2000)
            self.peak_finder_params.ammount.set(600)    

    def reset_peak_finder_buttons(self):
        # Reset the state of the buttons after calculation
        self.calc_btn.config(state='normal')
        self.calc_btn.config(text='Calculate')
        self.cont_btn.config(state='normal')
        self.cont_btn.config(text='Continue')

    def save_file(): pass
    def save_file_as(self): pass

    def continue_fn(self): 
        # TODO(Deogratias):
        #   1. Test what is fast and what is slow here O(#coordinates * #frames * (w*h)*log(w*h))
        #   2. Implement Threding tor the estractions and ffts
        #   3. Calculate the distance from the center to the center of the peaks (resolution)
        #   4. Have a look at the output format???
        #   5. seprate this whole file into sensible classes
        # TODO(Johannes):
        #   1. Make this work wit CTF estimation
        #  
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
            for x, y in self.peak_finder_cache.coordinates:
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = list(executor.map(extraction_worker, images, range(len(images))))
            elapsed = timer.stop()
            print(f"Extraction, fft and fitting took {elapsed:.3f} seconds.")

            # Write results to file
            out_path = os.path.join(os.path.dirname(self.state.filename or 'output.txt'), 'fitted_gaussians.txt')
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write('Frame, ' + ', '.join([f'Amplitude_{i+1}, Sigma_x_{i+1}, Sigma_y_{i+1}' for i in range(len(self.peak_finder_cache.coordinates))]) + '\n')
                    for line in results:
                        f.write(', '.join([f'{res}' for res in line]) + '\n')
                print(f"Results written to {out_path}")
            except Exception as e:
                print(f"Error writing results: {e}")
            # Schedule GUI update on main thread
            self.root.after(0, lambda: self._on_continue_done())

        # Disable buttons and give feedback
        self.cont_btn.config(state='disabled', text='Working...')
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

    def on_precalc_sum(self):
        if self.image_handler.handle is None:
            return
        self.image_handler.precompute()
        self.info_display.config(text="Sum precalculated.")

    def on_precalc_ffts(self):
        if self.image_handler.handle is None:
            return
        self.fft_image_cache = FFImageHandler()
        self.fft_image_cache.set_handle(self.image_handler.handle)
        self.fft_image_cache.precompute()
        self.info_display.config(text="FFT precalculated.")


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




