import torch
import torch.nn.functional as F
import time

class CTF:
    class DebugLevel:
        NONE = 0
        MINIMAL = 1
        TIMING = 2
        DETAILED = 3
        FULL = 4
    
    def __init__(self, pixel_size: float = 1.0, Cs: float = 2.7, voltage: float = 200.0, phase_shift: float = 0, amplitude_contrast: float = 0.07):
        # Fixed parameters 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pixel_size = pixel_size  # pixel size in Angstrom
        self.Cs = Cs  # spherical aberration in microns
        self.voltage = voltage  # voltage in kV
        self.phase_shift = phase_shift  # phase shift in degrees
        self.la = 12.264 * pixel_size / voltage  # wavelength in Angstrom
        try: 
            __phase_correction = torch.tensor((1 / (amplitude_contrast * amplitude_contrast)) - 1, device=self.device)
            contrast_phase = torch.atan2(torch.ones((1), device=self.device), torch.sqrt(__phase_correction))  # contrast phase in radians
        except ZeroDivisionError:
            contrast_phase = 0.0
        self.phase = contrast_phase + phase_shift * torch.pi / 180.0 # phase shift in radians
        self.prefactor = torch.pi*1e4  # prefactor for the XI equation

        # Fit parameters
        defocus1 = 0.0  # first defocus in microns  
        defocus2 = 0.0  # second defocus in microns
        defocus_angle = 0.0  # defocus angle in radianss
        self.__params: torch.Tensor = torch.tensor((defocus1, defocus2, defocus_angle), 
                                                    device='cuda' if torch.cuda.is_available() else 'cpu')
        
        self.__fit_params: torch.Tensor = torch.tensor((512, 5.0, 30.0, 0.5, 5.0, 0.05, 0.01),  
                                                       device='cuda' if torch.cuda.is_available() else 'cpu')
        self.__optional_fit_params: torch.Tensor = torch.tensor((0, 0, 0, 0, 3, 5), 
                                                                device='cuda' if torch.cuda.is_available() else 'cpu')
        self.angle_estimation: str = "parallel"
        self.defocus_estimation: str = "gradient"
        self.__g_mask: torch.Tensor|None = None
        self.__ray_mask: torch.Tensor|None = None
        self.__debug_level = 0

    def df(self, angle: float) -> torch.Tensor:
        df1, df2, a1 = self.__params
        df_delta = df1 - df2
        return 0.5 * (df1 + df2 + df_delta*torch.cos(2*(angle-a1))) 

    def XI(self, g: torch.Tensor) -> torch.Tensor:
        # g: (..., 2) tensor, last dim is (gx, gy)
        gx, gy = g[..., 0], g[..., 1]
        m_g2 = gx**2 + gy**2  # square magnitude of the reciprocal space vector in 1/A^2
        angle = torch.atan2(gy, gx)
        la = self.la # keep in Angstrom
        la2 = self.la * self.la # keep in 1/A^2
        df = self.df(angle) # keep in microns
        Cs = self.Cs # keep in microns
        return self.prefactor*la*m_g2*(df - 0.5*la2*m_g2*Cs) + self.phase
    
    def __SI_XI(self, g: torch.Tensor) -> torch.Tensor:
        # g: (..., 2) tensor, last dim is (gx, gy)
        gx, gy = g[..., 0], g[..., 1]
        m_2 = (gx**2 + gy**2) * 1e20 # square magnitude of the reciprocal space vector in 1/m^2
        angle = torch.atan2(gy, gx) # angle in radians
        la = self.la * 1e-10 # wavelength in meters
        la2 = self.la * self.la * 1e-20 # wavelength squared in 1/m^2
        df = self.df(angle) * 1e-6 # defocus in meters
        Cs = self.Cs * 1e-6 # spherical aberration in meters
        return torch.pi*la*m_2*(df - 0.5*la2*m_2*Cs) + self.phase
    
    def at(self, g: torch.Tensor, __use_SI_XI: bool = False) -> torch.Tensor:
        xi = self.__SI_XI(g) if __use_SI_XI else self.XI(g)
        return -torch.sin(xi)
    
    def get(self, size: tuple[int,int] = (0,0)) -> torch.Tensor:
        return self.at(self.get_reciprocal_space(size))
    
    def get_reciprocal_space(self, size: tuple[int,int] = (0,0)) -> torch.Tensor: 
        h, w = size
        if h <= 0 or w <= 0:
            raise ValueError("Size must be a positive tuple (height, width)")
        h_extend = 1 / (self.pixel_size)
        w_extent = 1 / (self.pixel_size)
        y = torch.linspace(-h_extend, h_extend, h, device=self.device)
        x = torch.linspace(-w_extent, w_extent, w, device=self.device)

        # Create 2D grid of coordinates
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack((yy, xx), dim=-1)  # shape (h, w, 2)
    
    def compute_g_mask(self, size: tuple[int,int] = (0,0)) -> torch.Tensor:
        g = self.get_reciprocal_space(size)
        g2 = (g[..., 0]**2 + g[..., 1]**2).float()
        
        g_bounds = 1 / self.get_fit_params()[1:3]**2
        return ((g2 <= g_bounds[0]) & (g2 >= g_bounds[1])).float()
    
    def get_g_mask(self, size: tuple[int,int] = (0,0)) -> torch.Tensor:
        """
        Returns the mask for the reciprocal space based on g_min and g_max.
        """
        if self.__g_mask is None or self.__g_mask.shape != size:
            self.__g_mask = self.compute_g_mask(size)
        return self.__g_mask
    
    def compute_ray_mask(self, shape: tuple[int, int]) -> torch.Tensor:
        """
        Returns a mask that zeros out pixels within ±ray_width of the center row and column.
        """
        h, w = shape
        ray_width = self.__optional_fit_params[4].item()  # ray width from optional fit params
        mask = torch.ones((h, w), device=self.device)
        cy, cx = h // 2, w // 2
        # Zero out bands around center row and column
        mask[max(0, cy-ray_width):min(h, cy+ray_width), :] = 0
        mask[:, max(0, cx-ray_width):min(w, cx+ray_width)] = 0
        return mask

    def get_ray_mask(self, shape: tuple[int, int]) -> torch.Tensor:
        """
        Returns a mask that zeros out pixels within ±ray_width of the center row and column.
        """
        if self.__ray_mask is None or self.__ray_mask.shape != shape:
            self.__ray_mask = self.compute_ray_mask(shape)
        return self.__ray_mask
    
    def set_initial_params(self, defocus1: float, defocus2: float, defocus_angle: float):
        self.__params = torch.tensor((defocus1, defocus2, defocus_angle), 
                                         device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_initial_params(self) -> torch.Tensor:
        return self.__params.cpu().clone()
        
    def set_fit_params(self, 
                       N_d: int = 512, # size of the defocus grid
                       g_range: tuple[float, float] = (5.0, 30.0), # in 1/A
                       defocus_range: tuple[float, float] = (0.5, 5), # in microns
                       defocus_step: float = 0.05, # in microns
                       defocus_restraint: float = 0.01 # in microns
                       ):
        # Set the parameters for the fitting process
        self.__fit_params = torch.tensor((N_d, g_range[0], g_range[1],
                                          defocus_range[0], defocus_range[1], defocus_step, defocus_restraint), 
                                         device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_fit_params(self) -> torch.Tensor:
        return self.__fit_params.cpu().clone()
    
    def set_optional_fit_params(self,
                                N_frames: int = -1, # number of frames to fit
                                delta_phi_range: tuple[float, float] = (0, torch.pi), # range of phase shifts in degrees
                                delta_phi_step: float = 0.01, # step size for phase shifts in
                                ray_mask: int = 3, # mask for rays in reciprocal space
                                angle_range: float = 5 # range around maximum angle for polynomial fitting in degrees
                                ):
        self.__optional_fit_params = torch.tensor((N_frames, delta_phi_range[0], delta_phi_range[1], delta_phi_step, ray_mask, angle_range),
                                                  device='cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_optional_fit_params(self) -> torch.Tensor:
        return self.__optional_fit_params.cpu().clone()
    
    def set_debug_level(self, level: int):
        """
        Set the debug level for the CTF class.
        Use CTF.DebugLevel constants:
        CTF.DebugLevel.NONE (0) - no debug output
        CTF.DebugLevel.MINIMAL (1) - basic debug output
        CTF.DebugLevel.TIMING (2) - detailed debug output including timing
        CTF.DebugLevel.DETAILED (3) - even more detailed debug output
        CTF.DebugLevel.FULL (4) - full debug output including plots
        """
        if level < CTF.DebugLevel.NONE or level > CTF.DebugLevel.FULL:
            raise ValueError(f"Debug level must be between {CTF.DebugLevel.NONE} and {CTF.DebugLevel.FULL}")
        self.__debug_level = level

    def fit(self, data: torch.Tensor, data_is_fourier: bool = False,
            initial_params: torch.Tensor|None = None, 
            fit_params: torch.Tensor|None = None,
            optional_fit_params: torch.Tensor|None = None,
            max_iter: int = 1000, tol: float = 1e-6, eps: float = 1e-16) -> torch.Tensor:
        if initial_params is not None:
            self.set_initial_params(*initial_params)
        if fit_params is not None:
            self.set_fit_params(*fit_params)    
        if optional_fit_params is not None:
            self.set_optional_fit_params(*optional_fit_params)
        
        if not data_is_fourier:
            N_frames = self.__optional_fit_params[0].item() if self.__optional_fit_params is not None else -1
            t0 = time.time()
            data = self.get_fft_data(data, N_frames, eps)
            t1 = time.time()
            self.__debug_print(f"[CTF::fit] FFT: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)

        if len(data.shape) == 2:
            t0 = time.time()
            self.__fit_2d(data, max_iter, tol, eps)
            t1 = time.time()
            self.__debug_print(f"[CTF::fit] 2D CTF fit: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        elif len(data.shape) == 3:
            t0 = time.time()
            #self.__fit_3d(data, max_iter, tol, eps)
            t1 = time.time()
            self.__debug_print(f"[CTF::fit] 3D CTF fit: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        else:
            raise ValueError("Data must be either 2D or 3D tensor")
        
        # Clean up resources after fitting
        self.__cleanup_resources()
        
        # This should implement the fitting algorithm to optimize defocus parameters
        # For now, we return the current parameters
        return self.get_initial_params().clone().detach().cpu()
    
    def __fit_2d(self, data: torch.Tensor, max_iter: int, tol: float, eps: float) -> torch.Tensor:
        # Center crop
        t0 = time.time()
        N_d = int(self.get_fit_params()[0])
        if N_d <= 0 or N_d > min(data.shape):
            raise ValueError(f"N_d must be a positive integer <= min(height, width), got {N_d} for size {data.shape}")
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        half = N_d // 2
        # Crop center square
        crop = data[cy-half:cy+half, cx-half:cx+half]
        t1 = time.time()
        self.__debug_print(f"[CTF::__fit_2d] Crop: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)

        # Create disc mask using g_min, g_max in reciprocal space
        t0 = time.time()
        g_mask = self.compute_g_mask(data.shape)
        mask = g_mask
        #mask = circ_mask * g_mask
        crop_masked = crop * mask[cy-half:cy+half, cx-half:cx+half]
        t1 = time.time()
        self.__debug_print(f"[CTF::__fit_2d] Masking create: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)

        # Angle estimation
        accuracy = 0.01  # Default angle step in radians
        if self.angle_estimation == "sequential":
            t0 = time.time()    
            result = self.__angle_search_sequential(crop_masked, accuracy)
            t1 = time.time()
            self.__debug_print(f"[CTF::__fit_2d] Sequential angle search: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        elif self.angle_estimation == "parallel":
            t0 = time.time()  
            result = self.__angle_search_parallel(crop_masked, accuracy)
            t1 = time.time()
            self.__debug_print(f"[CTF::__fit_2d] Parallel angle search: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        else:
            raise ValueError(f"Unknown angle estimation method: {self.angle_estimation}. Use 'sequential' or 'parallel'.")
        
        # Extract best angle from result tensor and fit 2nd order polynomial around maximum
        best_idx = torch.argmax(result[:, 1])
        coarse_angle = result[best_idx, 0].detach().item()  # coarse angle in radians
        
        # Define range around maximum for polynomial fitting
        angle_range = self.__optional_fit_params[5].item()  # angle range from optional fit params
        angle_range = angle_range * torch.pi / 180.0  # convert to radians
        angles = result[:, 0]
        overlaps = result[:, 1]
        
        # Find indices within the range
        mask = (angles >= coarse_angle - angle_range) & (angles <= coarse_angle + angle_range)
        if mask.sum() < 3:  # Need at least 3 points for 2nd order polynomial
            self.__debug_print(f"[CTF::__fit_2d] Not enough points for polynomial fit, using coarse angle", CTF.DebugLevel.DETAILED)
            defocus_angle = coarse_angle
        else:
            # Extract data for fitting
            fit_angles = angles[mask]
            fit_overlaps = overlaps[mask]
            
            # Fit 2nd order polynomial: y = ax² + bx + c using PyTorch
            # Create Vandermonde matrix: [x², x, 1]
            A = torch.stack([fit_angles**2, fit_angles, torch.ones_like(fit_angles)], dim=1)
            # Solve least squares: A @ coeffs = fit_overlaps
            coeffs = torch.linalg.lstsq(A, fit_overlaps).solution
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            
            # Find maximum of polynomial: derivative = 2ax + b = 0 => x = -b/(2a)
            if abs(a) < 1e-12:  # Nearly linear, use coarse estimate
                defocus_angle = coarse_angle
            else:
                defocus_angle = -b / (2 * a)
                # Ensure the fitted angle is within reasonable bounds
                if defocus_angle < coarse_angle - angle_range or defocus_angle > coarse_angle + angle_range:
                    defocus_angle = coarse_angle
        
        self.__debug_print(f"[CTF::__fit_2d] Best angle: {defocus_angle*180/torch.pi:.3f} degrees (coarse: {coarse_angle*180/torch.pi:.3f}), overlap: {result[best_idx, 1].item():.3f}", CTF.DebugLevel.DETAILED)
        self.__params[2] = defocus_angle  # Store the fitted angle in parameters
    
        recipocal_space = self.get_reciprocal_space(data.shape)[cy-half:cy+half, cx-half:cx+half]
        def batch_score(data: torch.Tensor, defocus: torch.Tensor) -> torch.Tensor:
            """
            Vectorized batch score function for multiple defocus pairs.
            defocus: [N, 2] tensor where each row is [defocus1, defocus2]
            Returns: [N] tensor of scores
            """
            N = defocus.shape[0]
            defocus_angle = self.__params[2]
            
            # Get reciprocal space coordinates
            gx, gy = recipocal_space[..., 0], recipocal_space[..., 1]  # [H, W]
            angle = torch.atan2(gy, gx)  # [H, W]
            
            # Expand angle to batch dimension: [1, H, W] -> [N, H, W]
            angle_batch = angle.unsqueeze(0).expand(N, -1, -1)
            
            # Compute defocus difference for each batch item
            defocus_delta = defocus[:, 0] - defocus[:, 1]  # [N]
            
            # Expand defocus parameters to spatial dimensions
            # defocus[:, 0] is [N] -> [N, 1, 1] -> [N, H, W]
            df1_batch = defocus[:, 0].view(N, 1, 1).expand(-1, *angle.shape)  # [N, H, W]
            df2_batch = defocus[:, 1].view(N, 1, 1).expand(-1, *angle.shape)  # [N, H, W]
            defocus_delta_batch = defocus_delta.view(N, 1, 1).expand(-1, *angle.shape)  # [N, H, W]
            
            # Compute defocus as function of angle for all batch items
            cos_term = torch.cos(2 * (angle_batch - defocus_angle))  # [N, H, W]
            df = 0.5 * (df1_batch + df2_batch + defocus_delta_batch * cos_term)  # [N, H, W]
            
            # Compute CTF parameters
            m_g2 = gx**2 + gy**2  # [H, W]
            m_g2_batch = m_g2.unsqueeze(0).expand(N, -1, -1)  # [N, H, W]
            
            la = self.la
            la2 = self.la * self.la
            Cs = self.Cs
            
            # Compute XI (CTF phase) for all batch items
            xi = self.prefactor * la * m_g2_batch * (df - 0.5 * la2 * m_g2_batch * Cs) + self.phase  # [N, H, W]
            
            # Compute CTF for all batch items
            ctf = 0.5 - 0.5 * torch.sin(xi)  # [N, H, W]

            # Compute scores for all batch items
            data_expanded = data.unsqueeze(0).expand(N, -1, -1)  # [N, H, W]
            
            # Cross-correlation numerator and denominator
            num = (ctf * data_expanded).sum(dim=(1, 2))  # [N]
            data_norm_sq = (data**2).sum()  # scalar
            ctf_norm_sq = (ctf**2).sum(dim=(1, 2))  # [N]
            denom = torch.sqrt(data_norm_sq * ctf_norm_sq)  # [N]
            
            # Cross-correlation
            cc = num / (denom + 1e-12)  # [N]
            
            # Restraint correction
            N_CC = data.shape[0] * data.shape[1]
            ddf_max = self.__fit_params[6].item()  # defocus restraint
            correction = defocus_delta**2 / (2 * ddf_max**2 * float(N_CC))  # [N]
            
            scores = cc - correction  # [N]
            return scores
        
        
        if self.defocus_estimation == "linear":
            t0 = time.time()
            defocus1, defocus2, final_score = self.__optimize_defocus_linear(crop_masked, batch_score)
            t1 = time.time()
            self.__debug_print(f"[CTF::__fit_2d] Linear defocus optimization: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        elif self.defocus_estimation == "gradient":
            t0 = time.time()
            defocus1, defocus2, final_score = self.__optimize_defocus_gradient(crop_masked, max_iter, tol, eps, batch_score)
            t1 = time.time()
            self.__debug_print(f"[CTF::__fit_2d] Gradient defocus optimization: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        elif self.defocus_estimation == "annealing":
            t0 = time.time()
            defocus1, defocus2, final_score = self.__optimize_defocus_annealing(crop_masked, max_iter, batch_score)
            t1 = time.time()
            self.__debug_print(f"[CTF::__fit_2d] Simulated annealing defocus optimization: {(t1-t0)*1000:.3f} ms", CTF.DebugLevel.TIMING)
        else:
            raise ValueError(f"Unknown defocus estimation method: {self.defocus_estimation}. Use 'linear', 'gradient', or 'annealing'.")

        self.__debug_print(f"[CTF::__fit_2d] Defocus1: {defocus1:.3f} µm, Defocus2: {defocus2:.3f} µm, score: {final_score:.6f}", CTF.DebugLevel.DETAILED)
        # Update parameters with final refined values
        self.__params[0] = defocus1
        self.__params[1] = defocus2

    def __optimize_defocus_linear(self, data: torch.Tensor, score_function) -> tuple[float, float, float]:
        """
        Linear grid search optimization for defocus parameters using batch scoring.
        
        Args:
            data: Cropped and masked data tensor
            score_function: Batch score function to evaluate score(data, defocus_tensor)
            
        Returns:
            tuple: (best_defocus1, best_defocus2, best_score)
        """        
        # Create defocus grid for optimization
        defocus_min, defocus_max, defocus_step = self.get_fit_params()[3:6] # defocus min, max, step
        defocus_values = torch.arange(defocus_min, defocus_max + defocus_step, defocus_step, device=data.device)
        n_defocus = len(defocus_values)
        
        self.__debug_print(f"[CTF::optimize_defocus_linear] Optimizing over {n_defocus}x{n_defocus} defocus grid using batch scoring", CTF.DebugLevel.DETAILED)
        
        # Create all combinations of defocus parameters as a batch tensor
        df1_grid, df2_grid = torch.meshgrid(defocus_values, defocus_values, indexing='ij')
        defocus_pairs = torch.stack([df1_grid.flatten(), df2_grid.flatten()], dim=1)  # [N*N, 2]
        
        # Use batch scoring for all combinations at once
        all_scores = score_function(data, defocus_pairs)  # [N*N]
        
        # Find the best score
        best_idx = torch.argmax(all_scores)
        best_defocus1 = defocus_pairs[best_idx, 0].item()
        best_defocus2 = defocus_pairs[best_idx, 1].item()
        best_score = all_scores[best_idx].item()
        
        self.__debug_print(f"[CTF::optimize_defocus_linear] Batch scoring completed: best score {best_score:.6f} at df1={best_defocus1:.3f}, df2={best_defocus2:.3f}", CTF.DebugLevel.DETAILED)
        
        return best_defocus1, best_defocus2, best_score

    def __optimize_defocus_gradient(self, data: torch.Tensor, max_iterations: int, tol: float, eps: float, score_function) -> tuple[float, float, float]:
        """
        Multi-start gradient ascent optimization for defocus parameters.
        Uses vectorized batch optimization for GPU parallelism.
        
        Args:
            data: Cropped and masked data tensor
            max_iterations: Maximum number of iterations for optimization
            eps: Small value to avoid division by zero in optimization
            score_function: Batch score function to evaluate score(data, defocus_tensor)
            
        Returns:
            tuple: (refined_defocus1, refined_defocus2, final_score)
        """
        defocus_min = self.get_fit_params()[3].item()  # defocus min from fit params
        defocus_max = self.get_fit_params()[4].item()  # defocus max
        
        # Multi-start optimization to avoid local minima
        n_starts = 8  # Number of parallel starts
        
        self.__debug_print(f"[CTF::optimize_defocus_gradient] Starting vectorized multi-start gradient ascent with {n_starts} starts", CTF.DebugLevel.DETAILED)
        
        # Initialize all starting points as a batch tensor
        initial_batch = torch.zeros((n_starts, 2), device=data.device, requires_grad=True)
        
        # First start uses initial params, others are random
        with torch.no_grad():
            initial_batch[0] = torch.tensor([self.__params[0].item(), self.__params[1].item()], device=data.device)
            if n_starts > 1:
                initial_batch[1:] = torch.rand((n_starts-1, 2), device=data.device) * (defocus_max - defocus_min) + defocus_min
        
        # Create batch of parameter tensors for parallel optimization
        defocus_params_batch = initial_batch.clone().detach().requires_grad_(True)
        
        # Use a single optimizer for the entire batch
        optimizer = torch.optim.Adam([defocus_params_batch], lr=0.05, eps=eps, weight_decay=1e-4)
        
        # Track convergence status for each start
        converged = torch.zeros(n_starts, dtype=torch.bool, device=data.device)
        last_params = torch.zeros_like(defocus_params_batch)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Use vectorized batch scoring for all starts simultaneously
            active_mask = ~converged
            if active_mask.any():
                active_params = defocus_params_batch[active_mask]
                batch_scores = score_function(data, active_params)
                # Negate for gradient ascent (optimizer minimizes)
                total_loss = -batch_scores.sum()
                
                if total_loss.requires_grad:
                    total_loss.backward()
                    optimizer.step()
            
            # Clamp all parameters to valid range
            with torch.no_grad():
                defocus_params_batch[:, 0] = torch.clamp(defocus_params_batch[:, 0], defocus_min, defocus_max)
                defocus_params_batch[:, 1] = torch.clamp(defocus_params_batch[:, 1], defocus_min, defocus_max)
            
            # Check convergence for all starts
            if iteration > 0:
                with torch.no_grad():
                    param_changes = torch.norm(defocus_params_batch - last_params, dim=1)
                    param_norms = torch.norm(defocus_params_batch, dim=1)
                    rel_errors = param_changes / (param_norms + 1e-8)
                    
                    newly_converged = (rel_errors < tol) & (~converged)
                    converged = converged | newly_converged
                    
                    if newly_converged.any():
                        for i in torch.where(newly_converged)[0]:
                            self.__debug_print(f"[CTF::optimize_defocus_gradient] Start {i+1} converged at iteration {iteration}", CTF.DebugLevel.DETAILED)
                    
                    # If all converged, break early
                    if converged.all():
                        self.__debug_print(f"[CTF::optimize_defocus_gradient] All starts converged at iteration {iteration}", CTF.DebugLevel.DETAILED)
                        break
            
            last_params = defocus_params_batch.clone().detach()
        
        # Evaluate final scores for all starts using batch scoring
        with torch.no_grad():
            final_scores = score_function(data, defocus_params_batch)
        
        # Find the best result
        best_idx = torch.argmax(final_scores)
        best_params = defocus_params_batch[best_idx].detach()
        best_score = final_scores[best_idx].item()
        
        # Debug output for all results
        for i in range(n_starts):
            self.__debug_print(f"[CTF::optimize_defocus_gradient] Start {i+1}: final=[{defocus_params_batch[i, 0].item():.3f}, {defocus_params_batch[i, 1].item():.3f}], score={final_scores[i].item():.6f}", CTF.DebugLevel.DETAILED)
        
        self.__debug_print(f"[CTF::optimize_defocus_gradient] Best result (start {best_idx+1}): [{best_params[0].item():.3f}, {best_params[1].item():.3f}], score={best_score:.6f}", CTF.DebugLevel.DETAILED)
        
        return best_params[0].item(), best_params[1].item(), best_score

    def __optimize_defocus_annealing(self, data: torch.Tensor, max_iterations: int, score_function) -> tuple[float, float, float]:
        """
        Simulated annealing optimization for defocus parameters using batch scoring.
        
        Args:
            data: Cropped and masked data tensor
            max_iterations: Maximum number of iterations
            score_function: Batch score function to evaluate score(data, defocus_tensor)
            
        Returns:
            tuple: (best_defocus1, best_defocus2, best_score)
        """
        defocus_min = self.get_fit_params()[3].item()
        defocus_max = self.get_fit_params()[4].item()
        
        # Initialize with current parameters
        current_defocus = torch.tensor([self.__params[0].item(), self.__params[1].item()], device=data.device)
        
        # Use batch scoring with current defocus
        current_batch = current_defocus.unsqueeze(0)  # [1, 2]
        current_score = score_function(data, current_batch)[0].item()
        
        # Keep track of best solution found
        best_defocus = current_defocus.clone()
        best_score = current_score
        
        # Annealing parameters
        initial_temp = 1.0
        final_temp = 0.001
        
        self.__debug_print(f"[CTF::optimize_defocus_annealing] Starting simulated annealing with batch scoring", CTF.DebugLevel.DETAILED)
        
        for iteration in range(max_iterations):
            # Temperature schedule (exponential decay)
            temperature = initial_temp * (final_temp / initial_temp) ** (iteration / max_iterations)
            
            # Generate random perturbation
            perturbation = torch.randn(2, device=data.device) * 0.1 * temperature
            new_defocus = current_defocus + perturbation
            
            # Clamp to valid range
            new_defocus = torch.clamp(new_defocus, defocus_min, defocus_max)
            
            # Evaluate new solution using batch scoring
            new_batch = new_defocus.unsqueeze(0)  # [1, 2]
            new_score = score_function(data, new_batch)[0].item()
            
            # Accept or reject based on Metropolis criterion
            delta_score = new_score - current_score
            
            if delta_score > 0 or torch.rand(1, device=data.device).item() < torch.exp(torch.tensor(delta_score / temperature, device=data.device)).item():
                # Accept the new solution
                current_defocus = new_defocus.clone()
                current_score = new_score.item() if torch.is_tensor(new_score) else new_score
                
                # Update best if improved
                if new_score > best_score:
                    best_defocus = new_defocus.clone()
                    best_score = new_score.item() if torch.is_tensor(new_score) else new_score
                    self.__debug_print(f"[CTF::optimize_defocus_annealing] New best at iteration {iteration}: [{best_defocus[0].item():.3f}, {best_defocus[1].item():.3f}], score={best_score:.6f}", CTF.DebugLevel.DETAILED)
            
            # Progress reporting
            if self.__debug_level >= CTF.DebugLevel.DETAILED and iteration % max(1, max_iterations // 10) == 0:
                self.__debug_print(f"[CTF::optimize_defocus_annealing] Iteration {iteration}: temp={temperature:.4f}, current_score={current_score:.6f}, best_score={best_score:.6f}", CTF.DebugLevel.DETAILED)
        
        return best_defocus[0].item(), best_defocus[1].item(), best_score

    def __angle_search_sequential(self, crop_masked: torch.Tensor, accuracy: float = 0.01):
        """
        Sequential angle search for astigmatism estimation.
        Returns: tensor of shape (num_angles, 2) with [angle, overlap].
        """
        def __F_rotate_tensor(img: torch.Tensor, angle_rad: float) -> torch.Tensor:
            if len(img.shape) != 2:
                raise ValueError("Input tensor must be 2D")
            theta = torch.tensor([
                [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                [torch.sin(angle_rad),  torch.cos(angle_rad), 0]
            ], device=img.device)
            theta = theta.unsqueeze(0)  # [1, 2, 3]
            N, C, H, W = 1, 1, img.shape[0], img.shape[1]
            grid = F.affine_grid(theta, size=(N, C, H, W), align_corners=False)
            img_batch = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            rotated = F.grid_sample(img_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            return rotated.squeeze(0).squeeze(0)

        # Apply ray mask
        ray_mask = self.get_ray_mask(crop_masked.shape)
        crop_masked = crop_masked * ray_mask
        # Mirror horizontally
        mirrored = torch.flip(crop_masked, dims=[1])

        angles = torch.arange(0, torch.pi, accuracy, device=crop_masked.device)
        overlap_list = []
        for angle in angles:
            rotated = __F_rotate_tensor(mirrored, angle)
            overlap = (crop_masked * rotated).sum().item()
            overlap_list.append(overlap)
        overlap_tensor = torch.tensor(overlap_list, device=crop_masked.device)
        result = torch.stack((angles / 2, overlap_tensor), dim=1)
        return result

    def __angle_search_parallel(self, crop_masked: torch.Tensor, accuracy: float = 0.01):
        """
        Parallel angle search for astigmatism estimation.
        Returns: tensor of shape (num_angles, 2) with [angle, overlap].
        """
        # Apply ray mask
        ray_mask = self.get_ray_mask(crop_masked.shape)
        crop_masked = crop_masked * ray_mask
        # Mirror horizontally
        mirrored = torch.flip(crop_masked, dims=[1])

        angles = torch.arange(0, torch.pi, accuracy, device=crop_masked.device)
        n_angles = angles.shape[0]
        theta = torch.zeros((n_angles, 2, 3), device=mirrored.device)
        theta[:,0,0] = torch.cos(angles)
        theta[:,0,1] = -torch.sin(angles)
        theta[:,1,0] = torch.sin(angles)
        theta[:,1,1] = torch.cos(angles)
        N, C, H, W = n_angles, 1, mirrored.shape[0], mirrored.shape[1]
        grid = F.affine_grid(theta, size=(N, C, H, W), align_corners=False)
        img_batch = mirrored.unsqueeze(0).unsqueeze(0).expand(n_angles, -1, -1, -1).clone()
        rotated_batch = F.grid_sample(img_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        rotated_batch = rotated_batch.squeeze(1)  # [n_angles, H, W]
        crop_masked_exp = crop_masked.unsqueeze(0).expand(n_angles, -1, -1)
        angle_buffer = (crop_masked_exp * rotated_batch).sum(dim=(1,2))
        result = torch.stack((angles / 2, angle_buffer), dim=1)
        return result
    
    def get_fft_data(self, data: torch.Tensor, N_frames: int, eps: float) -> torch.Tensor:
        if len(data.shape) == 2:
            return self.__shifted_fft(data, eps)
        if len(data.shape) != 3:
            raise ValueError("Data must be either 2D or 3D tensor")
        
        # Multi-frame data
        if N_frames <= 0 or N_frames > data.shape[0]:
            N_frames = data.shape[0]

        frames, height, width = data.shape
        n = frames // N_frames
        # TODO: make this parallel
        sums = torch.zeros((n, height, width), device=data.device)
        for i in range(n):
            start = max(i * N_frames, 0)
            end = min((i + 1) * N_frames, frames) 
            frame = data[start:end].sum(dim=0)
            sums[i] = self.__shifted_fft(frame, eps)
        return sums
        
        
    def __shifted_fft(self, data: torch.Tensor, eps: float) -> torch.Tensor:
        # Compute the FFT of the data
        f = torch.fft.fft2(data)
        fshift = torch.fft.fftshift(f)
        return torch.log(torch.abs(fshift) + eps)
    
    def __debug_print(self, msg: str, level: int = 0):
        if self.__debug_level >= level:
            print(f"[CTF] {msg}")
    
    def __cleanup_resources(self):
        """
        Clean up cached resources and free GPU memory after fitting is complete.
        """
        # Clear cached masks
        self.__g_mask = None
        self.__ray_mask = None
        
        # Force garbage collection and clear GPU cache if using CUDA
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.__debug_print("[CTF::cleanup] GPU memory cache cleared", CTF.DebugLevel.TIMING)