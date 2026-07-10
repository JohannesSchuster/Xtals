#!/usr/bin/env python3
"""
Simple Intensity Analysis Plotter (No pandas dependency)

This script reads peak coordinates and brightness analysis files using only
standard libraries and matplotlib, then creates intensity plots.

Now supports both legacy format (separate .coords and .max files) and 
new unified format (single .txt file with sections). Also supports glob patterns
for Windows compatibility.

Usage:
    python plot_intensity_simple.py analysis.txt  # Single unified file
    python plot_intensity_simple.py coords.csv brightness.csv  # Legacy format
    python plot_intensity_simple.py "folder/*.txt"  # Glob pattern (use quotes on Windows)
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

def load_unified_analysis_file(filename):
    """
    Load data from the new unified analysis file format.
    
    Returns:
        tuple: (coordinates_data, analysis_data, metadata)
    """
    try:
        coords = []
        analysis_data = {'frames': [], 'peaks': {}}
        metadata = {
            "source_filename": "Unknown", 
            "pixel_size": 1.0, 
            "peak_count": 0, 
            "frame_count": 0,
            "image_dimensions": "Unknown",
            "analysis_type": "Unknown"
        }
        
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse header section
        coordinate_section_start = -1
        analysis_section_start = -1
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Parse metadata from header
            if line.startswith('# Peak Analysis Results -'):
                if 'Brightness Analysis' in line:
                    metadata["analysis_type"] = "brightness"
                elif 'Gaussian Fitting' in line:
                    metadata["analysis_type"] = "gaussian"
            elif line.startswith('# Source file:'):
                metadata["source_filename"] = line.replace('# Source file:', '').strip()
            elif line.startswith('# Pixel size:'):
                try:
                    metadata["pixel_size"] = float(line.split(':')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('# Frame count:'):
                try:
                    metadata["frame_count"] = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('# Peak count:'):
                try:
                    metadata["peak_count"] = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('# Image dimensions:'):
                metadata["image_dimensions"] = line.split(':')[1].strip()
            
            # Find section boundaries
            elif line.startswith('# SECTION 1: PEAK COORDINATES'):
                coordinate_section_start = i + 2  # Skip the separator line and header
            elif line.startswith('# SECTION 2:'):
                analysis_section_start = i + 2  # Skip the separator line and header
        
        # Parse coordinates section
        if coordinate_section_start > 0:
            for i in range(coordinate_section_start, len(lines)):
                line = lines[i].strip()
                if line.startswith('#') or not line:
                    continue
                if line.startswith('# SECTION'):
                    break
                
                # Parse coordinate data
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    coords.append({
                        'Peak_ID': int(parts[0]),
                        'X_Pixel': float(parts[1]),
                        'Y_Pixel': float(parts[2]),
                        'Gx_1_per_A': float(parts[3]),
                        'Gy_1_per_A': float(parts[4]),
                        'G_Magnitude_1_per_A': float(parts[5]),
                        'Resolution_Angstrom': float(parts[6])
                    })
        
        # Parse analysis section
        if analysis_section_start > 0:
            header_parsed = False
            peak_columns = []
            
            for i in range(analysis_section_start, len(lines)):
                line = lines[i].strip()
                if line.startswith('#'):
                    continue
                if not line:
                    continue
                
                if not header_parsed:
                    # First data line - determine structure
                    parts = [p.strip() for p in line.split(',')]
                    if metadata["analysis_type"] == "brightness":
                        peak_columns = [f'Peak_{i+1}' for i in range(len(parts)-1)]
                    else:  # gaussian
                        # For gaussian: Amplitude_1, Sigma_x_1, Sigma_y_1, Amplitude_2, etc.
                        num_peaks = (len(parts) - 1) // 3
                        for peak_idx in range(num_peaks):
                            peak_columns.extend([
                                f'Amplitude_{peak_idx+1}',
                                f'Sigma_x_{peak_idx+1}',
                                f'Sigma_y_{peak_idx+1}'
                            ])
                    
                    # Initialize peak data storage
                    for peak_col in peak_columns:
                        analysis_data['peaks'][peak_col] = []
                    header_parsed = True
                
                # Parse data line
                parts = [p.strip() for p in line.split(',')]
                analysis_data['frames'].append(int(parts[0]))  # Frame number
                
                # Read analysis values
                for j, peak_col in enumerate(peak_columns):
                    if j + 1 < len(parts):
                        analysis_data['peaks'][peak_col].append(float(parts[j + 1]))
        
        print(f"Loaded unified analysis file: {filename}")
        print(f"Analysis type: {metadata['analysis_type']}")
        print(f"Coordinates: {len(coords)} peaks")
        print(f"Analysis data: {len(analysis_data['frames'])} frames")
        print(f"Source file: {metadata['source_filename']}")
        print(f"Pixel size: {metadata['pixel_size']:.6f} A/px")
        
        return coords, analysis_data, metadata
    
    except Exception as e:
        print(f"Error loading unified analysis file {filename}: {e}")
        return None, None, {"source_filename": "Unknown", "pixel_size": 1.0, "peak_count": 0, "frame_count": 0, "analysis_type": "unknown"}

def convert_to_brightness_format(analysis_data, metadata):
    """Convert analysis data (brightness or gaussian) to brightness format for plotting."""
    brightness_data = {'frames': analysis_data['frames'], 'peaks': {}}
    
    if metadata["analysis_type"] == "brightness":
        # Direct copy for brightness analysis
        brightness_data['peaks'] = analysis_data['peaks'].copy()
    elif metadata["analysis_type"] == "gaussian":
        # For Gaussian fitting, extract just the amplitude values
        num_peaks = metadata.get("peak_count", 0)
        for peak_idx in range(num_peaks):
            amplitude_key = f'Amplitude_{peak_idx+1}'
            if amplitude_key in analysis_data['peaks']:
                brightness_data['peaks'][f'Peak_{peak_idx+1}'] = analysis_data['peaks'][amplitude_key]
    
    return brightness_data

def load_coordinates_csv(filename):
    """Load peak coordinates from CSV file using csv module."""
    try:
        coords = []
        metadata = {"source_filename": "Unknown", "pixel_size": 1.0, "peak_count": 0, "image_dimensions": "Unknown"}
        
        with open(filename, 'r', encoding='utf-8') as f:
            # Read header comments to extract metadata
            pos = f.tell()
            line = f.readline().strip()
            
            while line.startswith('#'):
                if 'Source file:' in line:
                    metadata["source_filename"] = line.replace('# Source file:', '').strip()
                elif 'Pixel size:' in line:
                    try:
                        metadata["pixel_size"] = float(line.split(':')[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'Peak count:' in line:
                    try:
                        metadata["peak_count"] = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Image dimensions:' in line:
                    metadata["image_dimensions"] = line.split(':')[1].strip()
                
                pos = f.tell()
                line = f.readline().strip()
            
            # Reset to position before the non-comment line
            f.seek(pos)
            
            # Use DictReader for the CSV data
            reader = csv.DictReader(f)
            
            for row in reader:
                # Strip whitespace from all values
                row = {k.strip(): v.strip() for k, v in row.items()}
                
                g_mag = float(row['G_Magnitude_1_per_A']) if row['G_Magnitude_1_per_A'] else 0
                resolution = float(row.get('Resolution_Angstrom', 1.0/g_mag if g_mag > 0 else 0))
                
                coords.append({
                    'Peak_ID': int(row['Peak_ID']),
                    'X_Pixel': float(row['X_Pixel']),
                    'Y_Pixel': float(row['Y_Pixel']),
                    'Gx_1_per_A': float(row['Gx_1_per_A']),
                    'Gy_1_per_A': float(row['Gy_1_per_A']),
                    'G_Magnitude_1_per_A': g_mag,
                    'Resolution_Angstrom': resolution
                })
        
        print(f"Loaded {len(coords)} peak coordinates from {filename}")
        print(f"Source file: {metadata['source_filename']}")
        print(f"Pixel size: {metadata['pixel_size']:.6f} A/px")
        print(f"Image dimensions: {metadata['image_dimensions']}")
        return coords, metadata
    except Exception as e:
        print(f"Error loading coordinates file {filename}: {e}")
        return None, {"source_filename": "Unknown", "pixel_size": 1.0, "peak_count": 0, "image_dimensions": "Unknown"}

def load_brightness_csv(filename):
    """Load brightness analysis data from CSV file using csv module."""
    try:
        brightness_data = {'frames': [], 'peaks': {}}
        metadata = {"source_filename": "Unknown", "pixel_size": 1.0, "peak_count": 0, "frame_count": 0}
        
        with open(filename, 'r', encoding='utf-8') as f:
            # Read header comments to extract metadata
            pos = f.tell()
            line = f.readline().strip()
            
            while line.startswith('#'):
                if 'Source file:' in line:
                    metadata["source_filename"] = line.replace('# Source file:', '').strip()
                elif 'Pixel size:' in line:
                    try:
                        metadata["pixel_size"] = float(line.split(':')[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'Peak count:' in line:
                    try:
                        metadata["peak_count"] = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Frame count:' in line:
                    try:
                        metadata["frame_count"] = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                
                pos = f.tell()
                line = f.readline().strip()
            
            # Reset to position before the non-comment line
            f.seek(pos)
            
            # Read header and data
            reader = csv.reader(f)
            header = next(reader)  # Read header
            
            # Strip whitespace from header and find peak columns
            header = [col.strip() for col in header]
            peak_columns = [col for col in header[1:] if col.startswith('Peak_')]
            
            # Initialize peak data storage
            for peak_col in peak_columns:
                brightness_data['peaks'][peak_col] = []
            
            # Read data rows
            for row in reader:
                # Strip whitespace from all row values
                row = [val.strip() for val in row]
                
                brightness_data['frames'].append(int(row[0]))  # Frame number
                
                # Read peak intensities
                for i, peak_col in enumerate(peak_columns):
                    brightness_data['peaks'][peak_col].append(float(row[i + 1]))
        
        print(f"Loaded brightness data for {len(brightness_data['frames'])} frames from {filename}")
        print(f"Number of peaks in brightness data: {len(peak_columns)}")
        print(f"Source file: {metadata['source_filename']}")
        print(f"Pixel size: {metadata['pixel_size']:.6f} A/px")
        print(f"Frame count: {metadata['frame_count']}")
        return brightness_data, metadata
    except Exception as e:
        print(f"Error loading brightness file {filename}: {e}")
        return None, {"source_filename": "Unknown", "pixel_size": 1.0, "peak_count": 0, "frame_count": 0}

def load_legacy_format(coords_file, brightness_file):
    """
    Load data from legacy format (separate coordinates and brightness files).
    
    Returns:
        tuple: (coordinates_data, brightness_data, combined_metadata)
    """
    coords, coords_meta = load_coordinates_csv(coords_file)
    brightness_data, brightness_meta = load_brightness_csv(brightness_file)
    
    if coords is None or brightness_data is None:
        return None, None, None
    
    # Combine metadata
    combined_meta = {
        'source_filename': coords_meta.get('source_filename', brightness_meta.get('source_filename', 'Unknown')),
        'pixel_size': coords_meta.get('pixel_size', brightness_meta.get('pixel_size', 1.0)),
        'peak_count': max(coords_meta.get('peak_count', 0), brightness_meta.get('peak_count', 0)),
        'frame_count': brightness_meta.get('frame_count', len(brightness_data['frames'])),
        'image_dimensions': coords_meta.get('image_dimensions', 'Unknown'),
        'analysis_type': 'brightness'  # Legacy format is always brightness
    }
    
    return coords, brightness_data, combined_meta

def create_resolution_batch_plots(coords, brightness_data, frames, peak_columns, output_dir, batch_ranges=None, source_filename="Unknown"):
    """Create plots grouped by resolution ranges."""
    
    # Default resolution ranges if none provided
    if batch_ranges is None:
        batch_ranges = [30, 20, 10, 5]
    
    # Create resolution ranges from the batch_ranges list
    resolution_ranges = []
    prev_val = float('inf')
    
    for i, val in enumerate(batch_ranges):
        range_label = f'{prev_val:.0f}-{val}' if prev_val != float('inf') else f'inf-{val}'
        resolution_ranges.append((prev_val, val, f'{range_label} Å'))
        prev_val = val
    
    # Add final range from last value to 0
    if batch_ranges:
        range_label = f'{batch_ranges[-1]}-0'
        resolution_ranges.append((batch_ranges[-1], 0, f'{range_label} Å'))
    
    print(f"Resolution ranges: {[r[2] for r in resolution_ranges]}")
    
    # Group peaks by resolution ranges
    resolution_groups = {range_label: [] for _, _, range_label in resolution_ranges}
    
    for i, peak_col in enumerate(peak_columns):
        if i < len(coords):
            resolution = coords[i]['Resolution_Angstrom']
            intensities = np.array(brightness_data['peaks'][peak_col])
            
            # Use raw intensities without normalization
            raw_intensities = intensities
            
            # Find which range this peak belongs to
            for range_max, range_min, range_label in resolution_ranges:
                if range_min < resolution <= range_max:
                    resolution_groups[range_label].append({
                        'peak_id': i + 1,
                        'resolution': resolution,
                        'intensities': raw_intensities,
                        'peak_col': peak_col
                    })
                    break
    
    # Create resolution batch plots
    n_ranges = len(resolution_ranges)
    n_cols = 3
    n_rows = (n_ranges + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, (range_label, peaks_in_range) in enumerate(resolution_groups.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        if not peaks_in_range:
            ax.set_title(f'Resolution Range: {range_label}\n(No peaks found)')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Normalized Intensity')
            ax.grid(True, alpha=0.3)
            ax.text(0.5, 0.5, 'No peaks in this range', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            continue
        
        # Plot all peaks in this resolution range
        for i, peak_data in enumerate(peaks_in_range):
            color = colors[i % len(colors)]
            ax.plot(frames, peak_data['intensities'], 
                   color=color, linewidth=1.5, alpha=0.7,
                   label=f"Peak {peak_data['peak_id']} ({peak_data['resolution']:.1f} Å)")
        
        # Calculate and plot average for this resolution range
        if len(peaks_in_range) > 1:
            all_intensities = np.array([peak['intensities'] for peak in peaks_in_range])
            mean_intensities = np.mean(all_intensities, axis=0)
            ax.plot(frames, mean_intensities, 'k--', linewidth=2, alpha=0.8, 
                   label=f'Average ({len(peaks_in_range)} peaks)')
        
        ax.set_title(f'Resolution Range: {range_label}\n({len(peaks_in_range)} peaks)')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Intensity (FFT Magnitude)')
        ax.grid(True, alpha=0.3)
        # Remove fixed ylim to allow auto-scaling for raw intensities
        
        # Only show legend if not too many peaks
        if len(peaks_in_range) <= 8:
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.text(0.02, 0.98, f'{len(peaks_in_range)} peaks\n(legend hidden)', 
                   transform=ax.transAxes, va='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(resolution_ranges), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.suptitle(f'Intensity by Resolution Ranges\nSource: {source_filename}', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'resolution_batch_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plot showing average intensity per resolution range
    plt.figure(figsize=(12, 8))
    
    range_labels = []
    avg_intensities_per_range = []
    std_intensities_per_range = []
    peak_counts = []
    
    for range_label, peaks_in_range in resolution_groups.items():
        if peaks_in_range:
            all_intensities = np.concatenate([peak['intensities'] for peak in peaks_in_range])
            range_labels.append(range_label)
            avg_intensities_per_range.append(np.mean(all_intensities))
            std_intensities_per_range.append(np.std(all_intensities))
            peak_counts.append(len(peaks_in_range))
    
    if range_labels:
        x_pos = np.arange(len(range_labels))
        bars = plt.bar(x_pos, avg_intensities_per_range, yerr=std_intensities_per_range, 
                      capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add peak count labels on bars
        for i, (bar, count) in enumerate(zip(bars, peak_counts)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std_intensities_per_range[i] + 0.01,
                    f'{count} peaks', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Resolution Range')
        plt.ylabel('Average Intensity (FFT Magnitude)')
        plt.title(f'Average Intensity by Resolution Range\nSource: {source_filename}')
        plt.xticks(x_pos, range_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resolution_range_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create time series plot with average intensities and standard deviation areas
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(resolution_groups)))
    
    for idx, (range_label, peaks_in_range) in enumerate(resolution_groups.items()):
        if not peaks_in_range:
            continue
        
        # Calculate mean and std for each frame across all peaks in this resolution range
        all_intensities_array = np.array([peak['intensities'] for peak in peaks_in_range])
        mean_per_frame = np.mean(all_intensities_array, axis=0)
        std_per_frame = np.std(all_intensities_array, axis=0)
        
        # Normalize the averages to 0-1 range
        min_mean = np.min(mean_per_frame)
        max_mean = np.max(mean_per_frame)
        if max_mean > min_mean:
            normalized_mean = (mean_per_frame - min_mean) / (max_mean - min_mean)
        else:
            normalized_mean = np.zeros_like(mean_per_frame)
        
        # Also normalize the standard deviation proportionally
        if max_mean > min_mean:
            normalized_std = std_per_frame / (max_mean - min_mean)
        else:
            normalized_std = np.zeros_like(std_per_frame)
        
        color = colors[idx % len(colors)]
        
        # Plot the normalized mean line
        plt.plot(frames, normalized_mean, color=color, linewidth=2, 
                label=f'{range_label} (n={len(peaks_in_range)})', alpha=0.9)
        
        # Add standard deviation area with 50% transparency
        plt.fill_between(frames, 
                        normalized_mean - normalized_std/10, 
                        normalized_mean + normalized_std/10,
                        color=color, alpha=0.5, linewidth=0)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Average Intensity')
    plt.title(f'Average Intensity by Resolution Range (Time Series)\nSource: {source_filename}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resolution_batch_timeseries.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_peak_intensities(coords, brightness_data, output_dir="plots", max_peaks=20, batch_ranges=None, source_filename="Unknown", metadata=None):
    """Create intensity plots for peaks."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    frames = np.array(brightness_data['frames'])
    peak_columns = list(brightness_data['peaks'].keys())
    n_peaks = min(len(peak_columns), max_peaks)
    
    print(f"Creating plots for first {n_peaks} peaks...")
    
    # Create overview plot with multiple peaks
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_peaks))
    
    for i in range(n_peaks):
        peak_col = peak_columns[i]
        intensities = np.array(brightness_data['peaks'][peak_col])
        plt.plot(frames, intensities, color=colors[i], linewidth=1.5, 
                label=f'Peak {i+1}', alpha=0.8)
    
    plt.title(f'Intensity vs Frame Number - First {n_peaks} Peaks\nSource: {source_filename}')
    plt.xlabel('Frame Number')
    plt.ylabel('Intensity (FFT Magnitude)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intensity_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for first few peaks
    n_individual = min(6, n_peaks)  # Show first 6 peaks individually
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_individual):
        ax = axes[i]
        peak_col = peak_columns[i]
        intensities = np.array(brightness_data['peaks'][peak_col])
        
        # Normalize intensities (0-1 range)
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
        if max_intensity > min_intensity:
            normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
        else:
            normalized_intensities = np.zeros_like(intensities)
        
        # Get coordinate information
        if i < len(coords):
            coord = coords[i]
            x_pixel = coord['X_Pixel']
            y_pixel = coord['Y_Pixel']
            g_magnitude = coord['G_Magnitude_1_per_A']
            resolution = coord['Resolution_Angstrom']
        else:
            x_pixel = y_pixel = g_magnitude = resolution = 0
        
        ax.plot(frames, normalized_intensities, 'b-', linewidth=1.5, marker='o', markersize=2)
        ax.set_title(f'Peak {i+1} (Normalized)\nPos: ({x_pixel:.1f}, {y_pixel:.1f}) px\n'
                    f'|g| = {g_magnitude:.4f} Å⁻¹, Res = {resolution:.2f} Å')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Normalized Intensity')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add statistics for normalized data
        mean_norm = np.mean(normalized_intensities)
        std_norm = np.std(normalized_intensities)
        ax.axhline(y=mean_norm, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=mean_norm + std_norm, color='orange', linestyle=':', alpha=0.7)
        ax.axhline(y=mean_norm - std_norm, color='orange', linestyle=':', alpha=0.7)
    
    # Hide unused subplots
    for i in range(n_individual, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for suptitle
    plt.suptitle(f'Individual Peak Analysis (Normalized)\nSource: {source_filename}', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'individual_peaks_sample.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Resolution-based batch plots
    if coords:
        create_resolution_batch_plots(coords, brightness_data, frames, peak_columns, output_dir, batch_ranges, source_filename)
    
    # Peak positions plot
    if coords:
        plt.figure(figsize=(12, 10))
        
        # Calculate average intensities for each peak
        avg_intensities = []
        for i, peak_col in enumerate(peak_columns):
            if i < len(coords):
                avg_intensities.append(np.mean(brightness_data['peaks'][peak_col]))
            else:
                break
        
        x_positions = [coord['X_Pixel'] for coord in coords[:len(avg_intensities)]]
        y_positions = [coord['Y_Pixel'] for coord in coords[:len(avg_intensities)]]
        
        scatter = plt.scatter(x_positions, y_positions, c=avg_intensities, 
                            s=60, cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add peak numbers
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Average Intensity')
        plt.title(f'Peak Positions Colored by Average Intensity\nSource: {source_filename}')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'peak_positions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_statistics_report(coords, brightness_data, output_dir="plots", metadata=None):
    """Create a text report with statistics."""
    
    frames = brightness_data['frames']
    peak_columns = list(brightness_data['peaks'].keys())
    
    report_filename = os.path.join(output_dir, 'intensity_statistics.txt')
    
    with open(report_filename, 'w') as f:
        f.write("INTENSITY ANALYSIS STATISTICS\n")
        f.write("=" * 40 + "\n")
        
        # Write metadata if available
        if metadata:
            f.write("DATASET INFORMATION:\n")
            f.write(f"Source file: {metadata.get('source_filename', 'Unknown')}\n")
            f.write(f"Pixel size: {metadata.get('pixel_size', 1.0):.6f} A/px\n")
            f.write(f"Image dimensions: {metadata.get('image_dimensions', 'Unknown')}\n")
            f.write(f"Peak count: {metadata.get('peak_count', len(coords))}\n\n")
            
        f.write("ANALYSIS SUMMARY:\n")
        f.write(f"Number of frames: {len(frames)}\n")
        f.write(f"Number of peaks: {len(peak_columns)}\n")
        f.write(f"Frame range: {min(frames)} to {max(frames)}\n\n")
        
        f.write("PEAK STATISTICS:\n")
        f.write("Peak_ID, Mean, Std, Min, Max, CV%, X_Pixel, Y_Pixel, Resolution_A\n")
        
        for i, peak_col in enumerate(peak_columns):
            peak_id = i + 1
            intensities = np.array(brightness_data['peaks'][peak_col])
            
            mean_int = np.mean(intensities)
            std_int = np.std(intensities)
            min_int = np.min(intensities)
            max_int = np.max(intensities)
            cv = (std_int / mean_int * 100) if mean_int > 0 else 0
            
            # Get coordinate info if available
            if i < len(coords):
                coord = coords[i]
                x_pixel = coord['X_Pixel']
                y_pixel = coord['Y_Pixel']
                resolution = coord['Resolution_Angstrom']
            else:
                x_pixel = y_pixel = resolution = 0
            
            f.write(f"{peak_id}, {mean_int:.2f}, {std_int:.2f}, {min_int:.2f}, "
                   f"{max_int:.2f}, {cv:.1f}, {x_pixel:.1f}, {y_pixel:.1f}, {resolution:.2f}\n")
    
    print(f"Statistics report saved to {report_filename}")

def main():
    parser = argparse.ArgumentParser(description='Simple intensity analysis plotter with glob support')
    parser.add_argument('files', nargs='*', 
                       help='Analysis files: [unified_file] or [coords_file brightness_file] or glob pattern like "folder/*.txt"')
    parser.add_argument('--output-dir', '-o', default='plots',
                       help='Output directory for plots')
    parser.add_argument('--max-peaks', '-n', type=int, default=20,
                       help='Maximum number of peaks to plot (default: 20)')
    parser.add_argument('--batch-ranges', '-b', type=str, default='30,20,10,5',
                       help='Comma-separated resolution batch ranges in Angstroms (default: 30,20,10,5)')
    parser.add_argument('--title', '-t', type=str, default=None,
                       help='Override the title for plots (default: use source filename from data files)')
    
    args = parser.parse_args()
    
    # Parse batch ranges
    try:
        batch_ranges = [float(x.strip()) for x in args.batch_ranges.split(',')]
        batch_ranges.sort(reverse=True)  # Sort in descending order
        print(f"Using resolution batch ranges: {batch_ranges}")
    except ValueError:
        print(f"Error: Invalid batch ranges '{args.batch_ranges}'. Using default: [30, 20, 10, 5]")
        batch_ranges = [30, 20, 10, 5]
    
    if not args.files:
        print("Error: Please provide analysis files.")
        print("Examples:")
        print("  Single file: python plot_intensity_simple.py analysis.txt")
        print("  Legacy format: python plot_intensity_simple.py coords.csv brightness.csv")
        print("  Glob pattern: python plot_intensity_simple.py \"folder/*.txt\"")
        return 1
    
    # Expand glob patterns for Windows compatibility
    expanded_files = []
    for file_pattern in args.files:
        # Check if the pattern contains wildcards
        if '*' in file_pattern or '?' in file_pattern or '[' in file_pattern:
            # Expand the glob pattern
            matches = glob.glob(file_pattern)
            if matches:
                # For multiple matches, process each one individually
                if len(matches) == 1:
                    expanded_files.extend(matches)
                    print(f"Expanded '{file_pattern}' to: {matches[0]}")
                else:
                    print(f"Pattern '{file_pattern}' matched {len(matches)} files.")
                    print("For single-file analysis, please specify one file at a time.")
                    print("Available files:")
                    for match in matches:
                        print(f"  {match}")
                    return 1
            else:
                print(f"Warning: Pattern '{file_pattern}' matched no files")
        else:
            # Regular file, add as-is
            expanded_files.append(file_pattern)
    
    if not expanded_files:
        print("Error: No valid files found after expanding patterns!")
        return 1
    
    # Use expanded file list
    args.files = expanded_files
    
    # Check that all files exist
    for filename in args.files:
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found!")
            return 1
    
    coords = None
    brightness_data = None
    metadata = {"source_filename": "Unknown", "pixel_size": 1.0, "peak_count": 0, "frame_count": 0}
    
    # Determine file format and load data
    if len(args.files) == 1:
        # Try loading as unified format
        print(f"Attempting to load unified format file: {args.files[0]}")
        coords, analysis_data, metadata = load_unified_analysis_file(args.files[0])
        if coords is not None and analysis_data is not None:
            brightness_data = convert_to_brightness_format(analysis_data, metadata)
        else:
            print("Failed to load unified format file!")
            return 1
            
    elif len(args.files) == 2:
        # Legacy format: separate coordinates and brightness files
        coords_file, brightness_file = args.files
        print(f"Loading legacy format files: {coords_file}, {brightness_file}")
        
        coords, coords_metadata = load_coordinates_csv(coords_file)
        brightness_data, brightness_metadata = load_brightness_csv(brightness_file)
        
        if coords is None or brightness_data is None:
            print("Failed to load legacy format files!")
            return 1
        
        # Merge metadata from both files
        metadata = coords_metadata.copy()
        metadata.update(brightness_metadata)
        
    else:
        print("Error: Please provide either:")
        print("  1. One unified analysis file: python plot_intensity_simple.py analysis.txt")
        print("  2. Two legacy files: python plot_intensity_simple.py coords.csv brightness.csv")
        print("  3. Use quotes for glob patterns: python plot_intensity_simple.py \"folder/*.txt\"")
        return 1
    
    # Use the title argument or fall back to source filename from metadata
    if args.title:
        source_filename = args.title
    else:
        source_filename = metadata.get("source_filename", "Unknown")
    
    # Create plots
    print(f"Creating plots in directory: {args.output_dir}")
    plot_peak_intensities(coords, brightness_data, args.output_dir, args.max_peaks, batch_ranges, source_filename, metadata)
    create_statistics_report(coords, brightness_data, args.output_dir, metadata)
    
    print("Analysis complete!")
    print(f"Check the '{args.output_dir}' directory for:")
    print("  - intensity_overview.png (all peaks together)")
    print("  - individual_peaks_sample.png (first 6 peaks individually, normalized)")  
    print("  - peak_positions.png (peak positions colored by intensity)")
    print("  - resolution_batch_plots.png (peaks grouped by resolution ranges)")
    print("  - resolution_range_summary.png (average intensity per resolution range)")
    print("  - resolution_batch_timeseries.png (time series with std deviation areas)")
    print("  - intensity_statistics.txt (detailed statistics)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
