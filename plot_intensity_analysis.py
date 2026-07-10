#!/usr/bin/env python3
"""
Intensity Analysis Plotter

This script reads peak coordinates and brightness analysis files, then creates
plots showing intensity variations across frames for each peak.

Usage:
    python plot_intensity_analysis.py [coordinates_file] [brightness_file]

If no arguments are provided, defaults to:
    - peak_coordinates.txt
    - brightness_analysis.txt
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse

def load_coordinates(filename):
    """Load peak coordinates from CSV file."""
    try:
        coords = pd.read_csv(filename)
        print(f"Loaded {len(coords)} peak coordinates from {filename}")
        return coords
    except Exception as e:
        print(f"Error loading coordinates file {filename}: {e}")
        return None

def load_brightness_data(filename):
    """Load brightness analysis data from CSV file."""
    try:
        brightness = pd.read_csv(filename)
        print(f"Loaded brightness data for {len(brightness)} frames from {filename}")
        print(f"Number of peaks in brightness data: {len(brightness.columns) - 1}")
        return brightness
    except Exception as e:
        print(f"Error loading brightness file {filename}: {e}")
        return None

def create_intensity_plots(coords, brightness, output_dir="plots"):
    """Create intensity vs frame plots for all peaks."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get frame numbers
    frames = brightness['Frame'].values
    n_frames = len(frames)
    
    # Get peak columns (exclude 'Frame' column)
    peak_columns = [col for col in brightness.columns if col.startswith('Peak_')]
    n_peaks = len(peak_columns)
    
    print(f"Creating plots for {n_peaks} peaks across {n_frames} frames...")
    
    # Create individual plots for each peak
    for i, peak_col in enumerate(peak_columns):
        peak_id = i + 1
        
        # Get intensity data for this peak
        intensities = brightness[peak_col].values
        
        # Get coordinate information for this peak
        if peak_id <= len(coords):
            coord_info = coords.iloc[peak_id - 1]
            x_pixel = coord_info['X_Pixel']
            y_pixel = coord_info['Y_Pixel']
            g_magnitude = coord_info['G_Magnitude_1_per_A']
            resolution = coord_info.get('Resolution_Angstrom', 1.0 / g_magnitude if g_magnitude > 0 else 0)
        else:
            x_pixel = y_pixel = g_magnitude = resolution = 0
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Main intensity plot
        plt.subplot(2, 1, 1)
        plt.plot(frames, intensities, 'b-', linewidth=1.5, marker='o', markersize=3)
        plt.title(f'Peak {peak_id} - Intensity vs Frame\n'
                 f'Position: ({x_pixel:.1f}, {y_pixel:.1f}) px, '
                 f'|g| = {g_magnitude:.4f} Å⁻¹, Resolution = {resolution:.2f} Å')
        plt.xlabel('Frame Number')
        plt.ylabel('Intensity (FFT Magnitude)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
        
        plt.axhline(y=mean_intensity, color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_intensity:.1f}')
        plt.axhline(y=mean_intensity + std_intensity, color='orange', linestyle=':', alpha=0.7,
                   label=f'Mean + σ: {mean_intensity + std_intensity:.1f}')
        plt.axhline(y=mean_intensity - std_intensity, color='orange', linestyle=':', alpha=0.7,
                   label=f'Mean - σ: {mean_intensity - std_intensity:.1f}')
        plt.legend()
        
        # Histogram of intensities
        plt.subplot(2, 1, 2)
        plt.hist(intensities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=mean_intensity, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_intensity:.1f}')
        plt.axvline(x=mean_intensity + std_intensity, color='orange', linestyle=':', alpha=0.7)
        plt.axvline(x=mean_intensity - std_intensity, color='orange', linestyle=':', alpha=0.7)
        plt.title(f'Intensity Distribution (μ={mean_intensity:.1f}, σ={std_intensity:.1f})')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_filename = os.path.join(output_dir, f'peak_{peak_id:03d}_intensity.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print progress
        if (i + 1) % 10 == 0 or i == n_peaks - 1:
            print(f"  Processed {i + 1}/{n_peaks} peaks...")

def create_overview_plots(coords, brightness, output_dir="plots"):
    """Create overview plots showing multiple peaks."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    frames = brightness['Frame'].values
    peak_columns = [col for col in brightness.columns if col.startswith('Peak_')]
    
    # Plot 1: All peaks intensity traces (first 20 peaks to avoid clutter)
    plt.figure(figsize=(15, 10))
    
    max_peaks_to_show = min(20, len(peak_columns))
    colors = plt.cm.tab20(np.linspace(0, 1, max_peaks_to_show))
    
    for i in range(max_peaks_to_show):
        peak_col = peak_columns[i]
        intensities = brightness[peak_col].values
        plt.plot(frames, intensities, color=colors[i], linewidth=1, 
                label=f'Peak {i+1}', alpha=0.8)
    
    plt.title(f'Intensity Traces - First {max_peaks_to_show} Peaks')
    plt.xlabel('Frame Number')
    plt.ylabel('Intensity (FFT Magnitude)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview_intensity_traces.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Peak positions colored by average intensity
    if len(coords) > 0:
        plt.figure(figsize=(12, 10))
        
        # Calculate average intensities for each peak
        avg_intensities = []
        for peak_col in peak_columns:
            avg_intensities.append(np.mean(brightness[peak_col].values))
        
        # Extend avg_intensities if we have more coordinates than peaks
        while len(avg_intensities) < len(coords):
            avg_intensities.append(0)
        
        x_positions = coords['X_Pixel'].values
        y_positions = coords['Y_Pixel'].values
        avg_intensities = np.array(avg_intensities[:len(coords)])
        
        scatter = plt.scatter(x_positions, y_positions, c=avg_intensities, 
                            s=50, cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add peak numbers as annotations
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Average Intensity')
        plt.title('Peak Positions Colored by Average Intensity')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'peak_positions_by_intensity.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Resolution vs Average Intensity
    if len(coords) > 0 and 'Resolution_Angstrom' in coords.columns:
        plt.figure(figsize=(10, 8))
        
        resolutions = coords['Resolution_Angstrom'].values
        avg_intensities = np.array(avg_intensities[:len(coords)])
        
        plt.scatter(resolutions, avg_intensities, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(resolutions, avg_intensities, 1)
        p = np.poly1d(z)
        plt.plot(resolutions, p(resolutions), "r--", alpha=0.8, 
                label=f'Trend: y = {z[0]:.1f}x + {z[1]:.1f}')
        
        plt.xlabel('Resolution (Å)')
        plt.ylabel('Average Intensity')
        plt.title('Resolution vs Average Intensity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resolution_vs_intensity.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_report(coords, brightness, output_dir="plots"):
    """Create a summary report with statistics."""
    
    frames = brightness['Frame'].values
    peak_columns = [col for col in brightness.columns if col.startswith('Peak_')]
    
    report_lines = []
    report_lines.append("INTENSITY ANALYSIS SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Number of frames: {len(frames)}")
    report_lines.append(f"Number of peaks: {len(peak_columns)}")
    report_lines.append("")
    
    # Overall statistics
    all_intensities = []
    for peak_col in peak_columns:
        all_intensities.extend(brightness[peak_col].values)
    
    report_lines.append("OVERALL STATISTICS:")
    report_lines.append(f"  Total data points: {len(all_intensities)}")
    report_lines.append(f"  Overall mean intensity: {np.mean(all_intensities):.2f}")
    report_lines.append(f"  Overall std intensity: {np.std(all_intensities):.2f}")
    report_lines.append(f"  Overall min intensity: {np.min(all_intensities):.2f}")
    report_lines.append(f"  Overall max intensity: {np.max(all_intensities):.2f}")
    report_lines.append("")
    
    # Per-peak statistics
    report_lines.append("PER-PEAK STATISTICS:")
    report_lines.append("Peak_ID, Mean_Intensity, Std_Intensity, Min_Intensity, Max_Intensity, CV%, X_Pixel, Y_Pixel, Resolution_A")
    
    for i, peak_col in enumerate(peak_columns):
        peak_id = i + 1
        intensities = brightness[peak_col].values
        
        mean_int = np.mean(intensities)
        std_int = np.std(intensities)
        min_int = np.min(intensities)
        max_int = np.max(intensities)
        cv = (std_int / mean_int * 100) if mean_int > 0 else 0
        
        # Get coordinate info if available
        if peak_id <= len(coords):
            coord_info = coords.iloc[peak_id - 1]
            x_pixel = coord_info['X_Pixel']
            y_pixel = coord_info['Y_Pixel']
            resolution = coord_info.get('Resolution_Angstrom', 0)
        else:
            x_pixel = y_pixel = resolution = 0
        
        report_lines.append(f"{peak_id}, {mean_int:.2f}, {std_int:.2f}, {min_int:.2f}, "
                          f"{max_int:.2f}, {cv:.1f}, {x_pixel:.1f}, {y_pixel:.1f}, {resolution:.2f}")
    
    # Save report
    report_filename = os.path.join(output_dir, 'intensity_analysis_report.txt')
    with open(report_filename, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to {report_filename}")

def main():
    parser = argparse.ArgumentParser(description='Plot intensity analysis from peak coordinates and brightness data')
    parser.add_argument('coordinates', nargs='?', default='peak_coordinates.txt',
                       help='Path to peak coordinates CSV file (default: peak_coordinates.txt)')
    parser.add_argument('brightness', nargs='?', default='brightness_analysis.txt',
                       help='Path to brightness analysis CSV file (default: brightness_analysis.txt)')
    parser.add_argument('--output-dir', '-o', default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--individual', action='store_true',
                       help='Create individual plots for each peak (can be slow for many peaks)')
    parser.add_argument('--overview-only', action='store_true',
                       help='Create only overview plots (faster)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.coordinates):
        print(f"Error: Coordinates file '{args.coordinates}' not found!")
        return 1
    
    if not os.path.exists(args.brightness):
        print(f"Error: Brightness file '{args.brightness}' not found!")
        return 1
    
    # Load data
    print("Loading data files...")
    coords = load_coordinates(args.coordinates)
    brightness = load_brightness_data(args.brightness)
    
    if coords is None or brightness is None:
        print("Failed to load required data files!")
        return 1
    
    # Create plots
    print(f"Creating plots in directory: {args.output_dir}")
    
    if not args.overview_only:
        create_overview_plots(coords, brightness, args.output_dir)
    
    if args.individual:
        create_intensity_plots(coords, brightness, args.output_dir)
    
    if not args.overview_only and not args.individual:
        # Default: create overview plots and summary
        create_overview_plots(coords, brightness, args.output_dir)
    
    # Always create summary report
    create_summary_report(coords, brightness, args.output_dir)
    
    print("Analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
