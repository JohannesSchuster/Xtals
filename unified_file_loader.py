#!/usr/bin/env python3
"""
Updated plotting functions for unified analysis files

These functions can parse the new unified analysis file format with sections.
"""

import csv
import numpy as np

def load_unified_analysis_file(filename):
    """
    Load data from the new unified analysis file format.
    
    Returns:
        tuple: (coordinates_data, analysis_data, metadata)
        - coordinates_data: list of coordinate dictionaries
        - analysis_data: brightness or gaussian fitting data
        - metadata: file metadata dictionary
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
        section = "header"
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
            header_line = None
            for i in range(analysis_section_start, len(lines)):
                line = lines[i].strip()
                if line.startswith('#'):
                    if 'Frame,' in line or 'Frame ' in line:
                        # Extract header from comment
                        header_line = line.replace('#', '').strip()
                    continue
                if not line:
                    continue
                
                if header_line is None:
                    # First data line, use it to determine structure
                    parts = [p.strip() for p in line.split(',')]
                    if metadata["analysis_type"] == "brightness":
                        peak_columns = [f'Peak_{i+1}' for i in range(len(parts)-1)]
                    else:  # gaussian
                        # For gaussian: Amplitude_1, Sigma_x_1, Sigma_y_1, Amplitude_2, etc.
                        peak_columns = []
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
                
                # Parse data line
                parts = [p.strip() for p in line.split(',')]
                analysis_data['frames'].append(int(parts[0]))  # Frame number
                
                # Read analysis values
                for i, peak_col in enumerate(peak_columns):
                    if i + 1 < len(parts):
                        analysis_data['peaks'][peak_col].append(float(parts[i + 1]))
        
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

def convert_to_legacy_format(coords, analysis_data, metadata):
    """
    Convert unified format data to the legacy format expected by existing plotting functions.
    
    Returns:
        tuple: (coords, brightness_data, metadata) in legacy format
    """
    # Coordinates are already in the right format
    
    # Convert analysis data to legacy brightness format
    brightness_data = {
        'frames': analysis_data['frames'],
        'peaks': {}
    }
    
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
    
    return coords, brightness_data, metadata

# Example usage function
def load_analysis_file_for_plotting(filename):
    """
    Load analysis file and return data in format compatible with existing plotting functions.
    
    This function automatically detects the file format and converts to legacy format.
    """
    # Try loading as unified format first
    coords, analysis_data, metadata = load_unified_analysis_file(filename)
    
    if coords is not None and analysis_data is not None:
        # Convert to legacy format for compatibility
        return convert_to_legacy_format(coords, analysis_data, metadata)
    else:
        # Fallback to trying legacy format loaders
        print("Failed to load as unified format, trying legacy format...")
        # You would call your existing load_coordinates_csv and load_brightness_csv here
        return None, None, metadata

if __name__ == "__main__":
    # Test the new loader
    import sys
    if len(sys.argv) > 1:
        coords, brightness_data, metadata = load_analysis_file_for_plotting(sys.argv[1])
        if coords:
            print(f"Successfully loaded {len(coords)} coordinates")
            print(f"First coordinate: {coords[0]}")
        if brightness_data:
            print(f"Brightness data keys: {list(brightness_data['peaks'].keys())}")
            print(f"First few frames: {brightness_data['frames'][:5]}")
