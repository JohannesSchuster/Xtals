# File Change Detection and Cache Clearing

## Overview
Added functionality to automatically detect when a new file is loaded and perform cleanup operations to prevent confusion with data from previous files.

## Implementation Details

### Changes Made

1. **Added file path tracking** (`_current_file_path` attribute)
   - Tracks the currently loaded file path
   - Initialized to `None` in the constructor

2. **Enhanced `load_file()` method**
   - Compares new file path with currently loaded file
   - Performs cleanup operations when a different file is detected

3. **Fixed event bindings**
   - Updated entry field bindings to use lambda functions for proper event handling

### Cleanup Operations (performed when loading a new file)

1. **Peak Finder Cache Clearing**
   - Clears all found peak coordinates: `self.peak_finder.cache.coordinates = torch.empty((0, 2))`
   - Clears all masks: `self.peak_finder.clear_masks()`

2. **Display Window Management**
   - Closes existing image display window
   - Creates new ImageDisplay instance with proper callbacks
   - Resets display button state to "Show"
   - Same process for histogram display

3. **Cache Clearing**
   - Clears FFT image cache: `self.fft_image_cache = None`

4. **User Feedback**
   - Prints informative message indicating file loading
   - Shows count of cleared peaks if any existed

### Triggers
The cleanup is triggered when:
- User browses for a new file using the "Browse" button
- User manually types a different file path and presses Enter
- User manually types a different file path and clicks elsewhere (focus out)
- Application starts with a command line argument (different from any previously loaded file)

### Benefits
- Prevents confusion between data from different files
- Ensures clean state when analyzing new images
- Provides clear visual feedback that previous results are cleared
- Maintains performance by clearing unused caches
- Prevents accidentally running analysis on old peak data with new images

## Usage
No changes to user workflow - the cleanup happens automatically whenever a different file is loaded. Users will see console output indicating when the cache has been cleared.

Example output:
```
Loading new file: X2025-06-19_22.14.45_Xlzso-NO3-3_r1_82_000.tif - Cleared 25 peaks from cache
```

## Technical Notes
- File comparison is done using absolute paths to handle relative path variations
- All cleanup operations are performed before the new file loading begins
- Display windows are properly closed and recreated to prevent memory leaks
- Event callbacks are properly reestablished for new display instances
