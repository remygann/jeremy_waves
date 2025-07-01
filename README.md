# Sinusoidal Baseline Correction for Spectroscopic Data

A comprehensive Python program for Google Colab that performs advanced sinusoidal baseline correction on spectroscopic data with interactive threshold adjustment and comprehensive visualization.

## Features

- **Automatic Threshold Detection**: Uses reference baseline file to suggest optimal threshold
- **Complex Sinusoidal Fitting**: Multi-harmonic model with damping and frequency modulation
- **Interactive Threshold Adjustment**: User can fine-tune threshold values
- **Comprehensive Visualization**: Overlay plots showing original, fitted baseline, and corrected data
- **Batch Processing**: Handles multiple CSV files automatically with progress indication
- **Export Functionality**: Saves corrected data and parameters with automatic download
- **Error Handling**: Robust fitting with polynomial fallback for difficult cases

## Input Data Requirements

### File Format
- **CSV files with no headers**
- **Column 1**: Wavenumbers
- **Column 2**: Absorbance values

### Required Files
1. **Reference file**: `*_normalbaseline.csv` - Clean baseline for threshold detection
2. **Data files**: `*.csv` - Spectroscopic data with noisy sinusoidal baselines

## Algorithm Overview

### 1. Threshold Detection
- Analyzes reference baseline file
- Fits initial sinusoidal model
- Calculates threshold as mean + 2×std of residuals
- Allows user adjustment

### 2. Peak Identification
- Fits rough baseline to identify peak regions
- Removes data points above threshold from fitted baseline
- Preserves baseline points for accurate fitting

### 3. Baseline Fitting
Uses complex sinusoidal model:
```
baseline(x) = A₀ + Σᵢ Aᵢ × sin(2πfᵢx + φᵢ) × exp(-decay×x) + trend×x
```

Where:
- **A₀**: DC offset
- **Aᵢ**: Amplitudes for multiple harmonics
- **fᵢ**: Frequencies for different oscillation components
- **φᵢ**: Phase shifts
- **decay**: Exponential damping factor
- **trend**: Linear trend component

### 4. Baseline Correction
- Calculates baseline values at all points using fitted curve
- Subtracts fitted baseline from original absorbance
- Preserves peak information while removing baseline distortion

## Usage Instructions

### Quick Start in Google Colab

1. **Upload the program file**:
   ```python
   # Upload sinusoidal_baseline_correction.py to Colab
   ```

2. **Run the program**:
   ```python
   exec(open('sinusoidal_baseline_correction.py').read())
   corrector, results = main()
   ```

3. **Follow the interactive prompts**:
   - Upload your CSV files (including reference file)
   - Review suggested threshold
   - Adjust threshold if needed
   - Choose visualization options
   - Download results

### Alternative: Step-by-Step Usage

```python
# Initialize corrector
corrector = SpectroscopicBaselineCorrector()

# Upload files
reference_file, data_files = corrector.upload_files()

# Set threshold (interactive)
threshold = corrector.set_threshold(interactive=True)

# Process all files
results = corrector.process_all_files(plot_individual=True)

# Export results
exported_files = corrector.export_results()
```

### Demo with Synthetic Data

```python
# Create demonstration data
demo_with_synthetic_data()

# Then run main program
corrector, results = main()
```

## Output Files

### Corrected Data
- **Filename**: `original_name_corrected.csv`
- **Format**: Same as input (wavenumber, corrected_absorbance)
- **Content**: Baseline-corrected spectroscopic data

### Parameter Reports
- **Filename**: `original_name_correction_info.txt`
- **Content**:
  - Processing parameters
  - Fitting statistics
  - Quality metrics
  - Timestamp and file information

### Download Package
- **Filename**: `corrected_spectra.zip`
- **Content**: All corrected CSV files and parameter reports
- **Auto-download**: Initiated automatically in Colab

## Visualization

Each processed file generates two plots:

1. **Baseline Fitting Plot**:
   - Original absorbance data
   - Fitted sinusoidal baseline
   - Baseline points (green) vs peak points (orange)

2. **Corrected Spectrum Plot**:
   - Baseline-corrected absorbance data
   - Zero reference line

## Technical Details

### Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
import os
import zipfile
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Optional
from google.colab import files
import io
```

### Fitting Algorithm
1. **Initial rough fit** using curve_fit with bounded parameters
2. **Peak identification** using threshold-based masking
3. **Global optimization** using differential_evolution for robustness
4. **Refinement** using curve_fit on baseline points only
5. **Fallback** to polynomial fitting if sinusoidal fitting fails

### Error Handling
- Validates file formats and required files
- Handles fitting failures with polynomial fallback
- Provides informative error messages
- Continues processing other files if one fails

## Troubleshooting

### Common Issues

1. **"No reference file found"**
   - Ensure one file ends with `_normalbaseline.csv`
   - Check file naming convention

2. **"Very few baseline points detected"**
   - Threshold may be too low
   - Try increasing threshold value
   - Check if reference baseline is appropriate

3. **Fitting failures**
   - Program automatically falls back to polynomial fitting
   - Check data quality and format
   - Ensure sufficient baseline points

4. **Poor baseline correction**
   - Adjust threshold interactively
   - Check reference baseline quality
   - Verify data format (wavenumber, absorbance)

### Performance Tips
- Use reasonable file sizes (< 10MB per file)
- Ensure clean reference baseline
- Start with suggested threshold and adjust as needed
- Review individual plots before batch processing

## Example Workflow

```python
# 1. Create demo data (optional)
demo_with_synthetic_data()

# 2. Run main program
corrector, results = main()

# 3. Upload files when prompted
# 4. Review threshold suggestion
# 5. Adjust if needed
# 6. Process files
# 7. Download results

# 8. Access results programmatically
for filename, result in results.items():
    if result:
        print(f"{filename}: {len(result['original_x'])} points processed")
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure proper file format and naming
4. Try with synthetic demo data first

---

**Author**: AI Assistant  
**Date**: 2025-06-30  
**Version**: 1.0  
**Platform**: Google Colab
