# Local Baseline Correction - Usage Guide

## 🚀 Quick Start

### Step 1: Prepare Your Data
Place your CSV files in a folder with:
- **One reference file**: `*_normalbaseline.csv` (clean baseline)
- **Data files**: `*.csv` (files needing baseline correction)

### Step 2: Run the Program
```bash
# Navigate to your data folder
cd /path/to/your/csv/files

# Run the baseline correction
python3 /path/to/local_baseline_correction.py
```

Or use the Python interface:
```python
from local_baseline_correction import main
corrector, results = main()
```

## 📁 File Requirements

### Input Files
- **CSV format**: No headers, two columns
- **Column 1**: Wavenumbers
- **Column 2**: Absorbance values
- **Reference file**: Must contain `_normalbaseline` in filename

### Example Directory Structure
```
your_data_folder/
├── sample1_normalbaseline.csv  ← Reference (clean baseline)
├── sample1_measurement1.csv    ← Data file 1
├── sample1_measurement2.csv    ← Data file 2
└── sample1_measurement3.csv    ← Data file 3
```

## 🎯 Usage Options

### 1. Interactive Mode (Recommended)
```python
from local_baseline_correction import main
corrector, results = main()
```
- Allows threshold adjustment
- Shows individual plots
- Full user control

### 2. Quick Processing (Automated)
```python
from local_baseline_correction import quick_process
corrector, results = quick_process()
```
- No user interaction required
- Automatic threshold detection
- Saves plots but doesn't display them

### 3. Demo Mode (Testing)
```python
from local_baseline_correction import main_with_demo
corrector, results = main_with_demo()
```
- Creates synthetic test data
- Processes the demo data
- Perfect for testing the program

### 4. Command Line Interface
```bash
# Basic usage
python3 run_baseline_correction.py

# Create and process demo data
python3 run_baseline_correction.py --demo

# Quick processing without interaction
python3 run_baseline_correction.py --quick

# Set specific threshold
python3 run_baseline_correction.py --threshold 0.01

# Process specific directory
python3 run_baseline_correction.py --directory /path/to/data

# No individual plots
python3 run_baseline_correction.py --no-plots
```

## 📊 Output Files

The program creates several output files in the same directory:

### Corrected Data Files
- **`filename_corrected.csv`**: Baseline-corrected spectroscopic data
- Same format as input (wavenumber, corrected_absorbance)

### Information Files
- **`filename_correction_info.txt`**: Detailed processing report for each file
- **`baseline_correction_summary.txt`**: Overall summary of all processed files

### Plot Files (if enabled)
- **`filename_correction_plot.png`**: Visualization of correction for each file
- **`threshold_analysis_*.png`**: Threshold detection analysis plot

## 📈 Understanding Results

### Quality Metrics
- **Baseline Coverage**: 70-90% is typical (percentage of points used for baseline fitting)
- **Peak Preservation**: Should maintain positive peaks after correction
- **Correction Magnitude**: Indicates how much baseline distortion was removed

### Visual Inspection
- **Top plot**: Original data (blue) with fitted baseline (red)
- **Green dots**: Points used for baseline fitting
- **Orange dots**: Peak points (excluded from fitting)
- **Bottom plot**: Corrected spectrum (green)

## ⚙️ Advanced Usage

### Custom Directory Processing
```python
from local_baseline_correction import process_local_directory

# Process specific directory with custom settings
corrector, results = process_local_directory(
    directory="/path/to/data",
    interactive=False,           # No user interaction
    plot_individual=True,        # Show plots
    save_plots=True             # Save plot files
)
```

### Manual Threshold Setting
```python
from local_baseline_correction import LocalSpectroscopicBaselineCorrector

corrector = LocalSpectroscopicBaselineCorrector()
corrector.scan_directory()
corrector.set_threshold(0.015, interactive=False)  # Set specific threshold
results = corrector.process_all_files()
corrector.export_results()
```

### Processing Single Files
```python
corrector = LocalSpectroscopicBaselineCorrector()
corrector.scan_directory()
corrector.set_threshold()

# Process one file at a time
result = corrector.correct_spectrum('your_file.csv', plot=True)
```

## 🛠️ Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| No CSV files found | Check directory path and file extensions |
| No reference file | Ensure one file contains `_normalbaseline` |
| Poor baseline fitting | Adjust threshold interactively |
| Import errors | Install dependencies: `pip install numpy pandas matplotlib scipy` |
| Permission errors | Check write permissions in target directory |

### Quality Checks
1. **Baseline coverage** should be 70-90%
2. **Peaks should remain positive** after correction
3. **Fitted baseline** should follow the underlying trend
4. **Residuals** should be randomly distributed around zero

## 📋 Complete Example

```python
# Complete workflow example
from local_baseline_correction import *

# Option 1: Create demo data and test
create_demo_data()
corrector, results = main()

# Option 2: Process your own data
corrector, results = process_local_directory("/path/to/your/data")

# Option 3: Quick automated processing
corrector, results = quick_process("/path/to/your/data")

# Access results
for filename, result in results.items():
    if result:
        print(f"{filename}: Successfully processed")
        print(f"  Original range: {result['original_y'].min():.3f} to {result['original_y'].max():.3f}")
        print(f"  Corrected range: {result['corrected_y'].min():.3f} to {result['corrected_y'].max():.3f}")
```

## 🎉 Success Indicators

✅ **Processing completed successfully**
✅ **All files have corrected versions**
✅ **Baseline coverage is 70-90%**
✅ **Peaks are preserved (positive values)**
✅ **Fitted baselines follow data trends**
✅ **Summary report shows good statistics**

---

**Ready to process your data?**

1. Put your CSV files in a folder
2. Ensure you have a `*_normalbaseline.csv` reference file
3. Run: `python3 local_baseline_correction.py`
4. Follow the prompts
5. Check the results!

🔬 **Happy spectroscopy!**
