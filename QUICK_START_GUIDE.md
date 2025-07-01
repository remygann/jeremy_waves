# Quick Start Guide - Sinusoidal Baseline Correction

## 🚀 Getting Started in Google Colab

### Step 1: Upload Files to Colab
1. Open Google Colab
2. Upload these files to your Colab environment:
   - `sinusoidal_baseline_correction.py` (main program)
   - Your CSV data files (including `*_normalbaseline.csv`)

### Step 2: Run the Program
```python
# Load and run the program
exec(open('sinusoidal_baseline_correction.py').read())
corrector, results = main()
```

### Step 3: Follow Interactive Prompts
1. **Upload Files**: Select your CSV files when prompted
2. **Set Threshold**: Review suggested threshold and adjust if needed
3. **Choose Visualization**: Decide whether to show individual plots
4. **Download Results**: Get the `corrected_spectra.zip` file

## 📊 Demo Mode (No Data Required)

Want to test the program first? Create synthetic demonstration data:

```python
# Create demo data
exec(open('sinusoidal_baseline_correction.py').read())
demo_with_synthetic_data()

# Then run the main program
corrector, results = main()
```

## 📁 File Requirements

### Input Files
- **Format**: CSV with no headers
- **Columns**: 
  - Column 1: Wavenumbers
  - Column 2: Absorbance values
- **Required**: One reference file ending with `_normalbaseline.csv`
- **Data files**: Any other `.csv` files with noisy baselines

### Example File Structure
```
your_data/
├── sample1_normalbaseline.csv  ← Reference file
├── sample1_measurement1.csv    ← Data file 1
├── sample1_measurement2.csv    ← Data file 2
└── sample1_measurement3.csv    ← Data file 3
```

## 🔧 Advanced Usage

### Custom Workflow
```python
# Initialize
corrector = SpectroscopicBaselineCorrector()

# Upload files
reference_file, data_files = corrector.upload_files()

# Set threshold (interactive)
threshold = corrector.set_threshold(interactive=True)

# Process files
results = corrector.process_all_files(plot_individual=True)

# Export results
exported_files = corrector.export_results()
```

### Non-Interactive Threshold Setting
```python
# Set threshold manually
corrector.set_threshold(threshold=0.01, interactive=False)
```

### Process Single File
```python
# Process one file at a time
result = corrector.correct_spectrum('your_file.csv', plot=True)
```

## 📈 Understanding the Output

### Plots Generated
1. **Baseline Fitting Plot**:
   - Blue line: Original data
   - Red line: Fitted sinusoidal baseline
   - Green dots: Points used for baseline fitting
   - Orange dots: Peak points (excluded from fitting)

2. **Corrected Spectrum Plot**:
   - Green line: Baseline-corrected data
   - Dashed line: Zero reference

### Files Downloaded
- `filename_corrected.csv`: Baseline-corrected data
- `filename_correction_info.txt`: Processing parameters and statistics
- `corrected_spectra.zip`: All results bundled together

## ⚙️ Algorithm Parameters

### Complex Sinusoidal Model
The program fits a sophisticated baseline model:
```
baseline(x) = A₀ + Σᵢ Aᵢ × sin(2πfᵢx + φᵢ) × exp(-decay×x) + trend×x
```

### Key Features
- **Multiple harmonics**: Captures different oscillation frequencies
- **Exponential damping**: Models amplitude decay across wavenumber range
- **Adaptive fitting**: Automatically adjusts to data characteristics
- **Robust optimization**: Uses global optimization for reliable results

## 🛠️ Troubleshooting

### Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "No reference file found" | Ensure one file ends with `_normalbaseline.csv` |
| "Very few baseline points" | Increase threshold value |
| Poor baseline correction | Check reference baseline quality |
| Fitting failures | Program automatically uses polynomial fallback |
| Upload errors | Check CSV format (no headers, 2 columns) |

### Quality Checks
- **Baseline coverage**: Should be 30-80% of data points
- **Peak preservation**: Peaks should remain positive after correction
- **Smooth baseline**: Fitted curve should follow underlying trend

## 🧪 Testing the Program

Run the test suite to verify everything works:

```python
exec(open('test_baseline_correction.py').read())
```

This will:
- Create synthetic test data
- Test all major functions
- Validate results
- Show diagnostic plots

## 📋 Checklist for Success

### Before Processing
- [ ] Reference baseline file is clean and representative
- [ ] All CSV files have proper format (no headers, 2 columns)
- [ ] File naming follows convention (`*_normalbaseline.csv` for reference)
- [ ] Data quality is good (no major artifacts)

### During Processing
- [ ] Review suggested threshold carefully
- [ ] Check baseline fitting plots for quality
- [ ] Verify peak regions are properly identified
- [ ] Monitor processing progress for errors

### After Processing
- [ ] Review corrected spectra visually
- [ ] Check statistics in info files
- [ ] Verify peak preservation
- [ ] Download and save results

## 💡 Tips for Best Results

1. **Reference Baseline**: Use the cleanest, most representative baseline
2. **Threshold Tuning**: Start with suggested value, adjust based on visual inspection
3. **Data Quality**: Remove obvious artifacts before processing
4. **Batch Processing**: Process similar samples together
5. **Validation**: Always review results visually before using

## 📞 Getting Help

### Built-in Help
```python
help(SpectroscopicBaselineCorrector)
help(main)
```

### Example Notebook
Open `example_usage.ipynb` for step-by-step examples

### Documentation
See `README.md` for comprehensive documentation

---

**Ready to start?** Just run:
```python
exec(open('sinusoidal_baseline_correction.py').read())
corrector, results = main()
```

🎉 **Happy baseline correcting!**
