"""
Test script for the sinusoidal baseline correction program.
This script creates synthetic data and tests the baseline correction functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from sinusoidal_baseline_correction import SpectroscopicBaselineCorrector

def create_test_data():
    """Create synthetic test data for validation."""
    print("Creating synthetic test data...")
    
    # Generate wavenumbers
    wavenumbers = np.linspace(500, 3500, 800)
    
    # Create clean reference baseline
    np.random.seed(42)
    ref_baseline = (0.05 + 
                   0.03 * np.sin(2 * np.pi * wavenumbers / 1200) * np.exp(-(wavenumbers - 500) / 2000) +
                   0.01 * np.sin(2 * np.pi * wavenumbers / 400) +
                   np.random.normal(0, 0.001, len(wavenumbers)))
    
    # Create test sample with peaks and baseline distortion
    test_sample = ref_baseline.copy()
    
    # Add peaks
    peak_positions = [800, 1400, 2000, 2800]
    peak_heights = [0.15, 0.12, 0.18, 0.10]
    peak_widths = [40, 35, 45, 30]
    
    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        peak_mask = np.abs(wavenumbers - pos) < width
        test_sample[peak_mask] += height * np.exp(-((wavenumbers[peak_mask] - pos) / (width/3))**2)
    
    # Add sinusoidal baseline distortion
    test_sample += (0.06 * np.sin(2 * np.pi * wavenumbers / 700) * np.exp(-(wavenumbers - 500) / 1800) +
                   0.03 * np.sin(2 * np.pi * wavenumbers / 300) +
                   0.02 * np.sin(2 * np.pi * wavenumbers / 150))
    
    return wavenumbers, ref_baseline, test_sample

def test_baseline_correction():
    """Test the baseline correction functionality."""
    print("Testing baseline correction functionality...")
    
    # Create test data
    wavenumbers, ref_baseline, test_sample = create_test_data()
    
    # Save test files
    ref_df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': ref_baseline})
    test_df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': test_sample})
    
    ref_filename = 'test_normalbaseline.csv'
    test_filename = 'test_sample.csv'
    
    ref_df.to_csv(ref_filename, index=False, header=False)
    test_df.to_csv(test_filename, index=False, header=False)
    
    try:
        # Initialize corrector
        corrector = SpectroscopicBaselineCorrector()
        
        # Load data manually (simulating file upload)
        corrector.reference_data = ref_df
        corrector.data_files = {test_filename: test_df}
        
        print(f"✓ Test data loaded: {len(wavenumbers)} points")
        
        # Test threshold detection
        threshold = corrector.detect_threshold(plot=False)
        corrector.threshold = threshold
        print(f"✓ Threshold detected: {threshold:.6f}")
        
        # Test baseline correction
        result = corrector.correct_spectrum(test_filename, plot=False)
        print(f"✓ Baseline correction completed")
        
        # Validate results
        original_y = result['original_y']
        corrected_y = result['corrected_y']
        fitted_baseline = result['fitted_baseline']
        baseline_mask = result['baseline_mask']
        
        # Check that correction was applied
        assert not np.array_equal(original_y, corrected_y), "Correction was not applied!"
        
        # Check that baseline points were identified
        baseline_coverage = np.sum(baseline_mask) / len(baseline_mask)
        assert 0.3 < baseline_coverage < 0.9, f"Baseline coverage {baseline_coverage:.2f} seems unreasonable"
        
        # Check that peaks are preserved (should be positive after correction)
        peak_regions = []
        for pos in [800, 1400, 2000, 2800]:
            peak_idx = np.argmin(np.abs(wavenumbers - pos))
            peak_regions.append(peak_idx)
        
        peak_heights_corrected = [corrected_y[idx] for idx in peak_regions]
        assert all(h > 0.05 for h in peak_heights_corrected), "Peaks not properly preserved!"
        
        print(f"✓ Validation passed:")
        print(f"  - Baseline coverage: {baseline_coverage:.1%}")
        print(f"  - Peak preservation: {np.mean(peak_heights_corrected):.4f} avg height")
        print(f"  - Correction range: {corrected_y.min():.4f} to {corrected_y.max():.4f}")
        
        # Create validation plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original data with baseline
        plt.subplot(2, 2, 1)
        plt.plot(wavenumbers, original_y, 'b-', label='Original Data', alpha=0.7)
        plt.plot(wavenumbers, fitted_baseline, 'r-', label='Fitted Baseline', linewidth=2)
        plt.scatter(wavenumbers[baseline_mask], original_y[baseline_mask], 
                   c='green', s=5, alpha=0.6, label='Baseline Points')
        plt.xlabel('Wavenumber')
        plt.ylabel('Absorbance')
        plt.title('Original Data with Fitted Baseline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Corrected data
        plt.subplot(2, 2, 2)
        plt.plot(wavenumbers, corrected_y, 'g-', label='Corrected Data', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Wavenumber')
        plt.ylabel('Corrected Absorbance')
        plt.title('Baseline-Corrected Spectrum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Comparison
        plt.subplot(2, 2, 3)
        plt.plot(wavenumbers, original_y, 'b-', alpha=0.7, label='Original')
        plt.plot(wavenumbers, corrected_y, 'g-', alpha=0.7, label='Corrected')
        plt.plot(wavenumbers, ref_baseline, 'k--', alpha=0.5, label='True Reference')
        plt.xlabel('Wavenumber')
        plt.ylabel('Absorbance')
        plt.title('Comparison: Original vs Corrected')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Residuals
        plt.subplot(2, 2, 4)
        residuals = original_y - fitted_baseline
        plt.plot(wavenumbers, residuals, 'purple', alpha=0.7, label='Residuals')
        plt.axhline(y=threshold, color='orange', linestyle=':', label=f'Threshold: {threshold:.4f}')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Wavenumber')
        plt.ylabel('Residual')
        plt.title('Fitting Residuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        # Clean up test files
        for filename in [ref_filename, test_filename]:
            if os.path.exists(filename):
                os.remove(filename)

def test_complex_sinusoidal_function():
    """Test the complex sinusoidal baseline function."""
    print("Testing complex sinusoidal function...")
    
    corrector = SpectroscopicBaselineCorrector()
    
    # Test data
    x = np.linspace(0, 1, 100)
    
    # Test parameters
    params = [0.1, 0.05, 0.02, 0.01, 2.0, 5.0, 10.0, 0, np.pi/4, np.pi/2, 1.0, 0.01]
    
    try:
        y = corrector.complex_sinusoidal_baseline(x, *params)
        
        assert len(y) == len(x), "Output length mismatch!"
        assert np.all(np.isfinite(y)), "Non-finite values in output!"
        
        print(f"✓ Function test passed:")
        print(f"  - Input length: {len(x)}")
        print(f"  - Output range: {y.min():.4f} to {y.max():.4f}")
        print(f"  - All finite: {np.all(np.isfinite(y))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Function test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("BASELINE CORRECTION TEST SUITE")
    print("="*60)
    
    # Test 1: Complex sinusoidal function
    test1_passed = test_complex_sinusoidal_function()
    
    print("\n" + "-"*60)
    
    # Test 2: Full baseline correction
    test2_passed = test_baseline_correction()
    
    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The baseline correction program is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("="*60)
