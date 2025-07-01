"""
Sinusoidal Baseline Correction for Spectroscopic Data
=====================================================

A comprehensive Python program for Google Colab that performs sinusoidal baseline 
correction on spectroscopic data with interactive threshold adjustment and 
comprehensive visualization.

Author: AI Assistant
Date: 2025-06-30
"""

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

warnings.filterwarnings('ignore')

class SpectroscopicBaselineCorrector:
    """
    A comprehensive class for performing sinusoidal baseline correction on spectroscopic data.
    """
    
    def __init__(self):
        self.data_files = {}
        self.reference_data = None
        self.threshold = None
        self.fitted_parameters = {}
        self.corrected_data = {}
        
    def upload_files(self):
        """Upload CSV files using Google Colab file upload interface."""
        print("Please upload your CSV files (including the reference *_normalbaseline.csv file)")
        uploaded = files.upload()
        
        reference_file = None
        data_files = []
        
        for filename, content in uploaded.items():
            if filename.endswith('_normalbaseline.csv'):
                reference_file = filename
                self.reference_data = pd.read_csv(io.BytesIO(content), header=None, names=['wavenumber', 'absorbance'])
                print(f"✓ Reference file loaded: {filename}")
            elif filename.endswith('.csv'):
                data_files.append(filename)
                self.data_files[filename] = pd.read_csv(io.BytesIO(content), header=None, names=['wavenumber', 'absorbance'])
                print(f"✓ Data file loaded: {filename}")
        
        if reference_file is None:
            raise ValueError("No reference file (*_normalbaseline.csv) found!")
        
        if not data_files:
            raise ValueError("No data files found!")
            
        print(f"\nLoaded {len(data_files)} data files and 1 reference file")
        return reference_file, data_files
    
    def complex_sinusoidal_baseline(self, x: np.ndarray, *params) -> np.ndarray:
        """
        Complex sinusoidal baseline model with multiple harmonics and damping.
        
        Parameters:
        - A0: DC offset
        - A1, A2, A3: Amplitudes for harmonics
        - f1, f2, f3: Frequencies for harmonics
        - phi1, phi2, phi3: Phase shifts
        - decay: Exponential decay factor
        - trend: Linear trend
        """
        A0, A1, A2, A3, f1, f2, f3, phi1, phi2, phi3, decay, trend = params
        
        # Normalize x for better fitting
        x_norm = (x - x.min()) / (x.max() - x.min())
        
        # Exponential decay factor
        decay_factor = np.exp(-decay * x_norm)
        
        # Multiple harmonics with varying characteristics
        baseline = (A0 + 
                   A1 * np.sin(2 * np.pi * f1 * x_norm + phi1) * decay_factor +
                   A2 * np.sin(2 * np.pi * f2 * x_norm + phi2) * decay_factor**0.5 +
                   A3 * np.sin(2 * np.pi * f3 * x_norm + phi3) * decay_factor**0.25 +
                   trend * x_norm)
        
        return baseline
    
    def detect_threshold(self, plot: bool = True) -> float:
        """
        Automatically detect threshold using the reference baseline file.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded!")
        
        x_ref = self.reference_data['wavenumber'].values
        y_ref = self.reference_data['absorbance'].values
        
        # Fit a simple baseline to reference data
        try:
            # Initial parameter guess for reference baseline
            p0 = [np.mean(y_ref), 0.01, 0.005, 0.002, 1.0, 2.0, 5.0, 0, 0, 0, 0.1, 0]
            
            # Bounds for parameters
            bounds = ([-np.inf, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -2*np.pi, -2*np.pi, -2*np.pi, 0, -0.01],
                     [np.inf, 0.1, 0.1, 0.1, 10, 20, 50, 2*np.pi, 2*np.pi, 2*np.pi, 5, 0.01])
            
            popt, _ = curve_fit(self.complex_sinusoidal_baseline, x_ref, y_ref, 
                              p0=p0, bounds=bounds, maxfev=5000)
            
            fitted_baseline = self.complex_sinusoidal_baseline(x_ref, *popt)
            
            # Calculate threshold as mean + 2*std of residuals
            residuals = y_ref - fitted_baseline
            suggested_threshold = np.mean(residuals) + 2 * np.std(residuals)
            
        except Exception as e:
            print(f"Warning: Could not fit reference baseline: {e}")
            # Fallback: use statistical approach
            suggested_threshold = np.std(y_ref) * 0.5
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(x_ref, y_ref, 'b-', label='Reference Data', alpha=0.7)
            if 'fitted_baseline' in locals():
                plt.plot(x_ref, fitted_baseline, 'r--', label='Fitted Baseline', linewidth=2)
                plt.axhline(y=np.max(fitted_baseline) + suggested_threshold, 
                           color='orange', linestyle=':', label=f'Suggested Threshold: {suggested_threshold:.4f}')
            plt.xlabel('Wavenumber')
            plt.ylabel('Absorbance')
            plt.title('Reference Baseline Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return suggested_threshold
    
    def set_threshold(self, threshold: Optional[float] = None, interactive: bool = True) -> float:
        """
        Set the threshold for peak detection, with optional interactive adjustment.
        """
        if threshold is None:
            threshold = self.detect_threshold()
        
        if interactive:
            print(f"Suggested threshold: {threshold:.4f}")
            user_input = input("Enter new threshold value (or press Enter to use suggested): ").strip()
            if user_input:
                try:
                    threshold = float(user_input)
                    print(f"Using threshold: {threshold:.4f}")
                except ValueError:
                    print("Invalid input. Using suggested threshold.")
        
        self.threshold = threshold
        return threshold

    def fit_baseline(self, x: np.ndarray, y: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit sinusoidal baseline to data, excluding peaks above threshold.
        """
        # First, fit a rough baseline to identify peaks
        try:
            p0_rough = [np.mean(y), 0.01, 0.005, 0.002, 1.0, 2.0, 5.0, 0, 0, 0, 0.1, 0]
            bounds = ([-np.inf, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -2*np.pi, -2*np.pi, -2*np.pi, 0, -0.01],
                     [np.inf, 0.1, 0.1, 0.1, 10, 20, 50, 2*np.pi, 2*np.pi, 2*np.pi, 5, 0.01])

            popt_rough, _ = curve_fit(self.complex_sinusoidal_baseline, x, y,
                                    p0=p0_rough, bounds=bounds, maxfev=3000)
            rough_baseline = self.complex_sinusoidal_baseline(x, *popt_rough)
        except:
            # Fallback to simple polynomial if sinusoidal fit fails
            rough_baseline = np.polyval(np.polyfit(x, y, 3), x)

        # Identify baseline points (below threshold from rough baseline)
        baseline_mask = (y - rough_baseline) <= threshold

        if np.sum(baseline_mask) < 10:
            print("Warning: Very few baseline points detected. Adjusting threshold...")
            threshold *= 2
            baseline_mask = (y - rough_baseline) <= threshold

        x_baseline = x[baseline_mask]
        y_baseline = y[baseline_mask]

        print(f"Using {np.sum(baseline_mask)} baseline points out of {len(x)} total points")

        # Fit final baseline using only baseline points
        try:
            # Use differential evolution for robust global optimization
            def objective(params):
                try:
                    fitted = self.complex_sinusoidal_baseline(x_baseline, *params)
                    return np.sum((y_baseline - fitted)**2)
                except:
                    return 1e10

            # Parameter bounds for differential evolution
            bounds_de = [(-1, 1),      # A0
                        (-0.1, 0.1),   # A1
                        (-0.1, 0.1),   # A2
                        (-0.1, 0.1),   # A3
                        (0.1, 10),     # f1
                        (0.1, 20),     # f2
                        (0.1, 50),     # f3
                        (-2*np.pi, 2*np.pi),  # phi1
                        (-2*np.pi, 2*np.pi),  # phi2
                        (-2*np.pi, 2*np.pi),  # phi3
                        (0, 5),        # decay
                        (-0.01, 0.01)] # trend

            result = differential_evolution(objective, bounds_de, seed=42, maxiter=1000)
            popt = result.x

            # Refine with curve_fit
            popt, pcov = curve_fit(self.complex_sinusoidal_baseline, x_baseline, y_baseline,
                                 p0=popt, maxfev=5000)

        except Exception as e:
            print(f"Warning: Advanced fitting failed ({e}). Using simple polynomial fallback.")
            # Polynomial fallback
            poly_coeffs = np.polyfit(x_baseline, y_baseline, 5)
            fitted_baseline = np.polyval(poly_coeffs, x)
            return fitted_baseline, baseline_mask

        # Calculate fitted baseline for all x values
        fitted_baseline = self.complex_sinusoidal_baseline(x, *popt)

        return fitted_baseline, baseline_mask

    def correct_spectrum(self, filename: str, plot: bool = True) -> Dict:
        """
        Perform baseline correction on a single spectrum.
        """
        if filename not in self.data_files:
            raise ValueError(f"File {filename} not found in loaded data!")

        data = self.data_files[filename]
        x = data['wavenumber'].values
        y = data['absorbance'].values

        print(f"\nProcessing: {filename}")

        # Fit baseline
        fitted_baseline, baseline_mask = self.fit_baseline(x, y, self.threshold)

        # Correct spectrum
        corrected_y = y - fitted_baseline

        # Store results
        result = {
            'original_x': x,
            'original_y': y,
            'fitted_baseline': fitted_baseline,
            'corrected_y': corrected_y,
            'baseline_mask': baseline_mask,
            'filename': filename
        }

        self.corrected_data[filename] = result

        if plot:
            self.plot_correction(result)

        return result

    def plot_correction(self, result: Dict):
        """
        Create comprehensive visualization of the baseline correction.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x = result['original_x']
        y_orig = result['original_y']
        y_baseline = result['fitted_baseline']
        y_corrected = result['corrected_y']
        baseline_mask = result['baseline_mask']

        # Top plot: Original data with fitted baseline
        ax1.plot(x, y_orig, 'b-', label='Original Data', alpha=0.7, linewidth=1)
        ax1.plot(x, y_baseline, 'r-', label='Fitted Baseline', linewidth=2)
        ax1.scatter(x[baseline_mask], y_orig[baseline_mask], c='green', s=10,
                   alpha=0.6, label='Baseline Points', zorder=5)
        ax1.scatter(x[~baseline_mask], y_orig[~baseline_mask], c='orange', s=10,
                   alpha=0.6, label='Peak Points', zorder=5)

        ax1.set_xlabel('Wavenumber')
        ax1.set_ylabel('Absorbance')
        ax1.set_title(f'Baseline Fitting: {result["filename"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Corrected data
        ax2.plot(x, y_corrected, 'g-', label='Corrected Data', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Wavenumber')
        ax2.set_ylabel('Corrected Absorbance')
        ax2.set_title('Baseline-Corrected Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"Baseline correction statistics for {result['filename']}:")
        print(f"  - Original range: {y_orig.min():.4f} to {y_orig.max():.4f}")
        print(f"  - Corrected range: {y_corrected.min():.4f} to {y_corrected.max():.4f}")
        print(f"  - Baseline points used: {np.sum(baseline_mask)}/{len(x)} ({100*np.sum(baseline_mask)/len(x):.1f}%)")
        print(f"  - RMS baseline deviation: {np.sqrt(np.mean((y_orig[baseline_mask] - y_baseline[baseline_mask])**2)):.4f}")

    def process_all_files(self, plot_individual: bool = True) -> Dict:
        """
        Process all loaded data files with progress indication.
        """
        if self.threshold is None:
            raise ValueError("Threshold not set! Call set_threshold() first.")

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING - {len(self.data_files)} files")
        print(f"Threshold: {self.threshold:.4f}")
        print(f"{'='*60}")

        results = {}

        for i, filename in enumerate(self.data_files.keys(), 1):
            print(f"\n[{i}/{len(self.data_files)}] Processing {filename}...")

            try:
                result = self.correct_spectrum(filename, plot=plot_individual)
                results[filename] = result
                print(f"✓ Successfully processed {filename}")

            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                results[filename] = None

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        successful = sum(1 for r in results.values() if r is not None)
        print(f"Successfully processed: {successful}/{len(self.data_files)} files")
        print(f"{'='*60}")

        return results

    def export_results(self, output_dir: str = "corrected_spectra"):
        """
        Export corrected data and parameters to files.
        """
        if not self.corrected_data:
            raise ValueError("No corrected data to export! Process files first.")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nExporting results to '{output_dir}' directory...")

        exported_files = []

        for filename, result in self.corrected_data.items():
            if result is None:
                continue

            # Export corrected CSV
            base_name = filename.replace('.csv', '')
            corrected_filename = f"{base_name}_corrected.csv"
            corrected_path = os.path.join(output_dir, corrected_filename)

            corrected_df = pd.DataFrame({
                'wavenumber': result['original_x'],
                'absorbance': result['corrected_y']
            })
            corrected_df.to_csv(corrected_path, index=False, header=False)
            exported_files.append(corrected_path)

            # Export parameters and statistics
            params_filename = f"{base_name}_correction_info.txt"
            params_path = os.path.join(output_dir, params_filename)

            with open(params_path, 'w') as f:
                f.write(f"Baseline Correction Report\n")
                f.write(f"{'='*40}\n")
                f.write(f"Original file: {filename}\n")
                f.write(f"Corrected file: {corrected_filename}\n")
                f.write(f"Processing date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write(f"Parameters:\n")
                f.write(f"  Threshold: {self.threshold:.6f}\n")
                f.write(f"  Baseline points used: {np.sum(result['baseline_mask'])}/{len(result['original_x'])}\n")
                f.write(f"  Baseline coverage: {100*np.sum(result['baseline_mask'])/len(result['original_x']):.1f}%\n\n")

                f.write(f"Statistics:\n")
                f.write(f"  Original range: {result['original_y'].min():.6f} to {result['original_y'].max():.6f}\n")
                f.write(f"  Corrected range: {result['corrected_y'].min():.6f} to {result['corrected_y'].max():.6f}\n")
                f.write(f"  RMS baseline deviation: {np.sqrt(np.mean((result['original_y'][result['baseline_mask']] - result['fitted_baseline'][result['baseline_mask']])**2)):.6f}\n")

            exported_files.append(params_path)

        print(f"✓ Exported {len(exported_files)} files")

        # Create zip file for download
        zip_filename = f"{output_dir}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file_path in exported_files:
                zipf.write(file_path, os.path.basename(file_path))

        print(f"✓ Created zip file: {zip_filename}")

        # Download zip file in Colab
        try:
            files.download(zip_filename)
            print(f"✓ Download initiated for {zip_filename}")
        except:
            print(f"Note: Could not auto-download. Please manually download {zip_filename}")

        return exported_files

def main():
    """
    Main execution function with complete workflow.
    """
    print("="*80)
    print("SINUSOIDAL BASELINE CORRECTION FOR SPECTROSCOPIC DATA")
    print("="*80)
    print("\nThis program performs advanced sinusoidal baseline correction on spectroscopic data.")
    print("Features:")
    print("• Automatic threshold detection using reference baseline")
    print("• Complex sinusoidal fitting with multiple harmonics and damping")
    print("• Interactive threshold adjustment")
    print("• Comprehensive visualization")
    print("• Batch processing with progress indication")
    print("• Export corrected data and parameters")
    print("\nRequired file format:")
    print("• CSV files with no headers")
    print("• Column 1: Wavenumbers, Column 2: Absorbance")
    print("• One reference file: *_normalbaseline.csv")
    print("• Other files: data with noisy sinusoidal baselines")

    try:
        # Initialize corrector
        corrector = SpectroscopicBaselineCorrector()

        # Upload and load files
        print(f"\n{'-'*60}")
        print("STEP 1: FILE UPLOAD")
        print(f"{'-'*60}")
        reference_file, data_files = corrector.upload_files()

        # Detect and set threshold
        print(f"\n{'-'*60}")
        print("STEP 2: THRESHOLD DETECTION")
        print(f"{'-'*60}")
        threshold = corrector.set_threshold(interactive=True)

        # Process all files
        print(f"\n{'-'*60}")
        print("STEP 3: BASELINE CORRECTION")
        print(f"{'-'*60}")

        # Ask user about individual plotting
        plot_choice = input("Show individual plots for each file? (y/n, default=y): ").strip().lower()
        plot_individual = plot_choice != 'n'

        results = corrector.process_all_files(plot_individual=plot_individual)

        # Export results
        print(f"\n{'-'*60}")
        print("STEP 4: EXPORT RESULTS")
        print(f"{'-'*60}")
        exported_files = corrector.export_results()

        # Summary
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*80}")
        successful = sum(1 for r in results.values() if r is not None)
        print(f"✓ Successfully processed: {successful}/{len(data_files)} files")
        print(f"✓ Threshold used: {threshold:.6f}")
        print(f"✓ Exported: {len(exported_files)} files")
        print(f"✓ Results saved to: corrected_spectra.zip")

        return corrector, results

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your files and try again.")
        raise

def demo_with_synthetic_data():
    """
    Create synthetic data for demonstration purposes.
    """
    print("Creating synthetic demonstration data...")

    # Generate synthetic wavenumbers
    wavenumbers = np.linspace(400, 4000, 1000)

    # Create reference baseline (clean)
    np.random.seed(42)
    ref_baseline = (0.1 + 0.05 * np.sin(2 * np.pi * wavenumbers / 1000) *
                   np.exp(-(wavenumbers - 400) / 2000) +
                   0.02 * np.sin(2 * np.pi * wavenumbers / 500) +
                   np.random.normal(0, 0.002, len(wavenumbers)))

    # Create noisy data with peaks
    noisy_data = ref_baseline.copy()

    # Add some peaks
    peak_positions = [800, 1200, 1600, 2400, 3200]
    for pos in peak_positions:
        peak_width = 50
        peak_height = np.random.uniform(0.1, 0.3)
        peak_mask = np.abs(wavenumbers - pos) < peak_width
        noisy_data[peak_mask] += peak_height * np.exp(-((wavenumbers[peak_mask] - pos) / (peak_width/3))**2)

    # Add more complex baseline distortion
    noisy_data += (0.08 * np.sin(2 * np.pi * wavenumbers / 800) *
                  np.exp(-(wavenumbers - 400) / 1500) +
                  0.04 * np.sin(2 * np.pi * wavenumbers / 300))

    # Save synthetic files
    ref_df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': ref_baseline})
    ref_df.to_csv('synthetic_normalbaseline.csv', index=False, header=False)

    noisy_df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': noisy_data})
    noisy_df.to_csv('synthetic_sample1.csv', index=False, header=False)

    # Create second sample with different characteristics
    noisy_data2 = ref_baseline.copy()
    for pos in [700, 1400, 2000, 2800]:
        peak_width = 40
        peak_height = np.random.uniform(0.08, 0.25)
        peak_mask = np.abs(wavenumbers - pos) < peak_width
        noisy_data2[peak_mask] += peak_height * np.exp(-((wavenumbers[peak_mask] - pos) / (peak_width/3))**2)

    noisy_data2 += (0.06 * np.sin(2 * np.pi * wavenumbers / 600) *
                   np.exp(-(wavenumbers - 400) / 1800) +
                   0.03 * np.sin(2 * np.pi * wavenumbers / 250))

    noisy_df2 = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': noisy_data2})
    noisy_df2.to_csv('synthetic_sample2.csv', index=False, header=False)

    print("✓ Created synthetic files:")
    print("  - synthetic_normalbaseline.csv (reference)")
    print("  - synthetic_sample1.csv (noisy data)")
    print("  - synthetic_sample2.csv (noisy data)")
    print("\nYou can now upload these files to test the program!")

# Example usage and instructions
if __name__ == "__main__":
    print("SINUSOIDAL BASELINE CORRECTION - GOOGLE COLAB VERSION")
    print("="*60)
    print("\nTo run the program:")
    print("1. Execute: corrector, results = main()")
    print("2. Upload your CSV files when prompted")
    print("3. Adjust threshold if needed")
    print("4. Review results and download corrected files")
    print("\nTo create synthetic demo data:")
    print("Execute: demo_with_synthetic_data()")
    print("\nFor help with individual functions:")
    print("Execute: help(SpectroscopicBaselineCorrector)")
