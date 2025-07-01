"""
Local Sinusoidal Baseline Correction for Spectroscopic Data
==========================================================

A Python program that automatically processes CSV files in the current directory
and outputs corrected results back to the same folder.

Author: AI Assistant
Date: 2025-07-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
import os
import glob
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings('ignore')

class LocalSpectroscopicBaselineCorrector:
    """
    A class for performing sinusoidal baseline correction on local CSV files.
    """
    
    def __init__(self, directory: str = "."):
        self.directory = Path(directory)
        self.data_files = {}
        self.reference_data = None
        self.reference_file = None
        self.threshold = None
        self.fitted_parameters = {}
        self.corrected_data = {}
        
    def scan_directory(self) -> Tuple[Optional[str], List[str]]:
        """Scan directory for CSV files and identify reference file."""
        print(f"Scanning directory: {self.directory.absolute()}")
        
        # Find all CSV files
        csv_files = list(self.directory.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.directory}")
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {file.name}")
        
        # Identify reference file
        reference_files = [f for f in csv_files if "_normalbaseline" in f.name.lower()]
        
        if not reference_files:
            raise ValueError("No reference file (*_normalbaseline.csv) found!")
        
        if len(reference_files) > 1:
            print(f"Warning: Multiple reference files found. Using: {reference_files[0].name}")
        
        reference_file = reference_files[0]
        data_files = [f for f in csv_files if f != reference_file]
        
        if not data_files:
            raise ValueError("No data files found (only reference file present)!")
        
        # Load reference file
        try:
            self.reference_data = pd.read_csv(reference_file, header=None, names=['wavenumber', 'absorbance'])
            self.reference_file = reference_file.name
            print(f"✓ Reference file loaded: {reference_file.name}")
        except Exception as e:
            raise ValueError(f"Error loading reference file {reference_file.name}: {e}")
        
        # Load data files
        for file in data_files:
            try:
                self.data_files[file.name] = pd.read_csv(file, header=None, names=['wavenumber', 'absorbance'])
                print(f"✓ Data file loaded: {file.name}")
            except Exception as e:
                print(f"✗ Error loading {file.name}: {e}")
        
        print(f"\nSuccessfully loaded:")
        print(f"  - Reference: {reference_file.name}")
        print(f"  - Data files: {len(self.data_files)}")
        
        return reference_file.name, list(self.data_files.keys())
    
    def complex_sinusoidal_baseline(self, x: np.ndarray, *params) -> np.ndarray:
        """
        Complex sinusoidal baseline model with multiple harmonics and damping.
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
        """Automatically detect threshold using the reference baseline file."""
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
            plt.title(f'Reference Baseline Analysis - {self.reference_file}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.directory / f"threshold_analysis_{self.reference_file.replace('.csv', '.png')}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Threshold analysis plot saved: {plot_path.name}")
            plt.show()
        
        return suggested_threshold
    
    def set_threshold(self, threshold: Optional[float] = None, interactive: bool = True) -> float:
        """Set the threshold for peak detection."""
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
        """Fit sinusoidal baseline to data, excluding peaks above threshold."""
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

    def correct_spectrum(self, filename: str, plot: bool = True, save_plot: bool = True) -> Dict:
        """Perform baseline correction on a single spectrum."""
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
            self.plot_correction(result, save_plot=save_plot)

        return result

    def plot_correction(self, result: Dict, save_plot: bool = True):
        """Create comprehensive visualization of the baseline correction."""
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

        if save_plot:
            plot_filename = result['filename'].replace('.csv', '_correction_plot.png')
            plot_path = self.directory / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {plot_filename}")

        plt.show()

        # Print statistics
        print(f"Baseline correction statistics for {result['filename']}:")
        print(f"  - Original range: {y_orig.min():.4f} to {y_orig.max():.4f}")
        print(f"  - Corrected range: {y_corrected.min():.4f} to {y_corrected.max():.4f}")
        print(f"  - Baseline points used: {np.sum(baseline_mask)}/{len(x)} ({100*np.sum(baseline_mask)/len(x):.1f}%)")
        print(f"  - RMS baseline deviation: {np.sqrt(np.mean((y_orig[baseline_mask] - y_baseline[baseline_mask])**2)):.4f}")

    def process_all_files(self, plot_individual: bool = True, save_plots: bool = True) -> Dict:
        """Process all loaded data files with progress indication."""
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
                result = self.correct_spectrum(filename, plot=plot_individual, save_plot=save_plots)
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

    def export_results(self) -> List[str]:
        """Export corrected data and parameters to files in the same directory."""
        if not self.corrected_data:
            raise ValueError("No corrected data to export! Process files first.")

        print(f"\nExporting results to current directory...")

        exported_files = []

        for filename, result in self.corrected_data.items():
            if result is None:
                continue

            # Export corrected CSV
            base_name = filename.replace('.csv', '')
            corrected_filename = f"{base_name}_corrected.csv"
            corrected_path = self.directory / corrected_filename

            corrected_df = pd.DataFrame({
                'wavenumber': result['original_x'],
                'absorbance': result['corrected_y']
            })
            corrected_df.to_csv(corrected_path, index=False, header=False)
            exported_files.append(corrected_filename)
            print(f"✓ Saved: {corrected_filename}")

            # Export parameters and statistics
            params_filename = f"{base_name}_correction_info.txt"
            params_path = self.directory / params_filename

            with open(params_path, 'w') as f:
                f.write(f"Baseline Correction Report\n")
                f.write(f"{'='*40}\n")
                f.write(f"Original file: {filename}\n")
                f.write(f"Corrected file: {corrected_filename}\n")
                f.write(f"Reference file: {self.reference_file}\n")
                f.write(f"Processing date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write(f"Parameters:\n")
                f.write(f"  Threshold: {self.threshold:.6f}\n")
                f.write(f"  Baseline points used: {np.sum(result['baseline_mask'])}/{len(result['original_x'])}\n")
                f.write(f"  Baseline coverage: {100*np.sum(result['baseline_mask'])/len(result['original_x']):.1f}%\n\n")

                f.write(f"Statistics:\n")
                f.write(f"  Original range: {result['original_y'].min():.6f} to {result['original_y'].max():.6f}\n")
                f.write(f"  Corrected range: {result['corrected_y'].min():.6f} to {result['corrected_y'].max():.6f}\n")
                f.write(f"  RMS baseline deviation: {np.sqrt(np.mean((result['original_y'][result['baseline_mask']] - result['fitted_baseline'][result['baseline_mask']])**2)):.6f}\n")

                # Add data quality metrics
                baseline_coverage = np.sum(result['baseline_mask']) / len(result['baseline_mask'])
                peak_preservation = np.sum(result['corrected_y'] > 0) / len(result['corrected_y'])

                f.write(f"\nQuality Metrics:\n")
                f.write(f"  Baseline coverage: {baseline_coverage:.3f}\n")
                f.write(f"  Peak preservation ratio: {peak_preservation:.3f}\n")
                f.write(f"  Correction magnitude: {np.std(result['fitted_baseline']):.6f}\n")

            exported_files.append(params_filename)
            print(f"✓ Saved: {params_filename}")

        print(f"\n✓ Exported {len(exported_files)} files to {self.directory.absolute()}")
        return exported_files

    def create_summary_report(self) -> str:
        """Create a summary report of all processing results."""
        if not self.corrected_data:
            raise ValueError("No corrected data available!")

        summary_filename = "baseline_correction_summary.txt"
        summary_path = self.directory / summary_filename

        with open(summary_path, 'w') as f:
            f.write("BASELINE CORRECTION SUMMARY REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Processing date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Reference file: {self.reference_file}\n")
            f.write(f"Threshold used: {self.threshold:.6f}\n")
            f.write(f"Total files processed: {len(self.corrected_data)}\n\n")

            successful = [r for r in self.corrected_data.values() if r is not None]
            f.write(f"Successfully processed: {len(successful)}/{len(self.corrected_data)}\n\n")

            if successful:
                f.write("INDIVIDUAL FILE RESULTS:\n")
                f.write("-" * 30 + "\n")

                for filename, result in self.corrected_data.items():
                    if result is not None:
                        baseline_coverage = np.sum(result['baseline_mask']) / len(result['baseline_mask'])
                        correction_magnitude = np.std(result['fitted_baseline'])

                        f.write(f"\n{filename}:\n")
                        f.write(f"  Data points: {len(result['original_x'])}\n")
                        f.write(f"  Baseline coverage: {baseline_coverage:.1%}\n")
                        f.write(f"  Original range: {result['original_y'].min():.4f} to {result['original_y'].max():.4f}\n")
                        f.write(f"  Corrected range: {result['corrected_y'].min():.4f} to {result['corrected_y'].max():.4f}\n")
                        f.write(f"  Correction magnitude: {correction_magnitude:.4f}\n")
                    else:
                        f.write(f"\n{filename}: FAILED\n")

                # Overall statistics
                all_baseline_coverage = [np.sum(r['baseline_mask'])/len(r['baseline_mask'])
                                       for r in successful]
                all_correction_magnitude = [np.std(r['fitted_baseline']) for r in successful]

                f.write(f"\nOVERALL STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average baseline coverage: {np.mean(all_baseline_coverage):.1%}\n")
                f.write(f"Average correction magnitude: {np.mean(all_correction_magnitude):.4f}\n")
                f.write(f"Baseline coverage range: {np.min(all_baseline_coverage):.1%} - {np.max(all_baseline_coverage):.1%}\n")

        print(f"✓ Summary report saved: {summary_filename}")
        return summary_filename

def process_local_directory(directory: str = ".", interactive: bool = True,
                          plot_individual: bool = True, save_plots: bool = True) -> Tuple[LocalSpectroscopicBaselineCorrector, Dict]:
    """
    Main function to process all CSV files in a directory.

    Args:
        directory: Directory containing CSV files (default: current directory)
        interactive: Whether to allow interactive threshold adjustment
        plot_individual: Whether to show individual plots for each file
        save_plots: Whether to save plots as PNG files

    Returns:
        Tuple of (corrector instance, results dictionary)
    """
    print("="*80)
    print("LOCAL SINUSOIDAL BASELINE CORRECTION")
    print("="*80)
    print(f"Processing directory: {Path(directory).absolute()}")

    try:
        # Initialize corrector
        corrector = LocalSpectroscopicBaselineCorrector(directory)

        # Scan directory and load files
        print(f"\n{'-'*60}")
        print("STEP 1: SCANNING DIRECTORY")
        print(f"{'-'*60}")
        reference_file, data_files = corrector.scan_directory()

        # Detect and set threshold
        print(f"\n{'-'*60}")
        print("STEP 2: THRESHOLD DETECTION")
        print(f"{'-'*60}")
        threshold = corrector.set_threshold(interactive=interactive)

        # Process all files
        print(f"\n{'-'*60}")
        print("STEP 3: BASELINE CORRECTION")
        print(f"{'-'*60}")
        results = corrector.process_all_files(plot_individual=plot_individual, save_plots=save_plots)

        # Export results
        print(f"\n{'-'*60}")
        print("STEP 4: EXPORT RESULTS")
        print(f"{'-'*60}")
        exported_files = corrector.export_results()
        summary_file = corrector.create_summary_report()

        # Final summary
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*80}")
        successful = sum(1 for r in results.values() if r is not None)
        print(f"✓ Successfully processed: {successful}/{len(data_files)} files")
        print(f"✓ Threshold used: {threshold:.6f}")
        print(f"✓ Files exported: {len(exported_files)}")
        print(f"✓ Summary report: {summary_file}")
        print(f"✓ All results saved to: {Path(directory).absolute()}")

        return corrector, results

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your files and directory structure.")
        raise

def create_demo_data(directory: str = ".") -> List[str]:
    """
    Create synthetic demonstration data in the specified directory.

    Args:
        directory: Directory to create demo files in

    Returns:
        List of created filenames
    """
    print("Creating synthetic demonstration data...")

    directory = Path(directory)

    # Generate synthetic wavenumbers
    wavenumbers = np.linspace(400, 4000, 1000)

    # Create reference baseline (clean)
    np.random.seed(42)
    ref_baseline = (0.1 +
                   0.05 * np.sin(2 * np.pi * wavenumbers / 1000) * np.exp(-(wavenumbers - 400) / 2000) +
                   0.02 * np.sin(2 * np.pi * wavenumbers / 500) +
                   np.random.normal(0, 0.002, len(wavenumbers)))

    created_files = []

    # Save reference file
    ref_df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': ref_baseline})
    ref_filename = 'demo_normalbaseline.csv'
    ref_path = directory / ref_filename
    ref_df.to_csv(ref_path, index=False, header=False)
    created_files.append(ref_filename)
    print(f"✓ Created: {ref_filename}")

    # Create multiple sample files with different characteristics
    sample_configs = [
        {'name': 'demo_sample1.csv', 'peaks': [800, 1200, 1600, 2400, 3200],
         'baseline_freq': 700, 'baseline_amp': 0.08},
        {'name': 'demo_sample2.csv', 'peaks': [700, 1400, 2000, 2800],
         'baseline_freq': 600, 'baseline_amp': 0.06},
        {'name': 'demo_sample3.csv', 'peaks': [900, 1300, 1800, 2200, 3000],
         'baseline_freq': 800, 'baseline_amp': 0.07}
    ]

    for config in sample_configs:
        # Start with reference baseline
        sample_data = ref_baseline.copy()

        # Add peaks
        for peak_pos in config['peaks']:
            peak_width = np.random.uniform(30, 50)
            peak_height = np.random.uniform(0.08, 0.25)
            peak_mask = np.abs(wavenumbers - peak_pos) < peak_width
            sample_data[peak_mask] += peak_height * np.exp(-((wavenumbers[peak_mask] - peak_pos) / (peak_width/3))**2)

        # Add sinusoidal baseline distortion
        sample_data += (config['baseline_amp'] * np.sin(2 * np.pi * wavenumbers / config['baseline_freq']) *
                       np.exp(-(wavenumbers - 400) / 1500) +
                       0.03 * np.sin(2 * np.pi * wavenumbers / 250) +
                       np.random.normal(0, 0.001, len(wavenumbers)))

        # Save sample file
        sample_df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': sample_data})
        sample_path = directory / config['name']
        sample_df.to_csv(sample_path, index=False, header=False)
        created_files.append(config['name'])
        print(f"✓ Created: {config['name']}")

    print(f"\n✓ Created {len(created_files)} demonstration files in {directory.absolute()}")
    print("Files created:")
    for filename in created_files:
        print(f"  - {filename}")

    return created_files

# Main execution functions
def main():
    """Main function - process CSV files in current directory."""
    return process_local_directory()

def main_with_demo():
    """Create demo data and process it."""
    print("Creating demonstration data...")
    create_demo_data()
    print("\nProcessing demonstration data...")
    return process_local_directory()

def quick_process(directory: str = ".", threshold: float = None):
    """Quick processing without interactive prompts."""
    corrector = LocalSpectroscopicBaselineCorrector(directory)
    corrector.scan_directory()

    if threshold is None:
        threshold = corrector.detect_threshold(plot=False)

    corrector.set_threshold(threshold, interactive=False)
    results = corrector.process_all_files(plot_individual=False, save_plots=True)
    corrector.export_results()
    corrector.create_summary_report()

    return corrector, results

# Example usage and instructions
if __name__ == "__main__":
    print("LOCAL SINUSOIDAL BASELINE CORRECTION")
    print("="*60)
    print("\nThis program processes CSV files in the current directory.")
    print("\nRequired files:")
    print("• One reference file: *_normalbaseline.csv")
    print("• One or more data files: *.csv")
    print("\nFile format:")
    print("• CSV with no headers")
    print("• Column 1: Wavenumbers")
    print("• Column 2: Absorbance")

    print("\nUsage options:")
    print("1. Process existing files:")
    print("   corrector, results = main()")
    print("\n2. Create demo data and process:")
    print("   corrector, results = main_with_demo()")
    print("\n3. Quick processing (no interaction):")
    print("   corrector, results = quick_process()")
    print("\n4. Process specific directory:")
    print("   corrector, results = process_local_directory('/path/to/data')")
    print("\n5. Create demo data only:")
    print("   create_demo_data()")

    print(f"\nCurrent directory: {Path('.').absolute()}")

    # Check for existing CSV files
    csv_files = list(Path('.').glob('*.csv'))
    if csv_files:
        print(f"\nFound {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {file.name}")

        reference_files = [f for f in csv_files if '_normalbaseline' in f.name.lower()]
        if reference_files:
            print(f"\nReference file detected: {reference_files[0].name}")
            print("Ready to process! Run: corrector, results = main()")
        else:
            print("\nNo reference file (*_normalbaseline.csv) found.")
            print("Create one or run: create_demo_data()")
    else:
        print("\nNo CSV files found in current directory.")
        print("Run: create_demo_data() to create demonstration files.")
