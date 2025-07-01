#!/usr/bin/env python3
"""
Simple script to run baseline correction on CSV files in the current directory.

Usage:
    python run_baseline_correction.py [options]

Options:
    --demo          Create demo data and process it
    --quick         Quick processing without interactive prompts
    --threshold X   Set specific threshold value
    --no-plots      Don't show individual plots
    --directory X   Process files in specific directory
"""

import sys
import argparse
from pathlib import Path

# Import the local baseline correction module
from local_baseline_correction import (
    process_local_directory, 
    create_demo_data, 
    quick_process,
    LocalSpectroscopicBaselineCorrector
)

def main():
    parser = argparse.ArgumentParser(
        description="Sinusoidal baseline correction for spectroscopic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_baseline_correction.py                    # Interactive processing
    python run_baseline_correction.py --demo             # Create and process demo data
    python run_baseline_correction.py --quick            # Quick processing
    python run_baseline_correction.py --threshold 0.01   # Set specific threshold
    python run_baseline_correction.py --directory /path  # Process specific directory
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Create demonstration data and process it')
    parser.add_argument('--quick', action='store_true',
                       help='Quick processing without interactive prompts')
    parser.add_argument('--threshold', type=float,
                       help='Set specific threshold value')
    parser.add_argument('--no-plots', action='store_true',
                       help="Don't show individual plots")
    parser.add_argument('--directory', type=str, default='.',
                       help='Directory containing CSV files (default: current)')
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            print("Creating demonstration data...")
            create_demo_data(args.directory)
            print("\nProcessing demonstration data...")
            corrector, results = process_local_directory(
                directory=args.directory,
                interactive=not args.quick,
                plot_individual=not args.no_plots
            )
            
        elif args.quick:
            print("Quick processing mode...")
            corrector, results = quick_process(
                directory=args.directory,
                threshold=args.threshold
            )
            
        else:
            print("Interactive processing mode...")
            corrector, results = process_local_directory(
                directory=args.directory,
                interactive=True,
                plot_individual=not args.no_plots
            )
        
        # Print final summary
        successful = sum(1 for r in results.values() if r is not None)
        print(f"\n🎉 Processing completed!")
        print(f"Successfully processed: {successful}/{len(results)} files")
        print(f"Results saved to: {Path(args.directory).absolute()}")
        
        return corrector, results
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    corrector, results = main()
