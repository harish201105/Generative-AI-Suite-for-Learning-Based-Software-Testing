#!/usr/bin/env python3
"""
Simple script to run the complexity benchmarking system.
This provides an easy way to benchmark AI models for time and space complexity analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.analysis.complexity_benchmarker import ComplexityBenchmarker

def main():
    """Run the complexity benchmarking system."""
    print("🚀 Starting Complexity Analysis Benchmarking")
    print("=" * 50)
    print("This will test GPT-4, DeepSeek, and Phi models")
    print("on their ability to correctly identify time and space complexity.")
    print()
    
    try:
        # Initialize the benchmarker
        benchmarker = ComplexityBenchmarker()
        
        # Run the benchmark
        results = benchmarker.run_benchmark()
        
        # Print results
        benchmarker.print_benchmark_results(results)
        
        # Save results
        benchmarker.save_benchmark_results(results)
        
        print("\n✅ Benchmarking completed successfully!")
        print("\nKey Features of the Improved System:")
        print("1. ✅ Accurate complexity analysis using enhanced AST parsing")
        print("2. ✅ Algorithm pattern detection for common algorithms")
        print("3. ✅ Proper handling of loops, recursion, and data structures")
        print("4. ✅ Comprehensive benchmarking of multiple AI models")
        print("5. ✅ Detailed accuracy metrics and model rankings")
        print("6. ✅ JSON export for further analysis")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 