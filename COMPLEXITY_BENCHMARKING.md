# Complexity Analysis Benchmarking System

## Overview

This system provides comprehensive benchmarking of AI models (GPT-4, DeepSeek, Phi) for their ability to correctly identify time and space complexity of Python code. The system includes an improved `ComplexityAnalyzer` that provides accurate ground truth for comparison.

## Key Features

### 1. Enhanced Complexity Analyzer (`api/analysis/complexity_analyzer.py`)

- **Algorithm Pattern Detection**: Automatically detects common algorithms (binary search, sorting, etc.)
- **AST-based Analysis**: Uses Python's Abstract Syntax Tree for structural analysis
- **Loop Analysis**: Properly handles nested loops, while loops, and for loops
- **Recursion Detection**: Identifies and classifies recursive functions
- **Data Structure Analysis**: Analyzes usage of lists, dictionaries, sets, etc.
- **Refinement System**: Additional validation and refinement of complexity analysis

### 2. Comprehensive Benchmarking (`api/analysis/complexity_benchmarker.py`)

- **Multi-Model Testing**: Tests GPT-4, DeepSeek, and Phi models
- **Standardized Test Cases**: 7 different complexity scenarios
- **Accuracy Metrics**: Time complexity, space complexity, and overall accuracy
- **Performance Tracking**: Response times and detailed comparisons
- **Ranking System**: Automatic ranking of models by performance

### 3. Test Cases Included

1. **Constant Time Operation** - O(1) time, O(1) space
2. **Linear Search** - O(n) time, O(1) space
3. **Binary Search** - O(log n) time, O(1) space
4. **Bubble Sort** - O(n²) time, O(1) space
5. **Merge Sort** - O(n log n) time, O(n) space
6. **Recursive Fibonacci** - O(2^n) time, O(n) space
7. **Nested Loops** - O(n²) time, O(1) space

## Usage

### Quick Start

```bash
# Run the complete benchmarking system
python run_complexity_benchmark.py
```

### Individual Components

```python
# Test the complexity analyzer
python test_complexity_analyzer.py

# Run the benchmarker directly
python api/analysis/complexity_benchmarker.py
```

### Programmatic Usage

```python
from api.analysis.complexity_analyzer import ComplexityAnalyzer
from api.analysis.complexity_benchmarker import ComplexityBenchmarker

# Analyze a single piece of code
analyzer = ComplexityAnalyzer()
time_complexity, space_complexity = analyzer.analyze_test_case(code)

# Run full benchmarking
benchmarker = ComplexityBenchmarker()
results = benchmarker.run_benchmark()
benchmarker.print_benchmark_results(results)
```

## Output Format

### Benchmark Results

The system generates comprehensive results including:

1. **Model Rankings**: Ordered by overall accuracy
2. **Detailed Test Results**: Per-test-case analysis for each model
3. **Accuracy Metrics**: 
   - Overall accuracy
   - Time complexity accuracy
   - Space complexity accuracy
   - Average response time
4. **JSON Export**: Complete results saved to timestamped file

### Sample Output

```
🏆 MODEL RANKINGS:
1. GPT-4
   Overall Accuracy: 85.71%
   Time Complexity Accuracy: 85.71%
   Space Complexity Accuracy: 85.71%
   Average Response Time: 2.34s

2. DeepSeek
   Overall Accuracy: 78.57%
   Time Complexity Accuracy: 78.57%
   Space Complexity Accuracy: 78.57%
   Average Response Time: 1.89s

3. Phi
   Overall Accuracy: 64.29%
   Time Complexity Accuracy: 64.29%
   Space Complexity Accuracy: 64.29%
   Average Response Time: 1.45s
```

## System Architecture

### Complexity Analyzer Components

1. **Algorithm Detection**: Pattern matching for common algorithms
2. **AST Parser**: Structural analysis of Python code
3. **Loop Analyzer**: Detection and classification of loops
4. **Recursion Analyzer**: Identification of recursive functions
5. **Data Structure Analyzer**: Analysis of DS usage and implications
6. **Complexity Refiner**: Final validation and refinement

### Benchmarker Components

1. **Model Integration**: Connects to existing model classes
2. **Test Case Management**: Standardized test cases with known complexities
3. **Response Parser**: Handles JSON and text responses from models
4. **Metrics Calculator**: Computes accuracy and performance metrics
5. **Results Formatter**: Generates human-readable and JSON outputs

## Accuracy Improvements

### Previous Issues Fixed

1. **Incorrect Algorithm Detection**: Now properly identifies binary search, sorting algorithms
2. **Poor Loop Analysis**: Enhanced nesting detection and loop type classification
3. **Inadequate Recursion Handling**: Better recursive function detection and classification
4. **Simplistic Space Analysis**: More sophisticated space complexity calculation
5. **Limited Test Coverage**: Comprehensive test cases covering various complexity classes

### New Features

1. **Pattern-Based Detection**: Recognizes common algorithm patterns in code
2. **AST-Based Validation**: Uses Python's AST for structural analysis
3. **Refinement System**: Additional validation and correction of initial analysis
4. **Comprehensive Benchmarking**: Tests multiple models with standardized cases
5. **Detailed Metrics**: Granular accuracy tracking and performance analysis

## Integration with Existing Models

The system integrates seamlessly with your existing model classes:

- `GPTModel` from `api/models/gpt_model.py`
- `DeepSeekModel` from `api/models/deepseek_model.py`
- `PhiModel` from `api/models/phi_model.py`

The benchmarker uses the same interface as your existing models, making it easy to add new models or modify existing ones.

## Future Enhancements

1. **More Test Cases**: Additional complexity scenarios
2. **Custom Test Cases**: User-defined test cases
3. **Real-time Analysis**: Live complexity analysis during development
4. **Integration with IDEs**: Plugin for popular development environments
5. **Historical Tracking**: Track model performance over time
6. **Advanced Metrics**: More sophisticated accuracy measurements

## Requirements

- Python 3.7+
- Required packages: `ast`, `json`, `time`, `datetime`
- Access to AI models (GPT-4, DeepSeek, Phi)
- Environment variables configured for model access

## Troubleshooting

### Common Issues

1. **Model Connection Errors**: Check environment variables and API keys
2. **JSON Parsing Errors**: Models may return non-JSON responses
3. **Complexity Detection Issues**: Check code syntax and structure
4. **Performance Issues**: Large code samples may slow analysis

### Debug Mode

Enable debug output by modifying the benchmarker:

```python
# Add debug=True to see detailed analysis
benchmarker = ComplexityBenchmarker(debug=True)
```

## Contributing

To add new test cases or improve the analyzer:

1. Add test cases to `benchmark_test_cases` in `ComplexityBenchmarker`
2. Enhance algorithm detection patterns in `ComplexityAnalyzer`
3. Improve AST analysis methods for better accuracy
4. Add new model integrations as needed

## License

This system is part of the LBST project and follows the same licensing terms. 