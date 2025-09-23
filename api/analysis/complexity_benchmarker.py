#!/usr/bin/env python3
"""
Complexity Benchmarking System
This module provides comprehensive benchmarking of AI models for time and space complexity analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple
import json
import time
from datetime import datetime
from complexity_analyzer import ComplexityAnalyzer

# Import models
from models.gpt_model import GPTModel
from models.deepseek_model import DeepSeekModel
from models.phi_model import PhiModel

class ComplexityBenchmarker:
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.models = {
            'GPT-4': GPTModel(),
            'DeepSeek': DeepSeekModel(),
            'Phi': PhiModel()
        }
        
        # Standard test cases for benchmarking
        self.benchmark_test_cases = [
            {
                'name': 'Constant Time Operation',
                'code': '''
def constant_operation():
    x = 5
    y = 10
    return x + y
''',
                'expected_time': 'O(1)',
                'expected_space': 'O(1)'
            },
            {
                'name': 'Linear Search',
                'code': '''
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
''',
                'expected_time': 'O(n)',
                'expected_space': 'O(1)'
            },
            {
                'name': 'Binary Search',
                'code': '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
                'expected_time': 'O(log n)',
                'expected_space': 'O(1)'
            },
            {
                'name': 'Bubble Sort',
                'code': '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
''',
                'expected_time': 'O(n²)',
                'expected_space': 'O(1)'
            },
            {
                'name': 'Merge Sort',
                'code': '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
                'expected_time': 'O(n log n)',
                'expected_space': 'O(n)'
            },
            {
                'name': 'Recursive Fibonacci',
                'code': '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''',
                'expected_time': 'O(2^n)',
                'expected_space': 'O(n)'
            },
            {
                'name': 'Nested Loops',
                'code': '''
def matrix_operations(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = 0
    
    for i in range(rows):
        for j in range(cols):
            result += matrix[i][j]
    
    return result
''',
                'expected_time': 'O(n²)',
                'expected_space': 'O(1)'
            }
        ]

    def generate_complexity_prompt(self, code: str) -> str:
        """Generate a prompt specifically for complexity analysis."""
        return f"""Analyze the time and space complexity of the following Python code.

Code:
{code}

Please provide your analysis in the following JSON format:
{{
    "time_complexity": "O(...)",
    "space_complexity": "O(...)",
    "explanation": "Brief explanation of your analysis"
}}

Important:
- Use standard Big O notation (O(1), O(n), O(log n), O(n²), O(n log n), O(2^n), O(n!))
- Consider both time and space complexity
- Provide a brief explanation for your reasoning
- Return only valid JSON, no additional text"""

    def analyze_with_model(self, model_name: str, model: Any, code: str) -> Dict[str, Any]:
        """Analyze code complexity using a specific model."""
        try:
            start_time = time.time()
            
            # Create a custom prompt for complexity analysis
            prompt = self.generate_complexity_prompt(code)
            
            # Use the model's analyze_code method but with our custom prompt
            response = model.client.complete(
                messages=[
                    model.client.models.SystemMessage(prompt),
                    model.client.models.UserMessage(code)
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                top_p=0.9,
                model=model.model
            )
            
            analysis = response.choices[0].message.content
            end_time = time.time()
            
            # Try to parse the JSON response
            try:
                parsed_response = json.loads(analysis)
                return {
                    'model': model_name,
                    'time_complexity': parsed_response.get('time_complexity', 'Unknown'),
                    'space_complexity': parsed_response.get('space_complexity', 'Unknown'),
                    'explanation': parsed_response.get('explanation', ''),
                    'response_time': end_time - start_time,
                    'raw_response': analysis
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract complexity from text
                return self._extract_complexity_from_text(analysis, model_name, end_time - start_time)
                
        except Exception as e:
            return {
                'model': model_name,
                'error': str(e),
                'time_complexity': 'Error',
                'space_complexity': 'Error',
                'response_time': 0
            }

    def _extract_complexity_from_text(self, text: str, model_name: str, response_time: float) -> Dict[str, Any]:
        """Extract complexity information from text response when JSON parsing fails."""
        import re
        
        # Look for time complexity patterns
        time_patterns = [
            r'time.*complexity.*?O\([^)]+\)',
            r'O\([^)]+\).*time',
            r'Time.*?O\([^)]+\)'
        ]
        
        space_patterns = [
            r'space.*complexity.*?O\([^)]+\)',
            r'O\([^)]+\).*space',
            r'Space.*?O\([^)]+\)'
        ]
        
        time_complexity = 'Unknown'
        space_complexity = 'Unknown'
        
        # Extract time complexity
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the O(...) part
                o_match = re.search(r'O\([^)]+\)', match.group())
                if o_match:
                    time_complexity = o_match.group()
                    break
        
        # Extract space complexity
        for pattern in space_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the O(...) part
                o_match = re.search(r'O\([^)]+\)', match.group())
                if o_match:
                    space_complexity = o_match.group()
                    break
        
        return {
            'model': model_name,
            'time_complexity': time_complexity,
            'space_complexity': space_complexity,
            'explanation': 'Extracted from text response',
            'response_time': response_time,
            'raw_response': text
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmarking process."""
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'model_performance': {},
            'overall_rankings': []
        }
        
        print("Starting Complexity Benchmarking...")
        print("=" * 50)
        
        for test_case in self.benchmark_test_cases:
            print(f"\nAnalyzing: {test_case['name']}")
            
            # Get actual complexity using our analyzer
            actual_time, actual_space = self.complexity_analyzer.analyze_test_case(test_case['code'])
            
            test_result = {
                'name': test_case['name'],
                'code': test_case['code'],
                'actual_time_complexity': actual_time,
                'actual_space_complexity': actual_space,
                'expected_time_complexity': test_case['expected_time'],
                'expected_space_complexity': test_case['expected_space'],
                'model_predictions': {}
            }
            
            # Test each model
            for model_name, model in self.models.items():
                print(f"  Testing {model_name}...")
                
                prediction = self.analyze_with_model(model_name, model, test_case['code'])
                test_result['model_predictions'][model_name] = prediction
                
                # Calculate accuracy
                time_correct = prediction['time_complexity'] == actual_time
                space_correct = prediction['space_complexity'] == actual_space
                accuracy = (time_correct + space_correct) / 2
                
                prediction['accuracy'] = accuracy
                prediction['time_correct'] = time_correct
                prediction['space_correct'] = space_correct
            
            benchmark_results['test_cases'].append(test_result)
        
        # Calculate overall model performance
        model_scores = {}
        for model_name in self.models.keys():
            total_accuracy = 0
            total_time_accuracy = 0
            total_space_accuracy = 0
            total_response_time = 0
            test_count = 0
            
            for test_result in benchmark_results['test_cases']:
                if model_name in test_result['model_predictions']:
                    prediction = test_result['model_predictions'][model_name]
                    if 'accuracy' in prediction:
                        total_accuracy += prediction['accuracy']
                        total_time_accuracy += prediction.get('time_correct', False)
                        total_space_accuracy += prediction.get('space_correct', False)
                        total_response_time += prediction.get('response_time', 0)
                        test_count += 1
            
            if test_count > 0:
                model_scores[model_name] = {
                    'overall_accuracy': total_accuracy / test_count,
                    'time_accuracy': total_time_accuracy / test_count,
                    'space_accuracy': total_space_accuracy / test_count,
                    'avg_response_time': total_response_time / test_count,
                    'test_count': test_count
                }
        
        # Rank models by overall accuracy
        rankings = sorted(
            model_scores.items(),
            key=lambda x: x[1]['overall_accuracy'],
            reverse=True
        )
        
        benchmark_results['model_performance'] = model_scores
        benchmark_results['overall_rankings'] = rankings
        
        return benchmark_results

    def print_benchmark_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 60)
        print("COMPLEXITY ANALYSIS BENCHMARK RESULTS")
        print("=" * 60)
        
        # Print overall rankings
        print("\n🏆 MODEL RANKINGS:")
        print("-" * 40)
        for i, (model_name, scores) in enumerate(results['overall_rankings'], 1):
            print(f"{i}. {model_name}")
            print(f"   Overall Accuracy: {scores['overall_accuracy']:.2%}")
            print(f"   Time Complexity Accuracy: {scores['time_accuracy']:.2%}")
            print(f"   Space Complexity Accuracy: {scores['space_accuracy']:.2%}")
            print(f"   Average Response Time: {scores['avg_response_time']:.2f}s")
            print()
        
        # Print detailed test case results
        print("\n📊 DETAILED TEST CASE RESULTS:")
        print("-" * 40)
        
        for test_result in results['test_cases']:
            print(f"\nTest: {test_result['name']}")
            print(f"Actual: Time={test_result['actual_time_complexity']}, Space={test_result['actual_space_complexity']}")
            
            for model_name, prediction in test_result['model_predictions'].items():
                status = "✓" if prediction.get('accuracy', 0) == 1.0 else "✗"
                print(f"  {model_name}: Time={prediction['time_complexity']}, Space={prediction['space_complexity']} {status}")
        
        # Print summary statistics
        print("\n📈 SUMMARY STATISTICS:")
        print("-" * 40)
        for model_name, scores in results['model_performance'].items():
            print(f"{model_name}:")
            print(f"  Correct Predictions: {scores['test_count'] * 2} total complexity predictions")
            print(f"  Time Complexity Correct: {scores['time_accuracy'] * scores['test_count']:.0f}/{scores['test_count']}")
            print(f"  Space Complexity Correct: {scores['space_accuracy'] * scores['test_count']:.0f}/{scores['test_count']}")

    def save_benchmark_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complexity_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the complexity benchmarking system."""
    benchmarker = ComplexityBenchmarker()
    
    try:
        results = benchmarker.run_benchmark()
        benchmarker.print_benchmark_results(results)
        benchmarker.save_benchmark_results(results)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 