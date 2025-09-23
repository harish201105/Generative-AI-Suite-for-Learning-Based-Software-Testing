from typing import Dict, Any, List, Tuple, Set, Optional
import re
import ast
import json
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from advanced_complexity_analyzer import AdvancedPythonComplexityAnalyzer

class ComplexityAnalyzer:
    """
    Wrapper class that uses the Advanced Python Complexity Analyzer for accurate results.
    Maintains backward compatibility with the existing API.
    """
    def __init__(self):
        self.advanced_analyzer = AdvancedPythonComplexityAnalyzer()
        
        # Keep the original complexity patterns for reference
        self.complexity_patterns = {
            'O(1)': ['constant', 'single operation', 'direct access'],
            'O(log n)': ['binary search', 'divide by 2', 'logarithmic'],
            'O(n)': ['linear', 'single loop', 'traverse'],
            'O(n log n)': ['sort', 'merge', 'heap'],
            'O(n²)': ['nested loop', 'bubble sort', 'quadratic'],
            'O(2^n)': ['recursive fibonacci', 'exponential'],
            'O(n!)': ['factorial', 'permutation']
        }
        
        # Common algorithm patterns
        self.algorithm_patterns = {
            'binary_search': ['binary', 'search', 'sorted', 'middle'],
            'linear_search': ['linear', 'search', 'find', 'traverse'],
            'bubble_sort': ['bubble', 'sort', 'swap', 'adjacent'],
            'merge_sort': ['merge', 'sort', 'divide', 'conquer'],
            'quick_sort': ['quick', 'sort', 'pivot', 'partition'],
            'fibonacci': ['fibonacci', 'fib', 'recursive'],
            'factorial': ['factorial', 'fact', '!'],
            'dfs': ['depth', 'first', 'search', 'dfs'],
            'bfs': ['breadth', 'first', 'search', 'bfs']
        }

    def analyze_test_case(self, test_code: str) -> Tuple[str, str]:
        """Analyze a single test case for time and space complexity using the advanced analyzer."""
        # First try pattern detection for quick results
        pattern_result = self._detect_algorithm_patterns(test_code)
        if pattern_result:
            return pattern_result
        
        # For more sophisticated analysis, try to identify the main algorithmic function
        # and analyze it specifically to avoid counting input data as part of space complexity
        main_function = self._identify_main_function(test_code)
        if main_function:
            return self.advanced_analyzer.analyze_code_complexity(test_code, main_function)
        
        # Fallback to analyzing entire code
        return self.advanced_analyzer.analyze_code_complexity(test_code)

    def _identify_main_function(self, test_code: str) -> Optional[str]:
        """Identify the main algorithmic function to analyze."""
        import ast
        try:
            tree = ast.parse(test_code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # Look for common algorithmic function patterns
            algorithmic_patterns = [
                'quick_sort', 'quicksort', 'merge_sort', 'mergesort', 
                'bubble_sort', 'selection_sort', 'insertion_sort',
                'binary_search', 'linear_search', 'dfs', 'bfs',
                'fibonacci', 'factorial', 'matrix_multiply', 'matrix_mult'
            ]
            
            for func_name in functions:
                for pattern in algorithmic_patterns:
                    if pattern in func_name.lower():
                        return func_name
            
            # If no specific pattern found, return the longest function name
            # (heuristic: main algorithm is usually more descriptive)
            if functions:
                return max(functions, key=len)
                
        except:
            pass
        
        return None

    def _detect_algorithm_patterns(self, code_str: str) -> Tuple[str, str]:
        """Detect common algorithms and return their known complexities."""
        # Binary search - look for specific patterns
        if ('binary' in code_str or 'middle' in code_str) and 'search' in code_str:
            if 'left' in code_str and 'right' in code_str and 'mid' in code_str:
                return 'O(log n)', 'O(1)'
        
        # Linear search - look for simple loops with search
        if 'search' in code_str and 'for' in code_str and 'range' in code_str:
            if 'target' in code_str or 'find' in code_str:
                return 'O(n)', 'O(1)'
        
        # Bubble sort - look for nested loops with swap
        if 'bubble' in code_str or ('sort' in code_str and 'swap' in code_str):
            if 'for' in code_str and 'range' in code_str:
                return 'O(n²)', 'O(1)'
        
        # Merge sort - look for divide and conquer
        if 'merge' in code_str and 'sort' in code_str:
            if 'mid' in code_str and 'left' in code_str and 'right' in code_str:
                return 'O(n log n)', 'O(n)'
        
        # Matrix operations
        if 'matrix' in code_str and 'multiply' in code_str:
            if code_str.count('for') >= 3:  # Triple nested loop
                return 'O(n³)', 'O(n²)'  # Matrix multiplication
        
        # Quick sort - look for pivot and partition
        if 'quick' in code_str and 'sort' in code_str:
            if 'pivot' in code_str or 'partition' in code_str:
                return 'O(n log n)', 'O(log n)'  # Average case
        
        # Recursive algorithms
        if 'fibonacci' in code_str and ('dp' in code_str or 'memo' in code_str or '@lru_cache' in code_str):
            return 'O(n)', 'O(n)'  # Memoized/DP fibonacci
        elif 'fibonacci' in code_str or ('fib' in code_str and 'recursive' in code_str):
            if 'return' in code_str and 'fibonacci' in code_str and 'dp' not in code_str:
                return 'O(2^n)', 'O(n)'  # Naive recursive
        
        if 'factorial' in code_str or ('fact' in code_str and '!' in code_str):
            if 'return' in code_str and ('factorial' in code_str or 'fact' in code_str):
                return 'O(n)', 'O(n)'  # Recursive factorial
        
        # Graph algorithms
        if 'dfs' in code_str or ('depth' in code_str and 'first' in code_str):
            return 'O(V + E)', 'O(V)'  # V=vertices, E=edges
        
        if 'bfs' in code_str or ('breadth' in code_str and 'first' in code_str):
            return 'O(V + E)', 'O(V)'
        
        return None

    def _analyze_ast_complexity(self, tree: ast.AST) -> Tuple[str, str]:
        """Analyze AST for structural complexity patterns."""
        loop_analysis = self._analyze_loops(tree)
        recursion_analysis = self._analyze_recursion(tree)
        data_structure_analysis = self._analyze_data_structures(tree)
        
        # Determine time complexity
        time_complexity = self._determine_time_complexity(loop_analysis, recursion_analysis, data_structure_analysis)
        
        # Determine space complexity
        space_complexity = self._determine_space_complexity(loop_analysis, recursion_analysis, data_structure_analysis)
        
        return time_complexity, space_complexity

    def _analyze_loops(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze loop structures and their complexity."""
        loop_info = {
            'max_nesting': 0,
            'loop_types': [],
            'loop_variables': set(),
            'loop_bounds': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                nesting = self._get_loop_nesting_level(node, tree)
                loop_info['max_nesting'] = max(loop_info['max_nesting'], nesting)
                
                # Analyze loop type and bounds
                loop_type, bounds = self._analyze_single_loop(node)
                loop_info['loop_types'].append(loop_type)
                loop_info['loop_bounds'].append(bounds)
                
                # Extract loop variables
                if isinstance(node, ast.For) and hasattr(node.target, 'id'):
                    loop_info['loop_variables'].add(node.target.id)
        
        return loop_info

    def _get_loop_nesting_level(self, node: ast.AST, tree: ast.AST) -> int:
        """Calculate the nesting level of a loop."""
        level = 1
        current = node
        
        # Walk up the tree to find parent loops
        for parent in ast.walk(tree):
            if parent is node:
                continue
            if isinstance(parent, (ast.For, ast.While)):
                if self._is_descendant(current, parent):
                    level += 1
        
        return level

    def _is_descendant(self, child: ast.AST, parent: ast.AST) -> bool:
        """Check if child is a descendant of parent in the AST."""
        for node in ast.walk(parent):
            if node is child:
                return True
        return False

    def _analyze_single_loop(self, node: ast.AST) -> Tuple[str, str]:
        """Analyze a single loop for type and bounds."""
        if isinstance(node, ast.For):
            # Check if it's a range-based loop
            if hasattr(node.iter, 'func') and hasattr(node.iter.func, 'id'):
                if node.iter.func.id == 'range':
                    return 'range_loop', 'O(n)'
                elif node.iter.func.id in ['enumerate', 'zip']:
                    return 'enumerate_loop', 'O(n)'
            
            # Check if it's iterating over a data structure
            return 'collection_loop', 'O(n)'
        
        elif isinstance(node, ast.While):
            # Analyze while loop condition
            return 'while_loop', 'O(n)'  # Default assumption
        
        return 'unknown_loop', 'O(n)'

    def _analyze_recursion(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze recursion patterns."""
        recursion_info = {
            'has_recursion': False,
            'recursion_type': None,
            'recursion_depth': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_recursive_call(node, tree):
                    recursion_info['has_recursion'] = True
                    recursion_info['recursion_type'] = self._classify_recursion(node, tree)
                    recursion_info['recursion_depth'] = self._estimate_recursion_depth(node, tree)
        
        return recursion_info

    def _is_recursive_call(self, node: ast.Call, tree: ast.AST) -> bool:
        """Check if a function call is recursive."""
        if not isinstance(node.func, ast.Name):
            return False
        
        # Find the function definition
        for item in ast.walk(tree):
            if isinstance(item, ast.FunctionDef) and item.name == node.func.id:
                # Check if the function calls itself
                for call in ast.walk(item):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                        if call.func.id == item.name:
                            return True
        return False

    def _classify_recursion(self, node: ast.Call, tree: ast.AST) -> str:
        """Classify the type of recursion."""
        # This is a simplified classification
        # In practice, you'd need more sophisticated analysis
        return 'linear_recursion'  # Default assumption

    def _estimate_recursion_depth(self, node: ast.Call, tree: ast.AST) -> int:
        """Estimate the recursion depth."""
        # This is a simplified estimation
        return 1  # Default assumption

    def _analyze_data_structures(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze data structure usage and their complexity implications."""
        ds_info = {
            'lists': 0,
            'dicts': 0,
            'sets': 0,
            'stacks': 0,
            'queues': 0,
            'trees': 0,
            'graphs': 0
        }
        
        for node in ast.walk(tree):
            # Check for list operations
            if isinstance(node, ast.List):
                ds_info['lists'] += 1
            elif isinstance(node, ast.Dict):
                ds_info['dicts'] += 1
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                    if 'stack' in func_name or 'push' in func_name or 'pop' in func_name:
                        ds_info['stacks'] += 1
                    elif 'queue' in func_name or 'enqueue' in func_name or 'dequeue' in func_name:
                        ds_info['queues'] += 1
                    elif 'tree' in func_name or 'node' in func_name:
                        ds_info['trees'] += 1
                    elif 'graph' in func_name or 'vertex' in func_name or 'edge' in func_name:
                        ds_info['graphs'] += 1
        
        return ds_info

    def _determine_time_complexity(self, loop_analysis: Dict, recursion_analysis: Dict, ds_analysis: Dict) -> str:
        """Determine overall time complexity based on analysis."""
        # Check for recursion first (usually dominates)
        if recursion_analysis['has_recursion']:
            if recursion_analysis['recursion_type'] == 'linear_recursion':
                return 'O(n)'
            else:
                return 'O(2^n)'  # Default for complex recursion
        
        # Check loop nesting
        if loop_analysis['max_nesting'] >= 3:
            return 'O(n³)'
        elif loop_analysis['max_nesting'] == 2:
            return 'O(n²)'
        elif loop_analysis['max_nesting'] == 1:
            # Check if it's a simple linear operation
            if len(loop_analysis['loop_types']) == 1 and 'range_loop' in loop_analysis['loop_types']:
                return 'O(n)'
            else:
                return 'O(n)'
        
        # Check data structure operations
        if ds_analysis['lists'] > 0 or ds_analysis['dicts'] > 0:
            return 'O(n)'  # Assuming linear operations
        
        return 'O(1)'

    def _determine_space_complexity(self, loop_analysis: Dict, recursion_analysis: Dict, ds_analysis: Dict) -> str:
        """Determine overall space complexity based on analysis."""
        # Recursion typically uses O(n) space for call stack
        if recursion_analysis['has_recursion']:
            return 'O(n)'
        
        # Data structures
        total_ds = sum(ds_analysis.values())
        if total_ds > 0:
            return 'O(n)'  # Assuming linear space usage
        
        # Simple loops with constant space
        if loop_analysis['max_nesting'] > 0:
            return 'O(1)'
        
        return 'O(1)'

    def analyze_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all test cases and return complexity metrics using the advanced analyzer."""
        return self.advanced_analyzer.analyze_test_cases(test_cases)

    def _get_worst_complexity(self, comp1: str, comp2: str) -> str:
        """Determine the worst case complexity between two complexities."""
        complexity_order = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(n³)', 'O(2^n)', 'O(n!)']
        try:
            idx1 = complexity_order.index(comp1)
            idx2 = complexity_order.index(comp2)
            return complexity_order[max(idx1, idx2)]
        except ValueError:
            return comp1  # Default to first complexity if comparison fails

    def _refine_complexity_analysis(self, tree: ast.AST, time_complexity: str, space_complexity: str) -> Tuple[str, str]:
        """Refine complexity analysis based on more detailed code structure."""
        # Look for specific patterns in the AST
        has_while_loop = False
        has_for_loop = False
        has_nested_loops = False
        loop_variables = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                has_while_loop = True
            elif isinstance(node, ast.For):
                has_for_loop = True
                # Check if it's a range-based loop
                if hasattr(node.iter, 'func') and hasattr(node.iter.func, 'id'):
                    if node.iter.func.id == 'range':
                        loop_variables.add('range_loop')
                    elif node.iter.func.id in ['enumerate', 'zip']:
                        loop_variables.add('enumerate_loop')
        
        # Check for nested loops
        loop_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_count += 1
        has_nested_loops = loop_count > 1
        
        # Refine time complexity based on loop patterns
        if has_nested_loops:
            refined_time = 'O(n²)'
        elif has_for_loop and 'range_loop' in loop_variables:
            refined_time = 'O(n)'
        elif has_while_loop:
            refined_time = 'O(n)'  # Default for while loops
        else:
            refined_time = time_complexity
        
        # Refine space complexity
        if has_nested_loops or has_for_loop or has_while_loop:
            refined_space = 'O(1)'  # Most loops use constant space
        else:
            refined_space = space_complexity
        
        return refined_time, refined_space

    def get_benchmark_metrics(self, model_predictions: List[Dict[str, Any]], actual_complexities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare model predictions with actual complexities and return benchmark metrics."""
        metrics = {
            'time_complexity_accuracy': 0,
            'space_complexity_accuracy': 0,
            'overall_accuracy': 0,
            'model_rankings': [],
            'detailed_comparisons': []
        }
        
        correct_time = 0
        correct_space = 0
        total_tests = len(actual_complexities)
        
        for i, (prediction, actual) in enumerate(zip(model_predictions, actual_complexities)):
            time_correct = prediction.get('time_complexity', '') == actual.get('time_complexity', '')
            space_correct = prediction.get('space_complexity', '') == actual.get('space_complexity', '')
            
            if time_correct:
                correct_time += 1
            if space_correct:
                correct_space += 1
            
            metrics['detailed_comparisons'].append({
                'test_case': actual.get('name', f'Test {i+1}'),
                'actual_time': actual.get('time_complexity'),
                'predicted_time': prediction.get('time_complexity'),
                'time_correct': time_correct,
                'actual_space': actual.get('space_complexity'),
                'predicted_space': prediction.get('space_complexity'),
                'space_correct': space_correct
            })
        
        metrics['time_complexity_accuracy'] = correct_time / total_tests if total_tests > 0 else 0
        metrics['space_complexity_accuracy'] = correct_space / total_tests if total_tests > 0 else 0
        metrics['overall_accuracy'] = (correct_time + correct_space) / (2 * total_tests) if total_tests > 0 else 0
        
        return metrics 