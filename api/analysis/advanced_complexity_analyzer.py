from typing import Dict, Any, List, Tuple, Set, Optional
import re
import ast
import json
import math
from collections import defaultdict

class AdvancedPythonComplexityAnalyzer:
    """
    Advanced complexity analyzer specifically designed for Python code.
    Provides accurate time and space complexity analysis through sophisticated AST analysis.
    """
    
    def __init__(self):
        # Python built-in functions and their complexities
        self.builtin_complexities = {
            # Sorting operations
            'sorted': ('O(n log n)', 'O(n)'),
            'sort': ('O(n log n)', 'O(1)'),  # in-place sort
            
            # Search operations
            'min': ('O(n)', 'O(1)'),
            'max': ('O(n)', 'O(1)'),
            'sum': ('O(n)', 'O(1)'),
            'any': ('O(n)', 'O(1)'),
            'all': ('O(n)', 'O(1)'),
            
            # List operations
            'len': ('O(1)', 'O(1)'),
            'append': ('O(1)', 'O(1)'),
            'insert': ('O(n)', 'O(1)'),
            'remove': ('O(n)', 'O(1)'),
            'pop': ('O(1)', 'O(1)'),  # pop from end
            'index': ('O(n)', 'O(1)'),
            'count': ('O(n)', 'O(1)'),
            'reverse': ('O(n)', 'O(1)'),
            'copy': ('O(n)', 'O(n)'),
            'extend': ('O(k)', 'O(k)'),  # k = length of iterable
            'clear': ('O(1)', 'O(1)'),
            
            # String operations
            'join': ('O(n)', 'O(n)'),
            'split': ('O(n)', 'O(n)'),
            'replace': ('O(n)', 'O(n)'),
            'find': ('O(n)', 'O(1)'),
            'startswith': ('O(k)', 'O(1)'),  # k = length of prefix
            'endswith': ('O(k)', 'O(1)'),   # k = length of suffix
            'strip': ('O(n)', 'O(n)'),
            'lower': ('O(n)', 'O(n)'),
            'upper': ('O(n)', 'O(n)'),
            'isdigit': ('O(n)', 'O(1)'),
            'isalpha': ('O(n)', 'O(1)'),
            
            # Dictionary operations
            'get': ('O(1)', 'O(1)'),
            'keys': ('O(n)', 'O(n)'),
            'values': ('O(n)', 'O(n)'),
            'items': ('O(n)', 'O(n)'),
            'pop': ('O(1)', 'O(1)'),  # dict.pop
            'update': ('O(k)', 'O(k)'),  # k = size of other dict
            'clear': ('O(1)', 'O(1)'),   # dict.clear
            'copy': ('O(n)', 'O(n)'),    # dict.copy
            'setdefault': ('O(1)', 'O(1)'),
            
            # Set operations
            'add': ('O(1)', 'O(1)'),
            'discard': ('O(1)', 'O(1)'),
            'remove': ('O(1)', 'O(1)'),  # set.remove
            'union': ('O(n+m)', 'O(n+m)'),
            'intersection': ('O(min(n,m))', 'O(min(n,m))'),
            'difference': ('O(n)', 'O(n)'),
            'symmetric_difference': ('O(n+m)', 'O(n+m)'),
            'issubset': ('O(n)', 'O(1)'),
            'issuperset': ('O(m)', 'O(1)'),
            'set': ('O(n)', 'O(n)'),  # set() constructor
            
            # Type conversion operations
            'list': ('O(n)', 'O(n)'),
            'tuple': ('O(n)', 'O(n)'),
            'dict': ('O(n)', 'O(n)'),
            'str': ('O(n)', 'O(n)'),
            'int': ('O(1)', 'O(1)'),
            'float': ('O(1)', 'O(1)'),
            'bool': ('O(1)', 'O(1)'),
            
            # Iteration and enumeration
            'enumerate': ('O(n)', 'O(1)'),  # generator, lazy
            'zip': ('O(min(n,m))', 'O(1)'),  # generator, lazy
            'map': ('O(n)', 'O(1)'),         # generator, lazy
            'filter': ('O(n)', 'O(1)'),     # generator, lazy
            'range': ('O(1)', 'O(1)'),      # generator, lazy
            
            # Heap operations (heapq module)
            'heappush': ('O(log n)', 'O(1)'),
            'heappop': ('O(log n)', 'O(1)'),
            'heapify': ('O(n)', 'O(1)'),
            'heapreplace': ('O(log n)', 'O(1)'),
            'nlargest': ('O(n log k)', 'O(k)'),
            'nsmallest': ('O(n log k)', 'O(k)'),
            
            # Mathematical operations
            'abs': ('O(1)', 'O(1)'),
            'round': ('O(1)', 'O(1)'),
            'pow': ('O(log n)', 'O(1)'),
            'divmod': ('O(1)', 'O(1)'),
            'gcd': ('O(log(min(a,b)))', 'O(1)'),  # math.gcd
            'factorial': ('O(n)', 'O(1)'),        # math.factorial
            'sqrt': ('O(1)', 'O(1)'),             # math.sqrt
            'log': ('O(1)', 'O(1)'),              # math.log
            
            # Collections module
            'defaultdict': ('O(1)', 'O(1)'),      # access
            'Counter': ('O(n)', 'O(n)'),          # constructor
            'deque': ('O(1)', 'O(1)'),            # constructor
            'appendleft': ('O(1)', 'O(1)'),       # deque operation
            'popleft': ('O(1)', 'O(1)'),          # deque operation
            
            # Regular expressions (re module)
            'match': ('O(n)', 'O(1)'),            # re.match
            'search': ('O(n)', 'O(1)'),           # re.search
            'findall': ('O(n)', 'O(m)'),          # re.findall, m = number of matches
            'sub': ('O(n)', 'O(n)'),              # re.sub
            'split': ('O(n)', 'O(m)'),            # re.split, m = number of splits
        }
        
        # Common algorithmic patterns
        self.algorithm_patterns = {
            'binary_search': ('O(log n)', 'O(1)'),
            'linear_search': ('O(n)', 'O(1)'),
            'bubble_sort': ('O(n²)', 'O(1)'),
            'selection_sort': ('O(n²)', 'O(1)'),
            'insertion_sort': ('O(n²)', 'O(1)'),
            'merge_sort': ('O(n log n)', 'O(n)'),
            'quick_sort': ('O(n log n)', 'O(log n)'),
            'heap_sort': ('O(n log n)', 'O(1)'),
            'counting_sort': ('O(n+k)', 'O(k)'),
            'radix_sort': ('O(d*(n+k))', 'O(n+k)'),
            'fibonacci_naive': ('O(2^n)', 'O(n)'),
            'fibonacci_dp': ('O(n)', 'O(n)'),
            'factorial': ('O(n)', 'O(n)'),
            'dfs': ('O(V+E)', 'O(V)'),
            'bfs': ('O(V+E)', 'O(V)'),
            'dijkstra': ('O((V+E)log V)', 'O(V)'),
        }
        
        # Complexity ordering for comparison
        self.complexity_order = [
            'O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 
            'O(n²)', 'O(n³)', 'O(2^n)', 'O(n!)'
        ]

    def analyze_code_complexity(self, code: str, function_name: str = None) -> Tuple[str, str]:
        """
        Main method to analyze Python code complexity.
        Returns (time_complexity, space_complexity)
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # If function_name is provided, analyze only that function
            if function_name:
                function_node = self._find_function(tree, function_name)
                if function_node:
                    tree = function_node
            
            # Perform comprehensive analysis
            analysis_results = self._comprehensive_analysis(tree, code)
            
            # Determine final complexities with multi-case analysis
            complexity_analysis = self._determine_multi_case_complexity(analysis_results, code)
            
            # For backward compatibility, return worst-case scenario
            time_complexity = complexity_analysis.get('worst_time', 'O(1)')
            space_complexity = complexity_analysis.get('worst_space', 'O(1)')
            
            return time_complexity, space_complexity
            
        except Exception as e:
            print(f"Error analyzing code: {e}")
            return 'O(1)', 'O(1)'

    def analyze_code_complexity_detailed(self, code: str, function_name: str = None) -> Dict[str, str]:
        """
        Enhanced method that returns best/average/worst case analysis.
        Returns dict with best_time, avg_time, worst_time, best_space, avg_space, worst_space
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # If function_name is provided, analyze only that function
            if function_name:
                function_node = self._find_function(tree, function_name)
                if function_node:
                    tree = function_node
            
            # Perform comprehensive analysis
            analysis_results = self._comprehensive_analysis(tree, code)
            
            # Determine multi-case complexities
            return self._determine_multi_case_complexity(analysis_results, code)
            
        except Exception as e:
            print(f"Error analyzing code: {e}")
            return {
                'best_time': 'O(1)', 'avg_time': 'O(1)', 'worst_time': 'O(1)',
                'best_space': 'O(1)', 'avg_space': 'O(1)', 'worst_space': 'O(1)'
            }

    def _find_function(self, tree: ast.AST, function_name: str) -> Optional[ast.FunctionDef]:
        """Find a specific function in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None

    def _comprehensive_analysis(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Perform comprehensive complexity analysis."""
        results = {
            'loops': self._analyze_loops_advanced(tree),
            'function_calls': self._analyze_function_calls(tree),
            'recursion': self._analyze_recursion_advanced(tree),
            'data_structures': self._analyze_data_structures_advanced(tree),
            'list_comprehensions': self._analyze_list_comprehensions(tree),
            'algorithmic_patterns': self._detect_algorithmic_patterns(code, tree),
            'mathematical_operations': self._analyze_mathematical_operations(tree),
            'memory_allocation': self._analyze_memory_allocation(tree),
            'control_flow': self._analyze_control_flow(tree)
        }
        return results

    def _analyze_loops_advanced(self, tree: ast.AST) -> Dict[str, Any]:
        """Advanced loop analysis with nesting and dependency detection."""
        loop_info = {
            'total_loops': 0,
            'max_nesting': 0,
            'nested_structures': [],
            'loop_types': [],
            'loop_ranges': [],
            'loop_dependencies': []
        }
        
        # Find all loops and calculate proper nesting
        all_loops = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                all_loops.append(node)
        
        loop_info['total_loops'] = len(all_loops)
        
        # Calculate proper nesting levels
        nesting_levels = []
        for loop in all_loops:
            level = self._calculate_proper_nesting_level(loop, all_loops, tree)
            nesting_levels.append(level)
            loop_info['max_nesting'] = max(loop_info['max_nesting'], level)
            
            # Analyze individual loop
            loop_analysis = self._analyze_single_loop_advanced(loop)
            loop_info['loop_types'].append(loop_analysis['type'])
            loop_info['loop_ranges'].append(loop_analysis['range'])
            
            if level > 1:
                loop_info['nested_structures'].append({
                    'nesting_level': level,
                    'type': loop_analysis['type'],
                    'range': loop_analysis['range']
                })
        
        return loop_info

    def _calculate_proper_nesting_level(self, target_loop: ast.AST, all_loops: List[ast.AST], tree: ast.AST) -> int:
        """Calculate the proper nesting level by checking if target_loop is inside other loops."""
        nesting_level = 1
        
        for other_loop in all_loops:
            if other_loop is target_loop:
                continue
            
            # Check if target_loop is nested inside other_loop
            if self._is_nested_inside(target_loop, other_loop):
                nesting_level += 1
        
        return nesting_level

    def _is_nested_inside(self, inner_node: ast.AST, outer_node: ast.AST) -> bool:
        """Check if inner_node is nested inside outer_node."""
        for child in ast.walk(outer_node):
            if child is inner_node:
                return True
        return False

    def _analyze_single_loop_advanced(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze a single loop in detail."""
        if isinstance(node, ast.For):
            # Analyze the iterator
            if isinstance(node.iter, ast.Call):
                if hasattr(node.iter.func, 'id'):
                    func_name = node.iter.func.id
                    if func_name == 'range':
                        return self._analyze_range_loop(node.iter)
                    elif func_name == 'enumerate':
                        return {'type': 'enumerate', 'range': 'O(n)'}
                    elif func_name == 'zip':
                        return {'type': 'zip', 'range': 'O(min(n,m))'}
            
            # Check if iterating over a data structure
            return {'type': 'iteration', 'range': 'O(n)'}
        
        elif isinstance(node, ast.While):
            # Analyze while loop condition
            return self._analyze_while_loop(node)
        
        return {'type': 'unknown', 'range': 'O(n)'}

    def _analyze_range_loop(self, range_call: ast.Call) -> Dict[str, Any]:
        """Analyze range() function calls to determine loop complexity."""
        args = range_call.args
        
        if len(args) == 1:
            # range(n)
            return {'type': 'range_n', 'range': 'O(n)'}
        elif len(args) == 2:
            # range(start, stop)
            return {'type': 'range_start_stop', 'range': 'O(n)'}
        elif len(args) == 3:
            # range(start, stop, step)
            return {'type': 'range_step', 'range': 'O(n)'}
        
        return {'type': 'range', 'range': 'O(n)'}

    def _analyze_while_loop(self, node: ast.While) -> Dict[str, Any]:
        """Analyze while loop to estimate complexity."""
        # Convert AST to string for pattern analysis
        try:
            condition_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
        except:
            condition_str = str(node.test)
        
        condition_lower = condition_str.lower()
        
        # Check for binary search pattern
        binary_search_indicators = [
            'left' in condition_lower and 'right' in condition_lower,
            'low' in condition_lower and 'high' in condition_lower,
            '<=' in condition_str or '>=' in condition_str
        ]
        
        # Look for divide-by-2 patterns in the loop body
        has_divide_by_two = False
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.FloorDiv):
                if isinstance(child.right, ast.Constant) and child.right.value == 2:
                    has_divide_by_two = True
                    break
            elif isinstance(child, ast.BinOp) and isinstance(child.op, ast.Add):
                # Check for mid = (left + right) // 2 pattern
                try:
                    node_str = ast.unparse(child) if hasattr(ast, 'unparse') else str(child)
                    if '//2' in node_str or '/ 2' in node_str:
                        has_divide_by_two = True
                        break
                except:
                    pass
        
        if any(binary_search_indicators) and has_divide_by_two:
            return {'type': 'binary_search_while', 'range': 'O(log n)'}
        
        # Check for decrementing patterns
        if '//' in condition_str or '/2' in condition_str or '//2' in condition_str:
            return {'type': 'divide_while', 'range': 'O(log n)'}
        
        return {'type': 'general_while', 'range': 'O(n)'}

    def _analyze_function_calls(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze function calls and their complexities."""
        call_info = {
            'builtin_calls': [],
            'user_function_calls': [],
            'method_calls': [],
            'total_complexity': 'O(1)'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_analysis = self._analyze_single_call(node)
                
                if call_analysis['type'] == 'builtin':
                    call_info['builtin_calls'].append(call_analysis)
                elif call_analysis['type'] == 'method':
                    call_info['method_calls'].append(call_analysis)
                else:
                    call_info['user_function_calls'].append(call_analysis)
                
                # Update total complexity
                call_complexity = call_analysis.get('time_complexity', 'O(1)')
                call_info['total_complexity'] = self._get_worst_complexity(
                    call_info['total_complexity'], call_complexity
                )
        
        return call_info

    def _analyze_single_call(self, node: ast.Call) -> Dict[str, Any]:
        """Analyze a single function call."""
        if isinstance(node.func, ast.Name):
            # Direct function call
            func_name = node.func.id
            if func_name in self.builtin_complexities:
                time_comp, space_comp = self.builtin_complexities[func_name]
                return {
                    'type': 'builtin',
                    'name': func_name,
                    'time_complexity': time_comp,
                    'space_complexity': space_comp
                }
            else:
                return {
                    'type': 'user_function',
                    'name': func_name,
                    'time_complexity': 'O(1)',  # Default assumption
                    'space_complexity': 'O(1)'
                }
        
        elif isinstance(node.func, ast.Attribute):
            # Method call
            method_name = node.func.attr
            if method_name in self.builtin_complexities:
                time_comp, space_comp = self.builtin_complexities[method_name]
                return {
                    'type': 'method',
                    'name': method_name,
                    'time_complexity': time_comp,
                    'space_complexity': space_comp
                }
            else:
                return {
                    'type': 'method',
                    'name': method_name,
                    'time_complexity': 'O(1)',
                    'space_complexity': 'O(1)'
                }
        
        return {
            'type': 'unknown',
            'name': 'unknown',
            'time_complexity': 'O(1)',
            'space_complexity': 'O(1)'
        }

    def _analyze_recursion_advanced(self, tree: ast.AST) -> Dict[str, Any]:
        """Advanced recursion analysis."""
        recursion_info = {
            'has_recursion': False,
            'recursion_type': None,
            'recursive_calls': [],
            'recursion_depth': 'O(1)',
            'memoization': False
        }
        
        # Find function definitions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            recursive_calls = self._find_recursive_calls(func)
            if recursive_calls:
                recursion_info['has_recursion'] = True
                recursion_info['recursive_calls'].extend(recursive_calls)
                
                # Analyze recursion pattern
                pattern = self._classify_recursion_pattern(func, recursive_calls)
                recursion_info['recursion_type'] = pattern
                
                # Check for memoization
                if self._has_memoization(func):
                    recursion_info['memoization'] = True
        
        return recursion_info

    def _find_recursive_calls(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Find recursive calls within a function."""
        recursive_calls = []
        func_name = func_node.name
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == func_name:
                    recursive_calls.append({
                        'function': func_name,
                        'args': len(node.args) if node.args else 0
                    })
        
        return recursive_calls

    def _classify_recursion_pattern(self, func_node: ast.FunctionDef, recursive_calls: List[Dict]) -> str:
        """Classify the recursion pattern."""
        # Check for dynamic programming first
        if self._is_dynamic_programming(func_node, ""):
            return 'dynamic_programming'
        
        # Simple heuristics for common patterns
        if len(recursive_calls) == 1:
            return 'linear_recursion'
        elif len(recursive_calls) == 2:
            # Check for divide-and-conquer or fibonacci-like
            func_body = ast.unparse(func_node) if hasattr(ast, 'unparse') else str(func_node)
            func_body_lower = func_body.lower()
            
            # Check for quicksort pattern
            if ('pivot' in func_body_lower or 'partition' in func_body_lower) and func_node.name in ['quick_sort', 'quicksort']:
                return 'divide_and_conquer'
            # Check for mergesort pattern
            elif 'merge' in func_body_lower and func_node.name in ['merge_sort', 'mergesort']:
                return 'divide_and_conquer'
            # Check for general divide-and-conquer patterns
            elif any(keyword in func_body_lower for keyword in ['//2', '/2', 'middle', 'mid', 'pivot', 'partition']):
                return 'divide_and_conquer'
            else:
                return 'tree_recursion'
        else:
            return 'complex_recursion'

    def _has_memoization(self, func_node: ast.FunctionDef) -> bool:
        """Check if function uses memoization or dynamic programming."""
        func_str = ast.unparse(func_node) if hasattr(ast, 'unparse') else str(func_node)
        func_str_lower = func_str.lower()
        
        # Explicit memoization decorators
        if any(decorator in func_str for decorator in ['@lru_cache', '@cache', '@memoize']):
            return True
        
        # DP array patterns
        dp_patterns = [
            'dp = [', 'dp=[', 'memo = {', 'memo={', 'cache = {', 'cache={',
            'dp[i]', 'memo[', 'cache[', 'table = [', 'table=[',
            '[0] * (n', '[0]*(n', 'dp[i-1]', 'dp[i+1]'
        ]
        
        # Check for DP array creation and access patterns
        dp_creation = any(pattern in func_str for pattern in dp_patterns)
        
        # Check for iterative DP (bottom-up approach)
        iterative_dp = (
            'for i in range(' in func_str and 
            ('dp[i]' in func_str or 'table[i]' in func_str) and
            not any(recursion_indicator in func_str for recursion_indicator in [func_node.name + '('])
        )
        
        return dp_creation or iterative_dp

    def _is_dynamic_programming(self, func_node: ast.FunctionDef, code: str) -> bool:
        """Detect if this is a dynamic programming algorithm."""
        func_name = func_node.name.lower()
        func_str = ast.unparse(func_node) if hasattr(ast, 'unparse') else str(func_node)
        
        # Common DP function names
        dp_names = ['fibonacci_dp', 'fib_dp', 'dp_', '_dp', 'memoized']
        if any(name in func_name for name in dp_names):
            return True
        
        # Check for DP characteristics
        has_array_initialization = any(pattern in func_str for pattern in ['[0] * ', '[[0]', 'dp = [', 'table = ['])
        has_iterative_filling = 'for i in range(' in func_str and ('dp[i]' in func_str or 'table[i]' in func_str)
        has_base_case = ('dp[0]' in func_str or 'dp[1]' in func_str or 'table[0]' in func_str)
        
        return has_array_initialization and has_iterative_filling and has_base_case

    def _analyze_data_structures_advanced(self, tree: ast.AST) -> Dict[str, Any]:
        """Advanced data structure analysis."""
        ds_info = {
            'lists': {'created': 0, 'operations': []},
            'dicts': {'created': 0, 'operations': []},
            'sets': {'created': 0, 'operations': []},
            'tuples': {'created': 0, 'operations': []},
            'strings': {'created': 0, 'operations': []},
            'matrices': {'created': 0, 'dimensions': []},
            'total_space': 'O(1)'
        }
        
        # Convert AST to string for pattern analysis
        tree_str = ast.unparse(tree) if hasattr(ast, 'unparse') else str(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                ds_info['lists']['created'] += 1
                # Check for matrix patterns like [[0] * n for _ in range(n)]
                if self._is_matrix_creation(node, tree_str):
                    ds_info['matrices']['created'] += 1
                    ds_info['total_space'] = self._get_worst_complexity(ds_info['total_space'], 'O(n²)')
                else:
                    ds_info['total_space'] = self._get_worst_complexity(ds_info['total_space'], 'O(n)')
            elif isinstance(node, ast.Dict):
                ds_info['dicts']['created'] += 1
                ds_info['total_space'] = self._get_worst_complexity(ds_info['total_space'], 'O(n)')
            elif isinstance(node, ast.Set):
                ds_info['sets']['created'] += 1
                ds_info['total_space'] = self._get_worst_complexity(ds_info['total_space'], 'O(n)')
            elif isinstance(node, ast.Tuple):
                ds_info['tuples']['created'] += 1
                ds_info['total_space'] = self._get_worst_complexity(ds_info['total_space'], 'O(n)')
        
        return ds_info

    def _is_matrix_creation(self, node: ast.List, tree_str: str) -> bool:
        """Detect if this is a matrix/2D array creation."""
        # Common matrix patterns
        matrix_patterns = [
            '[[0] * n for _ in range(n)]',
            '[[0] * ', ' for _ in range(',
            '[[None] * ', '[[False] * ',
            'matrix = [', 'result = [',
            '* n] for', '* len(', 'range(len('
        ]
        
        # Check for nested list comprehensions or explicit 2D patterns
        if any(pattern in tree_str for pattern in matrix_patterns):
            return True
        
        # Check if this list contains other lists (2D structure)
        for child in ast.walk(node):
            if isinstance(child, ast.List) and child is not node:
                return True
        
        return False

    def _analyze_list_comprehensions(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze list comprehensions and generator expressions."""
        comp_info = {
            'list_comps': 0,
            'dict_comps': 0,
            'set_comps': 0,
            'generators': 0,
            'nested_comps': 0,
            'total_complexity': 'O(1)'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                comp_info['list_comps'] += 1
                complexity = self._analyze_comprehension_complexity(node)
                comp_info['total_complexity'] = self._get_worst_complexity(
                    comp_info['total_complexity'], complexity
                )
            elif isinstance(node, ast.DictComp):
                comp_info['dict_comps'] += 1
                complexity = self._analyze_comprehension_complexity(node)
                comp_info['total_complexity'] = self._get_worst_complexity(
                    comp_info['total_complexity'], complexity
                )
            elif isinstance(node, ast.SetComp):
                comp_info['set_comps'] += 1
                complexity = self._analyze_comprehension_complexity(node)
                comp_info['total_complexity'] = self._get_worst_complexity(
                    comp_info['total_complexity'], complexity
                )
            elif isinstance(node, ast.GeneratorExp):
                comp_info['generators'] += 1
                complexity = self._analyze_comprehension_complexity(node)
                comp_info['total_complexity'] = self._get_worst_complexity(
                    comp_info['total_complexity'], complexity
                )
        
        return comp_info

    def _analyze_comprehension_complexity(self, node: ast.AST) -> str:
        """Analyze the complexity of a comprehension."""
        # Count the number of generators
        if hasattr(node, 'generators'):
            num_generators = len(node.generators)
            if num_generators == 1:
                return 'O(n)'
            elif num_generators == 2:
                return 'O(n²)'
            else:
                return f'O(n^{num_generators})'
        return 'O(n)'

    def _detect_algorithmic_patterns(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Detect common algorithmic patterns."""
        pattern_info = {
            'detected_algorithms': [],
            'pattern_complexity': 'O(1)'
        }
        
        code_lower = code.lower()
        
        # Binary search detection
        if self._is_binary_search(code_lower, tree):
            pattern_info['detected_algorithms'].append('binary_search')
            pattern_info['pattern_complexity'] = 'O(log n)'
        
        # Sorting algorithm detection
        sort_pattern = self._detect_sorting_algorithm(code_lower, tree)
        if sort_pattern:
            pattern_info['detected_algorithms'].append(sort_pattern)
            if sort_pattern in self.algorithm_patterns:
                complexity = self.algorithm_patterns[sort_pattern][0]
                pattern_info['pattern_complexity'] = self._get_worst_complexity(
                    pattern_info['pattern_complexity'], complexity
                )
        
        # Fibonacci detection
        if self._is_fibonacci(code_lower, tree):
            if 'memo' in code_lower or 'cache' in code_lower or 'dp' in code_lower:
                pattern_info['detected_algorithms'].append('fibonacci_dp')
                pattern_info['pattern_complexity'] = 'O(n)'
            else:
                pattern_info['detected_algorithms'].append('fibonacci_naive')
                pattern_info['pattern_complexity'] = 'O(2^n)'
        
        return pattern_info

    def _is_binary_search(self, code: str, tree: ast.AST) -> bool:
        """Detect binary search pattern."""
        binary_search_indicators = [
            'left' in code and 'right' in code and 'mid' in code,
            'low' in code and 'high' in code and 'mid' in code,
            '//2' in code or '/2' in code,
            'middle' in code
        ]
        
        # Also check for while loop with comparison
        has_while_with_comparison = False
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Compare):
                    has_while_with_comparison = True
                    break
        
        return any(binary_search_indicators) and has_while_with_comparison

    def _detect_sorting_algorithm(self, code: str, tree: ast.AST) -> Optional[str]:
        """Detect sorting algorithm patterns."""
        code_lower = code.lower()
        
        # Check for specific function names and algorithm patterns
        if 'bubble' in code_lower and 'sort' in code_lower:
            return 'bubble_sort'
        elif 'merge' in code_lower and 'sort' in code_lower:
            return 'merge_sort'
        elif ('quick' in code_lower and 'sort' in code_lower) or ('pivot' in code_lower and 'partition' in code_lower):
            return 'quick_sort'
        elif 'selection' in code_lower and 'sort' in code_lower:
            return 'selection_sort'
        elif 'insertion' in code_lower and 'sort' in code_lower:
            return 'insertion_sort'
        elif 'heap' in code_lower and 'sort' in code_lower:
            return 'heap_sort'
        
        # Additional pattern detection for quicksort
        if 'partition' in code_lower and any(func_name in code_lower for func_name in ['quick_sort', 'quicksort']):
            return 'quick_sort'
        
        return None

    def _is_fibonacci(self, code: str, tree: ast.AST) -> bool:
        """Detect Fibonacci sequence calculation."""
        return 'fibonacci' in code or 'fib' in code

    def _analyze_mathematical_operations(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze mathematical operations and their complexities."""
        math_info = {
            'arithmetic_ops': 0,
            'power_ops': 0,
            'logarithmic_ops': 0,
            'total_complexity': 'O(1)'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Pow):
                    math_info['power_ops'] += 1
                    math_info['total_complexity'] = self._get_worst_complexity(
                        math_info['total_complexity'], 'O(log n)'
                    )
                else:
                    math_info['arithmetic_ops'] += 1
        
        return math_info

    def _analyze_memory_allocation(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze memory allocation patterns."""
        memory_info = {
            'new_objects': 0,
            'large_structures': 0,
            'total_space': 'O(1)'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Dict, ast.Set)):
                memory_info['new_objects'] += 1
                memory_info['total_space'] = self._get_worst_complexity(
                    memory_info['total_space'], 'O(n)'
                )
        
        return memory_info

    def _analyze_control_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze control flow complexity with early termination detection."""
        control_info = {
            'if_statements': 0,
            'branches': 0,
            'early_returns': 0,
            'break_statements': 0,
            'continue_statements': 0,
            'has_early_termination': False,
            'complexity_reduction': False,
            'complexity': 'O(1)'
        }
        
        # Find all control flow structures
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                control_info['if_statements'] += 1
                control_info['branches'] += 1
            elif isinstance(node, ast.Return):
                # Check if return is inside a loop (early termination)
                if self._is_inside_loop(node, tree):
                    control_info['early_returns'] += 1
                    control_info['has_early_termination'] = True
            elif isinstance(node, ast.Break):
                control_info['break_statements'] += 1
                control_info['has_early_termination'] = True
            elif isinstance(node, ast.Continue):
                control_info['continue_statements'] += 1
        
        # Determine if early termination reduces complexity
        if control_info['has_early_termination']:
            control_info['complexity_reduction'] = self._analyze_complexity_reduction(tree)
        
        return control_info

    def _is_inside_loop(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is inside a loop structure."""
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.For, ast.While)):
                for child in ast.walk(parent):
                    if child is node:
                        return True
        return False

    def _analyze_complexity_reduction(self, tree: ast.AST) -> bool:
        """Analyze if early termination actually reduces complexity."""
        tree_str = ast.unparse(tree) if hasattr(ast, 'unparse') else str(tree)
        
        # Common patterns that indicate complexity reduction
        reduction_patterns = [
            'return',              # Early return in search
            'break',               # Early break in search
            'found',               # Variable indicating found condition
            'target',              # Search target variable
            '== target',           # Comparison with target
            'if arr[i] ==',        # Linear search pattern
            'if target in',        # Membership test
        ]
        
        # If we find these patterns, early termination likely reduces average-case complexity
        return any(pattern in tree_str for pattern in reduction_patterns)

    def _calculate_nesting_level(self, target_node: ast.AST, tree: ast.AST) -> int:
        """Calculate the nesting level of a node."""
        # This is the corrected version that's already handled by _calculate_proper_nesting_level
        return 1

    def _determine_time_complexity(self, analysis_results: Dict[str, Any]) -> str:
        """Determine overall time complexity from analysis results."""
        complexities = ['O(1)']
        
        # Add loop complexities with proper nesting analysis
        loop_analysis = analysis_results['loops']
        max_nesting = loop_analysis['max_nesting']
        
        if max_nesting >= 3:
            complexities.append('O(n³)')
        elif max_nesting == 2:
            complexities.append('O(n²)')
        elif max_nesting == 1:
            # Check for logarithmic patterns in while loops
            loop_types = loop_analysis['loop_types']
            if any('binary_search' in loop_type or 'divide' in loop_type for loop_type in loop_types):
                complexities.append('O(log n)')
            else:
                complexities.append('O(n)')
        
        # Add function call complexities
        func_complexity = analysis_results['function_calls']['total_complexity']
        complexities.append(func_complexity)
        
        # Add recursion complexities
        recursion_analysis = analysis_results['recursion']
        if recursion_analysis['has_recursion']:
            if recursion_analysis['recursion_type'] == 'dynamic_programming':
                complexities.append('O(n)')  # DP is typically O(n) time
            elif recursion_analysis['recursion_type'] == 'tree_recursion' and not recursion_analysis['memoization']:
                complexities.append('O(2^n)')
            elif recursion_analysis['recursion_type'] == 'divide_and_conquer':
                complexities.append('O(n log n)')
            else:
                complexities.append('O(n)')
        
        # Add comprehension complexities
        comp_complexity = analysis_results['list_comprehensions']['total_complexity']
        complexities.append(comp_complexity)
        
        # Add algorithmic pattern complexities
        pattern_complexity = analysis_results['algorithmic_patterns']['pattern_complexity']
        complexities.append(pattern_complexity)
        
        # Return the worst complexity
        return self._get_worst_complexity_from_list(complexities)

    def _determine_multi_case_complexity(self, analysis_results: Dict[str, Any], code: str) -> Dict[str, str]:
        """Determine best/average/worst case complexities with control flow analysis."""
        # Get algorithm-specific multi-case analysis
        algorithm_analysis = self._analyze_algorithm_cases(analysis_results, code)
        if algorithm_analysis:
            return algorithm_analysis
        
        # Fallback to general analysis with control flow enhancement
        worst_time = self._determine_time_complexity(analysis_results)
        worst_space = self._determine_space_complexity(analysis_results)
        
        # Check for early termination patterns
        control_flow = analysis_results.get('control_flow', {})
        has_early_termination = control_flow.get('has_early_termination', False)
        complexity_reduction = control_flow.get('complexity_reduction', False)
        
        # Adjust best/average case if early termination is detected
        if has_early_termination and complexity_reduction:
            best_time = self._reduce_complexity_for_early_termination(worst_time)
            avg_time = self._average_case_with_early_termination(worst_time, best_time)
        else:
            best_time = worst_time
            avg_time = worst_time
        
        return {
            'best_time': best_time,
            'avg_time': avg_time, 
            'worst_time': worst_time,
            'best_space': worst_space,
            'avg_space': worst_space,
            'worst_space': worst_space
        }

    def _reduce_complexity_for_early_termination(self, worst_complexity: str) -> str:
        """Reduce complexity for best case with early termination."""
        if worst_complexity == 'O(n)':
            return 'O(1)'  # Linear search can find element immediately
        elif worst_complexity == 'O(n²)':
            return 'O(n)'  # Nested loop with early break
        elif worst_complexity == 'O(n³)':
            return 'O(n²)'  # Triple nested with early break
        return worst_complexity  # No reduction for other complexities

    def _average_case_with_early_termination(self, worst_case: str, best_case: str) -> str:
        """Calculate average case complexity with early termination."""
        if worst_case == 'O(n)' and best_case == 'O(1)':
            return 'O(n)'  # Average for linear search is still O(n)
        elif worst_case == 'O(n²)' and best_case == 'O(n)':
            return 'O(n²)'  # Average for nested loops is still O(n²)
        return worst_case  # Default to worst case

    def _analyze_algorithm_cases(self, analysis_results: Dict[str, Any], code: str) -> Optional[Dict[str, str]]:
        """Analyze specific algorithms for best/average/worst cases."""
        code_lower = code.lower()
        
        # Quicksort multi-case analysis
        if 'quick' in code_lower and 'sort' in code_lower and 'pivot' in code_lower:
            return {
                'best_time': 'O(n log n)',
                'avg_time': 'O(n log n)', 
                'worst_time': 'O(n²)',  # When pivot is always worst choice
                'best_space': 'O(log n)',
                'avg_space': 'O(log n)',
                'worst_space': 'O(n)'  # When recursion stack is deepest
            }
        
        # Binary search multi-case analysis
        elif 'binary' in code_lower and 'search' in code_lower:
            return {
                'best_time': 'O(1)',      # Element found immediately
                'avg_time': 'O(log n)',
                'worst_time': 'O(log n)', # Element not found or at end
                'best_space': 'O(1)',
                'avg_space': 'O(1)',
                'worst_space': 'O(1)'
            }
        
        # Linear search multi-case analysis
        elif ('linear' in code_lower or 'find' in code_lower) and 'search' in code_lower:
            return {
                'best_time': 'O(1)',      # Element found at beginning
                'avg_time': 'O(n)',       # Element found in middle on average
                'worst_time': 'O(n)',     # Element not found or at end
                'best_space': 'O(1)',
                'avg_space': 'O(1)',
                'worst_space': 'O(1)'
            }
        
        # Bubble sort multi-case analysis
        elif 'bubble' in code_lower and 'sort' in code_lower:
            return {
                'best_time': 'O(n)',      # Already sorted with early termination
                'avg_time': 'O(n²)',
                'worst_time': 'O(n²)',    # Reverse sorted
                'best_space': 'O(1)',
                'avg_space': 'O(1)',
                'worst_space': 'O(1)'
            }
        
        # Insertion sort multi-case analysis
        elif 'insertion' in code_lower and 'sort' in code_lower:
            return {
                'best_time': 'O(n)',      # Already sorted
                'avg_time': 'O(n²)',
                'worst_time': 'O(n²)',    # Reverse sorted
                'best_space': 'O(1)',
                'avg_space': 'O(1)',
                'worst_space': 'O(1)'
            }
        
        return None

    def _determine_space_complexity(self, analysis_results: Dict[str, Any]) -> str:
        """Determine overall space complexity from analysis results."""
        complexities = ['O(1)']
        
        # Add data structure space
        ds_analysis = analysis_results['data_structures']
        if ds_analysis['lists']['created'] > 0 or ds_analysis['dicts']['created'] > 0 or ds_analysis['sets']['created'] > 0:
            complexities.append('O(n)')
        
        # Add recursion space (call stack)
        recursion_analysis = analysis_results['recursion']
        if recursion_analysis['has_recursion']:
            if recursion_analysis['recursion_type'] == 'divide_and_conquer':
                complexities.append('O(log n)')  # Divide-and-conquer typically has O(log n) stack depth
            elif recursion_analysis['recursion_type'] == 'tree_recursion' and not recursion_analysis['memoization']:
                complexities.append('O(n)')  # Tree recursion like fibonacci has O(n) depth
            else:
                complexities.append('O(n)')  # Linear recursion has O(n) stack space
        
        # Add function call space implications
        func_calls = analysis_results['function_calls']
        for call in func_calls['builtin_calls']:
            space_comp = call.get('space_complexity', 'O(1)')
            complexities.append(space_comp)
        
        # Add list comprehension space
        comp_analysis = analysis_results['list_comprehensions']
        if comp_analysis['list_comps'] > 0 or comp_analysis['dict_comps'] > 0 or comp_analysis['set_comps'] > 0:
            complexities.append('O(n)')
        
        # Add memory allocation space
        memory_space = analysis_results['memory_allocation']['total_space']
        complexities.append(memory_space)
        
        return self._get_worst_complexity_from_list(complexities)

    def _get_worst_complexity(self, comp1: str, comp2: str) -> str:
        """Get the worse of two complexities."""
        try:
            idx1 = self.complexity_order.index(comp1)
            idx2 = self.complexity_order.index(comp2)
            return self.complexity_order[max(idx1, idx2)]
        except ValueError:
            return comp1

    def _get_worst_complexity_from_list(self, complexities: List[str]) -> str:
        """Get the worst complexity from a list."""
        worst = 'O(1)'
        for comp in complexities:
            worst = self._get_worst_complexity(worst, comp)
        return worst

    def analyze_test_case(self, test_code: str) -> Tuple[str, str]:
        """Analyze a single test case - main interface method."""
        return self.analyze_code_complexity(test_code)

    def analyze_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple test cases and return comprehensive results."""
        complexities = []
        total_time_complexity = 'O(1)'
        total_space_complexity = 'O(1)'
        
        for test_case in test_cases:
            test_code = test_case.get('test_code', '')
            time_complexity, space_complexity = self.analyze_code_complexity(test_code)
            
            complexities.append({
                'name': test_case.get('name', 'Unnamed Test'),
                'time_complexity': time_complexity,
                'space_complexity': space_complexity,
                'test_code': test_code
            })
            
            # Update total complexity (take the worst case)
            total_time_complexity = self._get_worst_complexity(total_time_complexity, time_complexity)
            total_space_complexity = self._get_worst_complexity(total_space_complexity, space_complexity)
        
        return {
            'individual_complexities': complexities,
            'total_time_complexity': total_time_complexity,
            'total_space_complexity': total_space_complexity,
            'analysis_method': 'advanced_python_analysis'
        }