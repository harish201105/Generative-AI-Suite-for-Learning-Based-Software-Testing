from typing import Dict, Any, List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import json
from collections import Counter

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'test_case_count': [],
            'complexity_score': [],
            'model_name': [],
            'input_types': [],
            'operation_coverage': [],
            'test_case_types': []
        }
        
    def analyze_model_performance(self, model_name: str, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Analyze performance metrics for a single model run."""
        end_time = time.time()
        response_time = end_time - start_time
        
        try:
            # Handle both string and already-parsed test_cases
            if isinstance(result.get("test_cases"), str):
                test_cases_data = json.loads(result["test_cases"])
            else:
                test_cases_data = result.get("test_cases", [])
            
            # Get test case count
            if isinstance(test_cases_data, list):
                test_case_count = len(test_cases_data)
                test_cases = {"test_cases": test_cases_data}
            elif isinstance(test_cases_data, dict):
                test_case_count = len(test_cases_data.get("test_cases", []))
                test_cases = test_cases_data
            else:
                test_case_count = 0
                test_cases = {"test_cases": []}
            
            # Calculate complexity score (1-10)
            complexity_score = self._calculate_complexity_score(test_cases)
            
            # Analyze input types
            input_types = self._analyze_input_types(test_cases)
            
            # Analyze operation coverage
            operation_coverage = self._analyze_operation_coverage(test_cases)
            
            # Analyze test case types
            test_case_types = self._analyze_test_case_types(test_cases)
            
            # Store metrics
            self.metrics['response_time'].append(response_time)
            self.metrics['test_case_count'].append(test_case_count)
            self.metrics['complexity_score'].append(complexity_score)
            self.metrics['model_name'].append(model_name)
            self.metrics['input_types'].append(input_types)
            self.metrics['operation_coverage'].append(operation_coverage)
            self.metrics['test_case_types'].append(test_case_types)
            
            return {
                'response_time': response_time,
                'test_case_count': test_case_count,
                'complexity_score': complexity_score,
                'input_types': input_types,
                'operation_coverage': operation_coverage,
                'test_case_types': test_case_types
            }
            
        except Exception as e:
            return {
                'response_time': response_time,
                'test_case_count': 0,
                'complexity_score': 0,
                'input_types': {},
                'operation_coverage': {},
                'test_case_types': {}
            }
    
    def _calculate_complexity_score(self, test_cases: Dict[str, Any]) -> float:
        """Calculate a complexity score based on test case characteristics."""
        score = 0
        cases = test_cases.get("test_cases", [])
        
        # Base score for number of test cases
        score += min(len(cases) / 10, 1) * 4  # Up to 4 points for test case count
        
        # Points for coverage areas
        coverage_areas = test_cases.get("coverage_areas", [])
        score += min(len(coverage_areas) / 5, 1) * 3  # Up to 3 points for coverage
        
        # Points for test framework
        if test_cases.get("test_framework"):
            score += 1  # 1 point for having a framework
        
        # Points for test case quality
        for case in cases:
            if case.get("description") and case.get("input") and case.get("expected_output"):
                score += 0.2  # 0.2 points per complete test case
        
        return min(score, 10)  # Cap at 10
    
    def _analyze_input_types(self, test_cases: Dict[str, Any]) -> Dict[str, int]:
        """Analyze the distribution of input types in test cases."""
        input_types = Counter()
        for case in test_cases.get("test_cases", []):
            input_data = case.get("input", "")
            if isinstance(input_data, (int, float)):
                input_types["Numeric"] += 1
            elif isinstance(input_data, str):
                if input_data.startswith("[") or input_data.startswith("{"):
                    input_types["Array/Object"] += 1
                else:
                    input_types["String"] += 1
            elif isinstance(input_data, (list, tuple)):
                input_types["Array"] += 1
            elif isinstance(input_data, dict):
                input_types["Object"] += 1
            elif input_data is None:
                input_types["Null"] += 1
            else:
                input_types["Other"] += 1
        return dict(input_types)
    
    def _analyze_operation_coverage(self, test_cases: Dict[str, Any]) -> Dict[str, int]:
        """Analyze the coverage of different operations in test cases."""
        operations = Counter()
        for case in test_cases.get("test_cases", []):
            test_code = case.get("test_code", "").lower()
            if "assert" in test_code:
                operations["Assertions"] += 1
            if "mock" in test_code or "stub" in test_code:
                operations["Mocking"] += 1
            if "setup" in test_code or "teardown" in test_code:
                operations["Setup/Teardown"] += 1
            if "exception" in test_code or "error" in test_code:
                operations["Error Handling"] += 1
            if "loop" in test_code or "iterate" in test_code:
                operations["Iteration"] += 1
            if "async" in test_code or "await" in test_code:
                operations["Async Operations"] += 1
        return dict(operations)
    
    def _analyze_test_case_types(self, test_cases: Dict[str, Any]) -> Dict[str, int]:
        """Analyze the distribution of test case types."""
        test_types = Counter()
        for case in test_cases.get("test_cases", []):
            name = case.get("name", "").lower()
            if "happy" in name or "success" in name:
                test_types["Happy Path"] += 1
            elif "edge" in name or "boundary" in name:
                test_types["Edge Cases"] += 1
            elif "error" in name or "exception" in name:
                test_types["Error Cases"] += 1
            elif "invalid" in name or "wrong" in name:
                test_types["Invalid Input"] += 1
            elif "performance" in name or "load" in name:
                test_types["Performance"] += 1
            else:
                test_types["Other"] += 1
        return dict(test_types)
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison data for visualization."""
        df = pd.DataFrame(self.metrics)
        
        # Check if there's any data to analyze
        if df.empty:
            return {
                'dataframe': df,
                'summary': {
                    'response_time': {},
                    'test_case_count': {},
                    'complexity_score': {}
                },
                'rankings': {
                    'best_model': 'No models available',
                    'worst_model': 'No models available',
                    'rankings': {},
                    'detailed_metrics': {}
                }
            }
        
        # Calculate model rankings
        rankings = self._calculate_model_rankings(df)
        
        return {
            'dataframe': df,
            'summary': {
                'response_time': df.groupby('model_name')['response_time'].mean().to_dict(),
                'test_case_count': df.groupby('model_name')['test_case_count'].mean().to_dict(),
                'complexity_score': df.groupby('model_name')['complexity_score'].mean().to_dict()
            },
            'rankings': rankings
        }
    
    def _calculate_model_rankings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate model rankings based on performance metrics."""
        # Group by model and calculate mean metrics
        model_metrics = df.groupby('model_name').agg({
            'response_time': 'mean',
            'test_case_count': 'mean',
            'complexity_score': 'mean'
        })
        
        # Normalize metrics (lower response time is better)
        normalized_metrics = pd.DataFrame()
        normalized_metrics['response_time'] = 1 - (model_metrics['response_time'] - model_metrics['response_time'].min()) / (model_metrics['response_time'].max() - model_metrics['response_time'].min())
        normalized_metrics['test_case_count'] = (model_metrics['test_case_count'] - model_metrics['test_case_count'].min()) / (model_metrics['test_case_count'].max() - model_metrics['test_case_count'].min())
        normalized_metrics['complexity_score'] = (model_metrics['complexity_score'] - model_metrics['complexity_score'].min()) / (model_metrics['complexity_score'].max() - model_metrics['complexity_score'].min())
        
        # Calculate overall score (weighted average)
        weights = {
            'response_time': 0.3,  # 30% weight
            'test_case_count': 0.3,  # 30% weight
            'complexity_score': 0.4  # 40% weight
        }
        
        normalized_metrics['overall_score'] = (
            normalized_metrics['response_time'] * weights['response_time'] +
            normalized_metrics['test_case_count'] * weights['test_case_count'] +
            normalized_metrics['complexity_score'] * weights['complexity_score']
        )
        
        # Sort models by overall score
        ranked_models = normalized_metrics.sort_values('overall_score', ascending=False)
        
        return {
            'best_model': ranked_models.index[0],
            'worst_model': ranked_models.index[-1],
            'rankings': ranked_models['overall_score'].to_dict(),
            'detailed_metrics': {
                model: {
                    'response_time': model_metrics.loc[model, 'response_time'],
                    'test_case_count': model_metrics.loc[model, 'test_case_count'],
                    'complexity_score': model_metrics.loc[model, 'complexity_score'],
                    'overall_score': ranked_models.loc[model, 'overall_score']
                }
                for model in ranked_models.index
            }
        }
    
    def plot_performance_comparison(self) -> Dict[str, plt.Figure]:
        """Generate performance comparison plots."""
        df = pd.DataFrame(self.metrics)
        plots = {}
        
        # Response Time Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='model_name', y='response_time')
        plt.title('Model Response Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['response_time'] = plt.gcf()
        plt.close()
        
        # Test Case Count Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='model_name', y='test_case_count')
        plt.title('Test Case Count Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['test_case_count'] = plt.gcf()
        plt.close()
        
        # Complexity Score Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='model_name', y='complexity_score')
        plt.title('Test Case Complexity Score Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['complexity_score'] = plt.gcf()
        plt.close()
        
        # Input Type Distribution
        plt.figure(figsize=(12, 6))
        input_types_df = pd.DataFrame([
            {'model': model, 'type': type_, 'count': count}
            for model, types in zip(df['model_name'], df['input_types'])
            for type_, count in types.items()
        ])
        sns.barplot(data=input_types_df, x='model', y='count', hue='type')
        plt.title('Input Type Distribution by Model')
        plt.xticks(rotation=45)
        plt.legend(title='Input Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plots['input_types'] = plt.gcf()
        plt.close()
        
        # Operation Coverage
        plt.figure(figsize=(12, 6))
        operation_df = pd.DataFrame([
            {'model': model, 'operation': op, 'count': count}
            for model, ops in zip(df['model_name'], df['operation_coverage'])
            for op, count in ops.items()
        ])
        sns.barplot(data=operation_df, x='model', y='count', hue='operation')
        plt.title('Operation Coverage by Model')
        plt.xticks(rotation=45)
        plt.legend(title='Operation', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plots['operation_coverage'] = plt.gcf()
        plt.close()
        
        # Test Case Type Distribution (Pie Chart)
        plt.figure(figsize=(10, 10))
        test_types_df = pd.DataFrame([
            {'model': model, 'type': type_, 'count': count}
            for model, types in zip(df['model_name'], df['test_case_types'])
            for type_, count in types.items()
        ])
        
        # Create a pie chart for each model
        n_models = len(df['model_name'].unique())
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, model in enumerate(df['model_name'].unique()):
            model_data = test_types_df[test_types_df['model'] == model]
            axes[idx].pie(model_data['count'], labels=model_data['type'], autopct='%1.1f%%')
            axes[idx].set_title(f'Test Case Types - {model}')
        
        # Hide unused subplots
        for idx in range(len(df['model_name'].unique()), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plots['test_case_types'] = plt.gcf()
        plt.close()
        
        # Combined Performance Heatmap
        plt.figure(figsize=(12, 8))
        metrics = ['response_time', 'test_case_count', 'complexity_score']
        pivot_df = df.pivot_table(
            values=metrics,
            index='model_name',
            aggfunc='mean'
        )
        # Normalize the data
        pivot_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min())
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Model Performance Comparison (Normalized)')
        plt.tight_layout()
        plots['heatmap'] = plt.gcf()
        plt.close()
        
        # Overall Score Comparison
        rankings = self._calculate_model_rankings(df)
        plt.figure(figsize=(10, 6))
        overall_scores = pd.Series(rankings['rankings'])
        sns.barplot(x=overall_scores.index, y=overall_scores.values)
        plt.title('Model Overall Performance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['overall_score'] = plt.gcf()
        plt.close()
        
        return plots 