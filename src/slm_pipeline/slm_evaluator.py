"""
SLM Evaluator - Comprehensive evaluation framework for SLM-based fact extraction.

This module provides detailed evaluation and comparison capabilities for
SLM pipelines vs baseline approaches.
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SLMEvaluationResult:
    """Results from evaluating a single SLM query."""
    query_id: str
    query: str
    predicted_facts: List[str]
    expected_facts: List[str]
    predicted_eligible: bool
    expected_eligible: bool
    processing_time: float
    fact_precision: float
    fact_recall: float
    fact_f1: float
    exact_match: bool
    task_success: bool
    model_confidence: Optional[float] = None


class SLMEvaluator:
    """
    Comprehensive evaluation framework for SLM-based legal fact extraction.
    
    This evaluator provides:
    - Detailed fact-level metrics (precision, recall, F1)
    - End-to-end task evaluation
    - Comparison with baseline approaches
    - Performance analysis and benchmarking
    - Robustness testing on edge cases
    """
    
    def __init__(self, slm_pipeline=None, baseline_pipeline=None, legal_engine=None):
        """
        Initialize the SLM evaluator.
        
        Args:
            slm_pipeline: SLM-based pipeline for evaluation
            baseline_pipeline: Baseline pipeline for comparison
            legal_engine: Legal reasoning engine for eligibility decisions
        """
        self.slm_pipeline = slm_pipeline
        self.baseline_pipeline = baseline_pipeline
        self.legal_engine = legal_engine
        
        logger.info("SLM Evaluator initialized")
    
    def evaluate_slm_pipeline(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the SLM pipeline on test data.
        
        Args:
            test_data: List of test queries with expected results
            
        Returns:
            Comprehensive evaluation results
        """
        if self.slm_pipeline is None:
            raise ValueError("SLM pipeline not provided")
        
        logger.info(f"Evaluating SLM pipeline on {len(test_data)} queries")
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_data):
            if i % 10 == 0:
                logger.info(f"Processing query {i+1}/{len(test_data)}")
            
            start_time = time.time()
            
            try:
                # Extract facts using SLM pipeline
                predicted_facts = self.slm_pipeline.process_query(test_case['query'])
                
                # Get eligibility decision
                decision = self.legal_engine.check_eligibility(predicted_facts) if self.legal_engine else {'eligible': False}
                
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # Calculate fact-level metrics
                fact_metrics = self._calculate_fact_metrics(
                    predicted_facts, 
                    test_case['expected_facts']
                )
                
                # Check task success
                task_success = (decision['eligible'] == test_case['expected_eligible'])
                exact_match = self._check_exact_match(predicted_facts, test_case['expected_facts'])
                
                result = SLMEvaluationResult(
                    query_id=test_case['id'],
                    query=test_case['query'],
                    predicted_facts=predicted_facts,
                    expected_facts=test_case['expected_facts'],
                    predicted_eligible=decision['eligible'],
                    expected_eligible=test_case['expected_eligible'],
                    processing_time=processing_time,
                    fact_precision=fact_metrics['precision'],
                    fact_recall=fact_metrics['recall'],
                    fact_f1=fact_metrics['f1'],
                    exact_match=exact_match,
                    task_success=task_success
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                # Create error result
                error_result = SLMEvaluationResult(
                    query_id=test_case['id'],
                    query=test_case['query'],
                    predicted_facts=[],
                    expected_facts=test_case['expected_facts'],
                    predicted_eligible=False,
                    expected_eligible=test_case['expected_eligible'],
                    processing_time=0.0,
                    fact_precision=0.0,
                    fact_recall=0.0,
                    fact_f1=0.0,
                    exact_match=False,
                    task_success=False
                )
                results.append(error_result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_slm_aggregate_metrics(results)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics,
            'total_processing_time': total_time,
            'avg_processing_time': total_time / len(test_data) if test_data else 0
        }
    
    def compare_pipelines(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare SLM pipeline with baseline pipeline.
        
        Args:
            test_data: Test data for comparison
            
        Returns:
            Detailed comparison results
        """
        logger.info(f"Comparing SLM vs Baseline on {len(test_data)} queries")
        
        # Evaluate both pipelines
        slm_results = self.evaluate_slm_pipeline(test_data)
        
        baseline_results = None
        if self.baseline_pipeline:
            # Import baseline evaluator
            from ..evaluation.evaluator import HybExEvaluator
            baseline_evaluator = HybExEvaluator()
            baseline_results = baseline_evaluator.evaluate_pipeline(test_data)
        
        # Calculate comparison metrics
        comparison = self._create_comparison_analysis(slm_results, baseline_results)
        
        return comparison
    
    def _calculate_fact_metrics(self, predicted: List[str], expected: List[str]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for fact extraction."""
        # Normalize facts for comparison
        pred_normalized = [self._normalize_fact(fact) for fact in predicted]
        exp_normalized = [self._normalize_fact(fact) for fact in expected]
        
        # Calculate metrics
        pred_set = set(pred_normalized)
        exp_set = set(exp_normalized)
        
        if len(pred_set) == 0 and len(exp_set) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if len(pred_set) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(exp_set) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate intersection
        correct = len(pred_set.intersection(exp_set))
        
        precision = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = correct / len(exp_set) if len(exp_set) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _normalize_fact(self, fact: str) -> str:
        """Normalize a fact for comparison."""
        # Remove whitespace and convert to lowercase
        normalized = fact.strip().lower()
        
        # Standardize spacing around punctuation
        import re
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = re.sub(r'\s*\.\s*', '.', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _check_exact_match(self, predicted: List[str], expected: List[str]) -> bool:
        """Check if predicted facts exactly match expected facts."""
        pred_normalized = sorted([self._normalize_fact(fact) for fact in predicted])
        exp_normalized = sorted([self._normalize_fact(fact) for fact in expected])
        
        return pred_normalized == exp_normalized
    
    def _calculate_slm_aggregate_metrics(self, results: List[SLMEvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {}
        
        # Task success rate
        task_success_rate = sum(1 for r in results if r.task_success) / len(results)
        
        # Exact match rate
        exact_match_rate = sum(1 for r in results if r.exact_match) / len(results)
        
        # Average fact metrics
        avg_fact_precision = statistics.mean([r.fact_precision for r in results])
        avg_fact_recall = statistics.mean([r.fact_recall for r in results])
        avg_fact_f1 = statistics.mean([r.fact_f1 for r in results])
        
        # Processing time statistics
        avg_processing_time = statistics.mean([r.processing_time for r in results])
        min_processing_time = min([r.processing_time for r in results])
        max_processing_time = max([r.processing_time for r in results])
        
        # Facts per query statistics
        facts_per_query = [len(r.predicted_facts) for r in results]
        avg_facts_per_query = statistics.mean(facts_per_query) if facts_per_query else 0
        
        return {
            'task_success_rate': task_success_rate,
            'exact_match_rate': exact_match_rate,
            'avg_fact_precision': avg_fact_precision,
            'avg_fact_recall': avg_fact_recall,
            'avg_fact_f1': avg_fact_f1,
            'avg_processing_time': avg_processing_time,
            'min_processing_time': min_processing_time,
            'max_processing_time': max_processing_time,
            'avg_facts_per_query': avg_facts_per_query,
            'total_queries': len(results)
        }
    
    def _create_comparison_analysis(
        self, 
        slm_results: Dict[str, Any], 
        baseline_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create detailed comparison analysis."""
        comparison = {
            'slm_results': slm_results,
            'baseline_results': baseline_results,
            'comparison_metrics': {}
        }
        
        if baseline_results:
            slm_metrics = slm_results['aggregate_metrics']
            baseline_metrics = baseline_results['aggregate_metrics']
            
            # Calculate improvements
            improvements = {}
            for metric in ['task_success_rate', 'avg_fact_f1', 'avg_fact_precision', 'avg_fact_recall']:
                if metric in slm_metrics and metric in baseline_metrics:
                    slm_val = slm_metrics[metric]
                    baseline_val = baseline_metrics[metric]
                    improvement = ((slm_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                    improvements[f'{metric}_improvement_percent'] = improvement
            
            # Processing time comparison
            if 'avg_processing_time' in slm_metrics and 'avg_processing_time' in baseline_metrics:
                slm_time = slm_metrics['avg_processing_time']
                baseline_time = baseline_metrics['avg_processing_time']
                time_change = ((slm_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
                improvements['processing_time_change_percent'] = time_change
            
            comparison['comparison_metrics'] = {
                'improvements': improvements,
                'slm_better_metrics': [k for k, v in improvements.items() if v > 0 and 'time' not in k],
                'baseline_better_metrics': [k for k, v in improvements.items() if v < 0 and 'time' not in k],
                'recommendation': self._generate_recommendation(improvements)
            }
        
        return comparison
    
    def _generate_recommendation(self, improvements: Dict[str, float]) -> str:
        """Generate recommendation based on comparison results."""
        # Key metrics for recommendation
        task_improvement = improvements.get('task_success_rate_improvement_percent', 0)
        f1_improvement = improvements.get('avg_fact_f1_improvement_percent', 0)
        time_change = improvements.get('processing_time_change_percent', 0)
        
        if task_improvement > 5 and f1_improvement > 5:
            if time_change < 100:  # Less than 2x slower
                return "SLM strongly recommended - significant accuracy gains with acceptable speed"
            else:
                return "SLM recommended - major accuracy gains but slower processing"
        elif task_improvement > 0 and f1_improvement > 0:
            return "SLM moderately recommended - consistent improvements across metrics"
        elif time_change < -20:  # 20% faster
            return "Consider baseline - SLM may not justify the complexity"
        else:
            return "Mixed results - choose based on specific requirements"
    
    def generate_detailed_report(
        self, 
        comparison_results: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> str:
        """Generate a detailed evaluation report."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# HybEx-Law SLM Evaluation Report",
            "=" * 50,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # SLM Results Summary
        if 'slm_results' in comparison_results:
            slm_metrics = comparison_results['slm_results']['aggregate_metrics']
            report_lines.extend([
                "## SLM Pipeline Results",
                f"- Task Success Rate: {slm_metrics['task_success_rate']:.1%}",
                f"- Exact Match Rate: {slm_metrics['exact_match_rate']:.1%}",
                f"- Average Fact F1: {slm_metrics['avg_fact_f1']:.3f}",
                f"- Average Processing Time: {slm_metrics['avg_processing_time']:.3f}s",
                f"- Total Queries Processed: {slm_metrics['total_queries']}",
                ""
            ])
        
        # Baseline Comparison
        if 'baseline_results' in comparison_results and comparison_results['baseline_results']:
            baseline_metrics = comparison_results['baseline_results']['aggregate_metrics']
            improvements = comparison_results['comparison_metrics']['improvements']
            
            report_lines.extend([
                "## Baseline vs SLM Comparison",
                f"- Task Success Rate: {improvements.get('task_success_rate_improvement_percent', 0):+.1f}%",
                f"- Fact F1 Score: {improvements.get('avg_fact_f1_improvement_percent', 0):+.1f}%",
                f"- Processing Time: {improvements.get('processing_time_change_percent', 0):+.1f}%",
                "",
                f"**Recommendation**: {comparison_results['comparison_metrics']['recommendation']}",
                ""
            ])
        
        # Performance Analysis
        if 'slm_results' in comparison_results:
            results = comparison_results['slm_results']['individual_results']
            
            # Error analysis
            failed_queries = [r for r in results if not r.task_success]
            if failed_queries:
                report_lines.extend([
                    f"## Error Analysis ({len(failed_queries)} failures)",
                    "Failed queries:",
                ])
                for failure in failed_queries[:5]:  # Show first 5 failures
                    report_lines.append(f"- Query {failure.query_id}: {failure.query[:100]}...")
                report_lines.append("")
        
        # Generate full report
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        # Convert results to JSON-serializable format
        serializable_results = self._make_serializable(results)
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


# Utility functions for analysis
def analyze_performance_by_case_type(results: List[SLMEvaluationResult]) -> Dict[str, Dict[str, float]]:
    """Analyze performance broken down by case type."""
    case_type_results = {}
    
    for result in results:
        # Extract case type from facts
        case_type = "unknown"
        for fact in result.expected_facts:
            if "case_type" in fact:
                # Extract case type from fact
                import re
                match = re.search(r"case_type\(user,\s*'([^']+)'\)", fact)
                if match:
                    case_type = match.group(1)
                    break
        
        if case_type not in case_type_results:
            case_type_results[case_type] = []
        
        case_type_results[case_type].append(result)
    
    # Calculate metrics for each case type
    analysis = {}
    for case_type, case_results in case_type_results.items():
        if case_results:
            analysis[case_type] = {
                'count': len(case_results),
                'task_success_rate': sum(1 for r in case_results if r.task_success) / len(case_results),
                'avg_fact_f1': statistics.mean([r.fact_f1 for r in case_results]),
                'exact_match_rate': sum(1 for r in case_results if r.exact_match) / len(case_results)
            }
    
    return analysis


# Example usage
if __name__ == "__main__":
    print("🧪 SLM Evaluator Example")
    print("=" * 50)
    
    # This would normally use actual pipelines and test data
    print("Note: This example requires initialized SLM and baseline pipelines")
    print("The evaluator provides comprehensive comparison and analysis capabilities")
    
    # Example of what the evaluator can do:
    features = [
        "✅ Fact-level precision, recall, F1 metrics",
        "✅ End-to-end task success evaluation", 
        "✅ Processing time benchmarking",
        "✅ Baseline comparison analysis",
        "✅ Detailed error analysis",
        "✅ Performance by case type breakdown",
        "✅ Automated recommendation generation",
        "✅ Comprehensive reporting"
    ]
    
    print("\nEvaluator capabilities:")
    for feature in features:
        print(f"  {feature}")
    
    print("\n🎯 Ready for comprehensive SLM evaluation!")
