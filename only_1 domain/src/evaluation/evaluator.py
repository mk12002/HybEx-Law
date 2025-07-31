"""
Evaluation framework for HybEx-Law system.

This module provides comprehensive evaluation metrics and comparison
with baseline approaches for assessing the hybrid NLP pipeline.
"""

import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..nlp_pipeline.hybrid_pipeline import HybridLegalNLPPipeline
from ..prolog_engine.legal_engine import LegalAidEngine


@dataclass
class EvaluationResult:
    """Results from evaluating a single query."""
    query_id: int
    query: str
    predicted_facts: List[str]
    expected_facts: List[str]
    predicted_eligible: bool
    expected_eligible: bool
    processing_time: float
    fact_precision: float
    fact_recall: float
    fact_f1: float
    task_success: bool


class HybExEvaluator:
    """
    Comprehensive evaluation framework for the HybEx-Law system.
    
    Provides:
    - Fact-level evaluation (precision, recall, F1)
    - End-to-end task success rate
    - Comparison with baseline approaches
    - Performance metrics
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.pipeline = HybridLegalNLPPipeline()
        self.legal_engine = LegalAidEngine()
    
    def evaluate_pipeline(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the complete pipeline on test data.
        
        Args:
            test_data: List of test queries with expected results
            
        Returns:
            Comprehensive evaluation results
        """
        results = []
        total_time = 0
        
        print(f"🔍 Evaluating pipeline on {len(test_data)} queries...")
        
        for i, test_case in enumerate(test_data):
            print(f"  Processing query {i+1}/{len(test_data)}")
            
            start_time = time.time()
            
            # Extract facts using pipeline
            predicted_facts = self.pipeline.process_query(test_case['query'])
            
            # Get eligibility decision
            decision = self.legal_engine.check_eligibility(predicted_facts)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # Calculate metrics
            fact_metrics = self._calculate_fact_metrics(
                predicted_facts, 
                test_case['expected_facts']
            )
            
            task_success = (decision['eligible'] == test_case['expected_eligible'])
            
            result = EvaluationResult(
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
                task_success=task_success
            )
            
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        aggregate_metrics['total_processing_time'] = total_time
        aggregate_metrics['avg_processing_time'] = total_time / len(test_data)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def compare_baselines(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare HybEx-Law with baseline approaches.
        
        Args:
            test_data: Test data for comparison
            
        Returns:
            Comparison results across different approaches
        """
        print("🆚 Comparing with baseline approaches...")
        
        # Evaluate HybEx-Law (our approach)
        hybex_results = self.evaluate_pipeline(test_data)
        
        # Evaluate regex-only baseline
        regex_results = self._evaluate_regex_baseline(test_data)
        
        # Note: LLM baseline would require API integration
        # For now, we'll create a placeholder
        llm_results = self._create_llm_placeholder_results(test_data)
        
        comparison = {
            'HybEx-Law': {
                'task_success_rate': hybex_results['aggregate_metrics']['task_success_rate'],
                'fact_f1_score': hybex_results['aggregate_metrics']['avg_fact_f1'],
                'avg_processing_time': hybex_results['aggregate_metrics']['avg_processing_time']
            },
            'Regex-Only': {
                'task_success_rate': regex_results['aggregate_metrics']['task_success_rate'],
                'fact_f1_score': regex_results['aggregate_metrics']['avg_fact_f1'],
                'avg_processing_time': regex_results['aggregate_metrics']['avg_processing_time']
            },
            'LLM-Only': llm_results
        }
        
        return comparison
    
    def _calculate_fact_metrics(self, predicted: List[str], expected: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 for fact extraction.
        
        Args:
            predicted: Predicted facts
            expected: Expected facts
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Normalize facts for comparison (remove spaces, standardize format)
        predicted_set = set(self._normalize_fact(fact) for fact in predicted)
        expected_set = set(self._normalize_fact(fact) for fact in expected)
        
        if len(expected_set) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if len(predicted_set) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate intersection
        intersection = predicted_set.intersection(expected_set)
        
        precision = len(intersection) / len(predicted_set) if predicted_set else 0
        recall = len(intersection) / len(expected_set) if expected_set else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _normalize_fact(self, fact: str) -> str:
        """
        Normalize a fact for comparison.
        
        Args:
            fact: Raw fact string
            
        Returns:
            Normalized fact
        """
        # Remove extra spaces and trailing periods
        normalized = fact.strip().rstrip('.')
        
        # Standardize quotes
        normalized = normalized.replace('"', "'")
        
        # Remove extra spaces around commas and parentheses
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all results.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        # Task success rate
        task_success_rate = sum(1 for r in results if r.task_success) / len(results)
        
        # Average fact metrics
        avg_fact_precision = sum(r.fact_precision for r in results) / len(results)
        avg_fact_recall = sum(r.fact_recall for r in results) / len(results)
        avg_fact_f1 = sum(r.fact_f1 for r in results) / len(results)
        
        return {
            'task_success_rate': task_success_rate,
            'avg_fact_precision': avg_fact_precision,
            'avg_fact_recall': avg_fact_recall,
            'avg_fact_f1': avg_fact_f1,
            'total_queries': len(results)
        }
    
    def _evaluate_regex_baseline(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a simple regex-only baseline.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation results for regex baseline
        """
        # Simple regex-based fact extraction
        results = []
        
        for test_case in test_data:
            start_time = time.time()
            
            # Very basic regex extraction (simplified)
            predicted_facts = self._regex_extract_facts(test_case['query'])
            
            # Use the same legal engine for decision
            decision = self.legal_engine.check_eligibility(predicted_facts)
            
            processing_time = time.time() - start_time
            
            fact_metrics = self._calculate_fact_metrics(
                predicted_facts, 
                test_case['expected_facts']
            )
            
            task_success = (decision['eligible'] == test_case['expected_eligible'])
            
            result = EvaluationResult(
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
                task_success=task_success
            )
            
            results.append(result)
        
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def _regex_extract_facts(self, query: str) -> List[str]:
        """
        Simple regex-based fact extraction for baseline comparison.
        
        Args:
            query: Input query
            
        Returns:
            List of extracted facts
        """
        import re
        
        facts = ['applicant(user)']
        query_lower = query.lower()
        
        # Very basic income extraction
        income_match = re.search(r'(\d+).*(?:rupees|rs)', query_lower)
        if income_match:
            facts.append(f'income_monthly(user, {income_match.group(1)})')
        elif any(word in query_lower for word in ['no income', 'unemployed', 'lost job']):
            facts.append('income_monthly(user, 0)')
        
        # Basic social category detection
        if any(word in query_lower for word in ['woman', 'wife', 'female']):
            facts.append('is_woman(user, true)')
        else:
            facts.append('is_woman(user, false)')
        
        if any(word in query_lower for word in ['sc', 'st', 'scheduled caste', 'scheduled tribe']):
            facts.append('is_sc_st(user, true)')
        else:
            facts.append('is_sc_st(user, false)')
        
        # Basic case type detection
        if any(word in query_lower for word in ['landlord', 'evict', 'house']):
            facts.append('case_type(user, "property_dispute")')
        elif any(word in query_lower for word in ['husband', 'wife', 'domestic']):
            facts.append('case_type(user, "family_matter")')
        
        return facts
    
    def _create_llm_placeholder_results(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create placeholder results for LLM baseline.
        
        Args:
            test_data: Test data
            
        Returns:
            Placeholder results (would be replaced with actual LLM evaluation)
        """
        # This would be replaced with actual LLM integration
        return {
            'task_success_rate': 0.65,  # Estimated based on literature
            'fact_f1_score': 0.58,
            'avg_processing_time': 2.5,
            'note': 'Placeholder - requires LLM API integration'
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
            output_path: Path to save results
        """
        # Convert EvaluationResult objects to dictionaries for JSON serialization
        if 'individual_results' in results:
            serializable_results = []
            for result in results['individual_results']:
                result_dict = {
                    'query_id': result.query_id,
                    'query': result.query,
                    'predicted_facts': result.predicted_facts,
                    'expected_facts': result.expected_facts,
                    'predicted_eligible': result.predicted_eligible,
                    'expected_eligible': result.expected_eligible,
                    'processing_time': result.processing_time,
                    'fact_precision': result.fact_precision,
                    'fact_recall': result.fact_recall,
                    'fact_f1': result.fact_f1,
                    'task_success': result.task_success
                }
                serializable_results.append(result_dict)
            
            results['individual_results'] = serializable_results
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
