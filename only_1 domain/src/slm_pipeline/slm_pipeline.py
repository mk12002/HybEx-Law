"""
SLM Pipeline - Main orchestrator for SLM-based fact extraction.

This module implements the complete SLM pipeline that can be used as a drop-in
replacement for the TF-IDF baseline pipeline.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .slm_fact_extractor import SLMFactExtractor
from ..utils.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class SLMPipeline:
    """
    Main SLM pipeline that orchestrates SLM-based fact extraction.
    
    This pipeline provides a direct replacement for the two-stage TF-IDF pipeline,
    using a fine-tuned SLM to directly extract Prolog facts from natural language.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        model_path: Optional[str] = None,
        use_preprocessing: bool = True,
        device: str = "auto"
    ):
        """
        Initialize the SLM pipeline.
        
        Args:
            model_name: Base SLM model name
            model_path: Path to fine-tuned model
            use_preprocessing: Whether to apply text preprocessing
            device: Device for model computation
        """
        self.model_name = model_name
        self.model_path = model_path
        self.use_preprocessing = use_preprocessing
        self.device = device
        
        # Initialize components
        self.slm_extractor = SLMFactExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device
        )
        
        if use_preprocessing:
            self.preprocessor = TextPreprocessor()
        else:
            self.preprocessor = None
        
        logger.info(f"SLM Pipeline initialized with model: {model_name}")
    
    def process_query(self, query: str, verbose: bool = False) -> List[str]:
        """
        Process a natural language query and extract legal facts.
        
        Args:
            query: Natural language description of legal situation
            verbose: Whether to show detailed processing steps
            
        Returns:
            List of Prolog facts as strings
        """
        if verbose:
            print(f"📝 Original query: {query}")
        
        # Preprocess the text (optional)
        processed_text = query
        if self.use_preprocessing and self.preprocessor:
            processed_text = self.preprocessor.preprocess(query)
            if verbose:
                print(f"🔧 Preprocessed: {processed_text}")
        
        # Extract facts using SLM
        if verbose:
            print("🤖 Extracting facts with SLM...")
        
        start_time = time.time()
        extracted_facts = self.slm_extractor.extract_facts(processed_text, verbose=verbose)
        processing_time = time.time() - start_time
        
        # Validate and clean facts
        validated_facts = self._validate_facts(extracted_facts)
        
        if verbose:
            print(f"✅ Final validated facts: {validated_facts}")
            print(f"⏱️ Total processing time: {processing_time:.3f}s")
        
        return validated_facts
    
    def _validate_facts(self, facts: List[str]) -> List[str]:
        """
        Validate and clean extracted facts.
        
        Args:
            facts: Raw extracted facts
            
        Returns:
            Cleaned and validated facts
        """
        validated = []
        seen_facts = set()
        
        for fact in facts:
            # Clean the fact
            cleaned_fact = fact.strip()
            if not cleaned_fact.endswith('.'):
                cleaned_fact += '.'
            
            # Remove duplicates
            if cleaned_fact not in seen_facts:
                validated.append(cleaned_fact)
                seen_facts.add(cleaned_fact)
        
        return validated
    
    def load_model(self):
        """Load the SLM model."""
        self.slm_extractor.load_model()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline component information
        """
        info = {
            'pipeline_type': 'SLM',
            'model_name': self.model_name,
            'model_path': self.model_path,
            'use_preprocessing': self.use_preprocessing,
            'device': self.device
        }
        
        # Add SLM extractor info
        info.update(self.slm_extractor.get_model_info())
        
        return info
    
    def benchmark_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Benchmark pipeline performance on test queries.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking SLM pipeline on {len(test_queries)} queries")
        
        # Load model if not already loaded
        if not self.slm_extractor.is_loaded:
            self.load_model()
        
        total_time = 0
        successful_runs = 0
        total_facts = 0
        errors = []
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                facts = self.process_query(query, verbose=False)
                processing_time = time.time() - start_time
                
                total_time += processing_time
                successful_runs += 1
                total_facts += len(facts)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(test_queries)} queries")
                    
            except Exception as e:
                error_msg = f"Query {i}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        # Calculate metrics
        avg_time = total_time / successful_runs if successful_runs > 0 else 0
        success_rate = successful_runs / len(test_queries)
        avg_facts_per_query = total_facts / successful_runs if successful_runs > 0 else 0
        
        results = {
            'total_queries': len(test_queries),
            'successful_runs': successful_runs,
            'success_rate': success_rate,
            'avg_processing_time': avg_time,
            'total_processing_time': total_time,
            'avg_facts_per_query': avg_facts_per_query,
            'total_facts_extracted': total_facts,
            'errors': errors
        }
        
        logger.info(f"Benchmark complete: {success_rate:.1%} success rate, {avg_time:.3f}s avg time")
        
        return results
    
    def compare_with_baseline(self, test_queries: List[str], baseline_pipeline) -> Dict[str, Any]:
        """
        Compare SLM pipeline performance with baseline pipeline.
        
        Args:
            test_queries: Test queries for comparison
            baseline_pipeline: Baseline pipeline to compare against
            
        Returns:
            Comparison results
        """
        logger.info("Comparing SLM pipeline with baseline...")
        
        # Benchmark both pipelines
        slm_results = self.benchmark_performance(test_queries)
        baseline_results = baseline_pipeline.benchmark_performance(test_queries)
        
        # Calculate improvements
        time_improvement = (
            (baseline_results['avg_processing_time'] - slm_results['avg_processing_time']) /
            baseline_results['avg_processing_time'] * 100
        ) if baseline_results['avg_processing_time'] > 0 else 0
        
        success_improvement = (
            slm_results['success_rate'] - baseline_results['success_rate']
        ) * 100
        
        comparison = {
            'slm_pipeline': slm_results,
            'baseline_pipeline': baseline_results,
            'improvements': {
                'processing_time_change_percent': time_improvement,
                'success_rate_change_percent': success_improvement,
                'avg_facts_difference': slm_results['avg_facts_per_query'] - baseline_results['avg_facts_per_query']
            },
            'summary': {
                'slm_faster': time_improvement > 0,
                'slm_more_accurate': success_improvement > 0,
                'recommended_pipeline': 'SLM' if (time_improvement > -50 and success_improvement > 0) else 'Baseline'
            }
        }
        
        logger.info(f"Comparison complete. Recommended: {comparison['summary']['recommended_pipeline']}")
        
        return comparison


# Compatibility wrapper for drop-in replacement
class HybridLegalNLPPipelineSLM(SLMPipeline):
    """
    Drop-in replacement for HybridLegalNLPPipeline using SLM.
    
    This class provides the same interface as the original pipeline
    but uses SLM-based fact extraction instead of TF-IDF.
    """
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """Initialize with same interface as original pipeline."""
        super().__init__(model_path=model_path, **kwargs)
        logger.info("SLM-based HybridLegalNLPPipeline initialized")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    pipeline = SLMPipeline()
    
    test_queries = [
        "I am a woman facing domestic violence. My husband beats me and I earn 15000 rupees.",
        "My landlord is trying to evict me. I am from scheduled caste and unemployed.",
        "I need help with child custody. I work as laborer earning 8000 monthly."
    ]
    
    print("🧪 Testing SLM Pipeline")
    print("=" * 50)
    
    try:
        # Test individual query
        facts = pipeline.process_query(test_queries[0], verbose=True)
        print(f"\n✅ Extracted facts: {facts}")
        
        # Benchmark performance
        print("\n📊 Benchmarking performance...")
        results = pipeline.benchmark_performance(test_queries)
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Avg Time: {results['avg_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: This requires proper SLM dependencies and model access.")
