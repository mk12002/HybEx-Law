"""
SLM Evaluation Script - Comprehensive evaluation of SLM vs baseline pipelines.

This script provides end-to-end evaluation comparing SLM-based fact extraction
with the baseline TF-IDF approach.

Usage:
    python scripts/evaluate_slm.py --model-path models/slm_legal_facts
    python scripts/evaluate_slm.py --baseline-only  # Test baseline only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.slm_pipeline.slm_pipeline import SLMPipeline
from src.slm_pipeline.slm_evaluator import SLMEvaluator
from src.nlp_pipeline.hybrid_pipeline import HybridLegalNLPPipeline
from src.prolog_engine.legal_engine import LegalAidEngine
from data.sample_data import SAMPLE_QUERIES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_data_path: str = None) -> List[Dict[str, Any]]:
    """Load test data from file or use samples."""
    if test_data_path and Path(test_data_path).exists():
        logger.info(f"Loading test data from: {test_data_path}")
        with open(test_data_path, 'r') as f:
            return json.load(f)
    else:
        logger.info("Using sample queries for testing")
        return SAMPLE_QUERIES


def run_slm_evaluation(
    slm_model_path: str,
    test_data: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Run SLM pipeline evaluation."""
    logger.info(f"Evaluating SLM model: {slm_model_path}")
    
    # Initialize SLM pipeline
    slm_pipeline = SLMPipeline(model_path=slm_model_path)
    
    # Initialize legal engine
    legal_engine = LegalAidEngine()
    
    # Initialize evaluator
    evaluator = SLMEvaluator(
        slm_pipeline=slm_pipeline,
        legal_engine=legal_engine
    )
    
    # Run evaluation
    results = evaluator.evaluate_slm_pipeline(test_data)
    
    # Save results
    results_path = Path(output_dir) / "slm_evaluation_results.json"
    evaluator.save_results(results, str(results_path))
    
    return results


def run_baseline_evaluation(
    test_data: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Run baseline pipeline evaluation."""
    logger.info("Evaluating baseline pipeline")
    
    # Initialize baseline pipeline
    baseline_pipeline = HybridLegalNLPPipeline()
    
    # Initialize legal engine
    legal_engine = LegalAidEngine()
    
    # Initialize baseline evaluator
    from src.evaluation.evaluator import HybExEvaluator
    evaluator = HybExEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_pipeline(test_data)
    
    # Save results
    results_path = Path(output_dir) / "baseline_evaluation_results.json"
    evaluator.save_results(results, str(results_path))
    
    return results


def run_comparison_evaluation(
    slm_model_path: str,
    test_data: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Run comprehensive comparison between SLM and baseline."""
    logger.info("Running comprehensive SLM vs Baseline comparison")
    
    # Initialize pipelines
    slm_pipeline = SLMPipeline(model_path=slm_model_path)
    baseline_pipeline = HybridLegalNLPPipeline()
    legal_engine = LegalAidEngine()
    
    # Initialize evaluator
    evaluator = SLMEvaluator(
        slm_pipeline=slm_pipeline,
        baseline_pipeline=baseline_pipeline,
        legal_engine=legal_engine
    )
    
    # Run comparison
    comparison_results = evaluator.compare_pipelines(test_data)
    
    # Generate detailed report
    report = evaluator.generate_detailed_report(
        comparison_results,
        output_path=str(Path(output_dir) / "evaluation_report.md")
    )
    
    # Save comparison results
    results_path = Path(output_dir) / "comparison_results.json"
    evaluator.save_results(comparison_results, str(results_path))
    
    return comparison_results


def print_results_summary(results: Dict[str, Any], pipeline_name: str):
    """Print a summary of evaluation results."""
    if 'aggregate_metrics' not in results:
        print(f"❌ No metrics found for {pipeline_name}")
        return
    
    metrics = results['aggregate_metrics']
    
    print(f"\n📊 {pipeline_name} Results Summary:")
    print("=" * 40)
    print(f"Task Success Rate: {metrics.get('task_success_rate', 0):.1%}")
    print(f"Average Fact F1: {metrics.get('avg_fact_f1', 0):.3f}")
    print(f"Average Fact Precision: {metrics.get('avg_fact_precision', 0):.3f}")
    print(f"Average Fact Recall: {metrics.get('avg_fact_recall', 0):.3f}")
    print(f"Average Processing Time: {metrics.get('avg_processing_time', 0):.3f}s")
    
    if 'exact_match_rate' in metrics:
        print(f"Exact Match Rate: {metrics['exact_match_rate']:.1%}")
    
    print(f"Total Queries: {metrics.get('total_queries', 0)}")


def print_comparison_summary(comparison_results: Dict[str, Any]):
    """Print comparison summary."""
    if 'comparison_metrics' not in comparison_results:
        print("❌ No comparison metrics found")
        return
    
    comp_metrics = comparison_results['comparison_metrics']
    
    print(f"\n🆚 SLM vs Baseline Comparison:")
    print("=" * 40)
    
    if 'improvements' in comp_metrics:
        improvements = comp_metrics['improvements']
        
        for metric, improvement in improvements.items():
            if 'improvement_percent' in metric:
                metric_name = metric.replace('_improvement_percent', '').replace('_', ' ').title()
                status = "📈" if improvement > 0 else "📉"
                print(f"{status} {metric_name}: {improvement:+.1f}%")
    
    if 'recommendation' in comp_metrics:
        print(f"\n💡 Recommendation: {comp_metrics['recommendation']}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate SLM vs baseline pipelines")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained SLM model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate baseline pipeline"
    )
    parser.add_argument(
        "--slm-only",
        action="store_true",
        help="Only evaluate SLM pipeline"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation on first 5 samples"
    )
    
    args = parser.parse_args()
    
    print("🔍 HybEx-Law Pipeline Evaluation")
    print("=" * 50)
    
    # Load test data
    test_data = load_test_data(args.test_data)
    
    if args.quick:
        test_data = test_data[:5]
        print(f"🚀 Quick evaluation mode: using {len(test_data)} samples")
    
    print(f"📊 Test Data: {len(test_data)} queries")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        
        if args.baseline_only:
            # Evaluate baseline only
            print("\n🔧 Evaluating baseline pipeline...")
            baseline_results = run_baseline_evaluation(test_data, str(output_dir))
            print_results_summary(baseline_results, "Baseline")
            
        elif args.slm_only:
            # Evaluate SLM only
            if not args.model_path:
                print("❌ Error: --model-path required for SLM evaluation")
                sys.exit(1)
            
            print(f"\n🤖 Evaluating SLM pipeline...")
            slm_results = run_slm_evaluation(args.model_path, test_data, str(output_dir))
            print_results_summary(slm_results, "SLM")
            
        else:
            # Run comprehensive comparison
            if not args.model_path:
                print("❌ Error: --model-path required for comparison")
                sys.exit(1)
            
            print(f"\n🆚 Running comprehensive comparison...")
            comparison_results = run_comparison_evaluation(args.model_path, test_data, str(output_dir))
            
            # Print summaries
            if 'slm_results' in comparison_results:
                print_results_summary(comparison_results['slm_results'], "SLM")
            
            if 'baseline_results' in comparison_results:
                print_results_summary(comparison_results['baseline_results'], "Baseline")
            
            print_comparison_summary(comparison_results)
        
        evaluation_time = time.time() - start_time
        
        print(f"\n✅ Evaluation completed in {evaluation_time:.2f} seconds")
        print(f"📁 Results saved to: {output_dir}")
        
        # Show next steps
        print(f"\n🎯 Next Steps:")
        print("1. Review detailed results in the output directory")
        print("2. Check evaluation_report.md for comprehensive analysis")
        print("3. Use results for research paper or further optimization")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check model path exists (for SLM evaluation)")
        print("3. Verify test data format")
        sys.exit(1)


if __name__ == "__main__":
    main()
