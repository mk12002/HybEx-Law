"""
Master Training Pipeline for HybEx-Law System.

This script orchestrates the complete training workflow:
1. Data generation
2. Data validation and splitting
3. Individual model training
4. System evaluation
5. Performance reporting
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_training_pipeline(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute the complete training pipeline for HybEx-Law system.
    
    Args:
        config: Configuration dictionary with training parameters
        
    Returns:
        Dictionary containing pipeline execution results
    """
    if config is None:
        config = {
            'data_samples': 5000,
            'training_epochs': 3,
            'validation_split': 0.15,
            'test_split': 0.15,
            'output_base_dir': 'training_output'
        }
    
    logger.info("🚀 Starting Complete HybEx-Law Training Pipeline")
    logger.info(f"Configuration: {config}")
    
    pipeline_results = {
        'start_time': time.time(),
        'stages_completed': [],
        'stage_results': {},
        'errors': [],
        'final_status': 'unknown'
    }
    
    try:
        # Stage 1: Data Generation
        logger.info("\n" + "="*60)
        logger.info("📊 STAGE 1: COMPREHENSIVE DATA GENERATION")
        logger.info("="*60)
        
        stage1_result = run_data_generation_stage(config)
        pipeline_results['stages_completed'].append('data_generation')
        pipeline_results['stage_results']['data_generation'] = stage1_result
        
        if not stage1_result['success']:
            raise Exception(f"Data generation failed: {stage1_result['error']}")
        
        # Stage 2: Data Validation and Splitting
        logger.info("\n" + "="*60)
        logger.info("🔍 STAGE 2: DATA VALIDATION AND SPLITTING")
        logger.info("="*60)
        
        stage2_result = run_data_validation_stage(stage1_result['output_path'])
        pipeline_results['stages_completed'].append('data_validation')
        pipeline_results['stage_results']['data_validation'] = stage2_result
        
        if not stage2_result['success']:
            raise Exception(f"Data validation failed: {stage2_result['error']}")
        
        # Stage 3: Individual Model Training
        logger.info("\n" + "="*60)
        logger.info("🤖 STAGE 3: NEURAL MODEL TRAINING")
        logger.info("="*60)
        
        stage3_result = run_model_training_stage(config)
        pipeline_results['stages_completed'].append('model_training')
        pipeline_results['stage_results']['model_training'] = stage3_result
        
        if not stage3_result['success']:
            logger.warning(f"Model training had issues: {stage3_result.get('error', 'Unknown error')}")
            # Continue with evaluation even if training had issues
        
        # Stage 4: System Evaluation
        logger.info("\n" + "="*60)
        logger.info("📈 STAGE 4: COMPREHENSIVE EVALUATION")
        logger.info("="*60)
        
        stage4_result = run_evaluation_stage(config)
        pipeline_results['stages_completed'].append('evaluation')
        pipeline_results['stage_results']['evaluation'] = stage4_result
        
        if not stage4_result['success']:
            logger.warning(f"Evaluation had issues: {stage4_result.get('error', 'Unknown error')}")
        
        # Stage 5: Final Reporting
        logger.info("\n" + "="*60)
        logger.info("📋 STAGE 5: FINAL REPORTING")
        logger.info("="*60)
        
        stage5_result = generate_final_report(pipeline_results, config)
        pipeline_results['stages_completed'].append('reporting')
        pipeline_results['stage_results']['reporting'] = stage5_result
        
        pipeline_results['final_status'] = 'completed'
        
    except Exception as e:
        logger.error(f"Pipeline failed at stage: {e}")
        pipeline_results['errors'].append(str(e))
        pipeline_results['final_status'] = 'failed'
    
    pipeline_results['end_time'] = time.time()
    pipeline_results['total_duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
    
    return pipeline_results

def run_data_generation_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute data generation stage"""
    try:
        logger.info(f"Generating {config['data_samples']} training samples...")
        
        # Import and run data generation
        from comprehensive_data_generation import ComprehensiveLegalDataGenerator
        
        generator = ComprehensiveLegalDataGenerator()
        dataset = generator.generate_comprehensive_dataset(total_samples=config['data_samples'])
        
        output_path = "data/comprehensive_legal_training_data.json"
        generator.save_dataset(dataset, output_path)
        
        return {
            'success': True,
            'output_path': output_path,
            'samples_generated': len(dataset),
            'message': f"Successfully generated {len(dataset)} training samples"
        }
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': "Data generation stage failed"
        }

def run_data_validation_stage(data_path: str) -> Dict[str, Any]:
    """Execute data validation and splitting stage"""
    try:
        logger.info("Validating training data and creating splits...")
        
        # Import and run validation
        from validate_training_data import LegalDataValidator, TrainingDataPreparator
        
        # Validate dataset
        validator = LegalDataValidator()
        validation_results = validator.validate_dataset(data_path)
        
        if validation_results['invalid_samples'] > validation_results['valid_samples'] * 0.1:
            logger.warning(f"High invalid sample rate: {validation_results['invalid_samples']} invalid samples")
        
        # Prepare training splits
        preparator = TrainingDataPreparator()
        splits = preparator.prepare_training_splits(data_path)
        
        return {
            'success': True,
            'validation_results': validation_results,
            'splits_created': splits,
            'message': f"Data validation completed with {validation_results['valid_samples']} valid samples"
        }
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': "Data validation stage failed"
        }

def run_model_training_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute individual model training stage"""
    try:
        logger.info("Training individual neural models...")
        
        # Check if training splits exist
        splits_dir = Path("data/splits")
        if not splits_dir.exists():
            logger.warning("Training splits not found, skipping neural training")
            return {
                'success': False,
                'error': "Training splits not available",
                'message': "Skipped neural training - splits not found"
            }
        
        # Import and run training
        from train_individual_models import train_all_models
        
        model_paths = train_all_models()
        
        return {
            'success': True,
            'trained_models': model_paths,
            'message': f"Successfully trained {len(model_paths)} neural models"
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': "Model training stage failed"
        }

def run_evaluation_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute comprehensive evaluation stage"""
    try:
        logger.info("Running comprehensive system evaluation...")
        
        # Import and run evaluation
        from evaluate_hybrid_system import run_comprehensive_evaluation
        
        # Find test data
        test_data_paths = [
            "data/splits/domain_classification_test.json",
            "data/comprehensive_legal_training_data.json"
        ]
        
        test_data_path = None
        for path in test_data_paths:
            if Path(path).exists():
                test_data_path = path
                break
        
        if not test_data_path:
            logger.warning("No test data found for evaluation")
            return {
                'success': False,
                'error': "Test data not available",
                'message': "Evaluation skipped - no test data found"
            }
        
        evaluation_results = run_comprehensive_evaluation(test_data_path)
        
        return {
            'success': True,
            'evaluation_results': evaluation_results,
            'message': "Comprehensive evaluation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': "Evaluation stage failed"
        }

def generate_final_report(pipeline_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final pipeline report"""
    try:
        logger.info("Generating final pipeline report...")
        
        # Create output directory
        output_dir = Path("training_output/pipeline_report")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report
        report = {
            'pipeline_summary': {
                'status': pipeline_results['final_status'],
                'total_duration_minutes': pipeline_results['total_duration'] / 60,
                'stages_completed': pipeline_results['stages_completed'],
                'stages_total': 5
            },
            'configuration': config,
            'stage_details': pipeline_results['stage_results'],
            'errors_encountered': pipeline_results['errors'],
            'recommendations': generate_recommendations(pipeline_results)
        }
        
        # Save report
        import json
        report_path = output_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary
        summary_path = output_dir / "pipeline_summary.txt"
        generate_text_summary(report, summary_path)
        
        logger.info(f"Final report saved to {output_dir}")
        
        return {
            'success': True,
            'report_path': str(report_path),
            'summary_path': str(summary_path),
            'message': "Final report generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': "Report generation failed"
        }

def generate_recommendations(pipeline_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on pipeline results"""
    recommendations = []
    
    # Check data generation results
    data_stage = pipeline_results['stage_results'].get('data_generation', {})
    if data_stage.get('success') and data_stage.get('samples_generated', 0) < 5000:
        recommendations.append("Consider generating more training samples for better model performance")
    
    # Check validation results
    validation_stage = pipeline_results['stage_results'].get('data_validation', {})
    if validation_stage.get('success'):
        validation_results = validation_stage.get('validation_results', {})
        invalid_samples = validation_results.get('invalid_samples', 0)
        if invalid_samples > 0:
            recommendations.append(f"Review and fix {invalid_samples} invalid training samples")
    
    # Check training results
    training_stage = pipeline_results['stage_results'].get('model_training', {})
    if not training_stage.get('success'):
        recommendations.append("Install required ML libraries (transformers, torch, sklearn) for neural training")
    
    # Check evaluation results
    eval_stage = pipeline_results['stage_results'].get('evaluation', {})
    if eval_stage.get('success'):
        eval_results = eval_stage.get('evaluation_results', {})
        overall_perf = eval_results.get('overall_performance', {})
        if overall_perf.get('overall_accuracy', 0) < 0.8:
            recommendations.append("Model performance below 80% - consider additional training or data improvements")
    
    # General recommendations
    if 'model_training' not in pipeline_results['stages_completed']:
        recommendations.append("Complete neural model training for full hybrid capabilities")
    
    if pipeline_results['final_status'] == 'failed':
        recommendations.append("Address pipeline failures and re-run complete training")
    
    return recommendations

def generate_text_summary(report: Dict[str, Any], filepath: Path):
    """Generate human-readable text summary"""
    with open(filepath, 'w') as f:
        f.write("HYBEX-LAW TRAINING PIPELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        summary = report['pipeline_summary']
        f.write(f"Status: {summary['status'].upper()}\n")
        f.write(f"Duration: {summary['total_duration_minutes']:.1f} minutes\n")
        f.write(f"Stages Completed: {len(summary['stages_completed'])}/{summary['stages_total']}\n\n")
        
        f.write("STAGES COMPLETED:\n")
        for stage in summary['stages_completed']:
            f.write(f"  ✅ {stage.replace('_', ' ').title()}\n")
        f.write("\n")
        
        if report['errors_encountered']:
            f.write("ERRORS ENCOUNTERED:\n")
            for error in report['errors_encountered']:
                f.write(f"  ❌ {error}\n")
            f.write("\n")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            f.write("RECOMMENDATIONS:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"  {i}. {rec}\n")
            f.write("\n")
        
        f.write("NEXT STEPS:\n")
        if summary['status'] == 'completed':
            f.write("  1. Review evaluation results\n")
            f.write("  2. Test system with real queries\n")
            f.write("  3. Deploy for production use\n")
        else:
            f.write("  1. Address pipeline failures\n")
            f.write("  2. Re-run training pipeline\n")
            f.write("  3. Verify system functionality\n")

def main():
    """Main function to run complete training pipeline"""
    logger.info("Starting HybEx-Law Complete Training Pipeline")
    
    # Configuration
    config = {
        'data_samples': 5000,
        'training_epochs': 3,
        'validation_split': 0.15,
        'test_split': 0.15,
        'output_base_dir': 'training_output'
    }
    
    # Run pipeline
    results = run_complete_training_pipeline(config)
    
    # Print final status
    print("\n" + "="*60)
    print("🎯 TRAINING PIPELINE COMPLETE")
    print("="*60)
    
    print(f"\n📊 Status: {results['final_status'].upper()}")
    print(f"⏱️  Duration: {results['total_duration']/60:.1f} minutes")
    print(f"✅ Stages Completed: {len(results['stages_completed'])}/5")
    
    if results['stages_completed']:
        print(f"\n📋 Completed Stages:")
        for stage in results['stages_completed']:
            print(f"   • {stage.replace('_', ' ').title()}")
    
    if results['errors']:
        print(f"\n⚠️  Errors Encountered:")
        for error in results['errors']:
            print(f"   • {error}")
    
    # Show final recommendations
    final_report = results['stage_results'].get('reporting', {})
    if final_report.get('success'):
        print(f"\n📄 Full Report: {final_report['report_path']}")
        print(f"📋 Summary: {final_report['summary_path']}")
    
    print(f"\n🚀 HybEx-Law training pipeline {'completed successfully!' if results['final_status'] == 'completed' else 'completed with issues.'}")

if __name__ == "__main__":
    main()
