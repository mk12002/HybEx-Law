"""
HybEx-Law Complete Training Pipeline
===================================

Comprehensive training orchestrator with robust monitoring and evaluation.
"""
# hybex_system/trainer.py

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import time
import numpy as np
from dataclasses import asdict
from tqdm import tqdm

# Assuming these modules exist and are importable relative to the package root
from .config import HybExConfig
from .data_processor import DataPreprocessor
from .neural_models import ModelTrainer
from .prolog_engine import PrologEngine
from .evaluator import ModelEvaluator

# Setup logging
logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: HybExConfig):
        self.config = config
        self.start_time = None
        self.components = {}
        
        # Setup comprehensive logging
        self.setup_logging()
        
        # Initialize components
        self.initialize_components()
        
        logger.info("HybEx-Law Training Orchestrator Initialized")
        # Assuming HybExConfig has a to_dict method
        logger.info(f"Configuration: {self.config.to_dict()}")
    
    def setup_logging(self):
        """Setup comprehensive training logging"""
        # Ensure handlers are not duplicated if called multiple times
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('training_orchestrator.log') for h in logger.handlers):
            # Assuming config.get_log_path is implemented and returns a file path string
            log_file = self.config.get_log_path('training_orchestrator')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            # Assuming config.LOGGING_CONFIG is a dictionary with 'format' key
            formatter = logging.Formatter(self.config.LOGGING_CONFIG['format'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to TrainingOrchestrator logger.")
        
        logger.info("="*80)
        logger.info("HybEx-Law Complete Training Pipeline Started")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("="*80)
    
    def initialize_components(self):
        """Initialize all training components"""
        logger.info("Initializing training components...")
        
        try:
            # Data preprocessor
            self.components['data_processor'] = DataPreprocessor(self.config)
            logger.info("Data Preprocessor initialized")
            
            # Neural model trainer
            self.components['model_trainer'] = ModelTrainer(self.config)
            logger.info("Neural Model Trainer initialized")
            
            # Prolog reasoning engine
            self.components['prolog_engine'] = PrologEngine(self.config)
            logger.info("Prolog Reasoning Engine initialized")

            # Model Evaluator
            self.components['model_evaluator'] = ModelEvaluator(self.config)
            logger.info("Model Evaluator initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    # --- New GNN Training Method ---
    def _train_gnn_model(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict[str, Any]:
        """Dedicated stage for training the Knowledge Graph Neural Network."""
        logger.info("Starting dedicated GNN Model Training...")
        try:
            # ASSUMPTION: ModelTrainer has a method for GNN training, 
            # and it's separate from the 'train_all_models' in Stage 2.
            gnn_results = self.components['model_trainer'].train_gnn_model_component(
                train_samples,
                val_samples
            )
            logger.info("GNN Model Training completed.")
            return gnn_results
        except AttributeError:
            logger.error("ModelTrainer component is missing the 'train_gnn_model_component' method. Skipping GNN training.")
            return {
                'gnn_training': {'status': 'skipped', 'error': 'Method not found in ModelTrainer'}
            }
        except Exception as e:
            logger.error(f"GNN Model Training failed: {e}", exc_info=True)
            raise

    def run_complete_training_pipeline(self, data_directory: str) -> Dict[str, Any]:
        """Execute the complete HybEx-Law training pipeline"""
        self.start_time = time.time()
        
        logger.info("Starting Complete HybEx-Law Training Pipeline")
        logger.info(f"Data Directory: {data_directory}")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'pipeline_stages': {},
            'final_models': {},
            'evaluation_results': {},
            'errors': []
        }
        
        # Define pipeline stages for progress tracking (Added GNN Training as Stage 2B)
        stages = [
            "Data Preprocessing", 
            "Neural Model Training (Non-GNN)", 
            "Knowledge Graph Neural Network Training (GNN)", # NEW STAGE
            "Prolog Integration", 
            "System Evaluation"
        ]
        
        processed_data = {'train_samples': [], 'val_samples': [], 'test_samples': []}

        # Overall progress bar for main stages
        main_pbar = tqdm(stages, desc="HybEx-Law Training Pipeline", unit="stage", colour="green")
        
        try:
            # Stage 1: Data Preprocessing
            main_pbar.set_description("STAGE 1: DATA PREPROCESSING")
            logger.info("\n" + "="*80)
            logger.info("STAGE 1: DATA PREPROCESSING")
            logger.info("="*80)
            
            preprocessing_start = time.time()
            preprocessing_results = self.components['data_processor'].run_preprocessing_pipeline(data_directory)
            preprocessing_time = time.time() - preprocessing_start
            
            pipeline_results['pipeline_stages']['preprocessing'] = {
                'results': preprocessing_results,
                'duration_seconds': preprocessing_time,
                'status': 'completed'
            }
            logger.info(f"Data preprocessing completed in {preprocessing_time:.2f} seconds")
            main_pbar.update(1)
            
            # Load processed data
            processed_data = self._load_processed_data(preprocessing_results['saved_files'])
            
            # Stage 2: Neural Model Training (Standard/Non-GNN Models)
            main_pbar.set_description("STAGE 2: NEURAL MODEL TRAINING (STANDARD)")
            logger.info("\n" + "="*80)
            logger.info("STAGE 2: NEURAL MODEL TRAINING (STANDARD/NON-GNN)")
            logger.info("="*80)
            
            training_start = time.time()
            training_results_standard = self.components['model_trainer'].train_all_models(
                processed_data['train_samples'],
                processed_data['val_samples']
            )
            training_time_standard = time.time() - training_start
            
            pipeline_results['pipeline_stages']['neural_training_standard'] = {
                'results': {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()} for k,v in training_results_standard.items()},
                'duration_seconds': training_time_standard,
                'status': 'completed'
            }
            # Add standard models to final_models
            pipeline_results['final_models'].update({
                model_name: {
                    'path': str(model_info['path']),
                    'best_f1': model_info['best_f1']
                } for model_name, model_info in training_results_standard.items()
            })
            
            logger.info(f"Standard neural model training completed in {training_time_standard:.2f} seconds")
            main_pbar.update(1)

            # Stage 3: Knowledge Graph Neural Network Training (GNN)
            main_pbar.set_description("STAGE 3: KGNN TRAINING")
            logger.info("\n" + "="*80)
            logger.info("STAGE 3: KNOWLEDGE GRAPH NEURAL NETWORK TRAINING")
            logger.info("="*80)

            gnn_training_start = time.time()
            gnn_training_results = self._train_gnn_model(
                processed_data['train_samples'],
                processed_data['val_samples']
            )
            gnn_training_time = time.time() - gnn_training_start

            pipeline_results['pipeline_stages']['neural_training_gnn'] = {
                'results': self._make_serializable(gnn_training_results),
                'duration_seconds': gnn_training_time,
                'status': gnn_training_results.get('gnn_training', {}).get('status', 'completed')
            }
            # Add GNN model to final_models (assuming 'gnn_model' is a key in results)
            if 'gnn_model' in gnn_training_results:
                 model_info = gnn_training_results['gnn_model']
                 pipeline_results['final_models']['gnn_model'] = {
                    'path': str(model_info['path']),
                    'best_f1': model_info['best_f1']
                }
            
            logger.info(f"GNN training completed in {gnn_training_time:.2f} seconds (Status: {pipeline_results['pipeline_stages']['neural_training_gnn']['status']})")
            main_pbar.update(1)
            
            # Stage 4: Prolog Rule Integration / Initial Symbolic Evaluation
            main_pbar.set_description("STAGE 4: PROLOG RULE INTEGRATION")
            logger.info("\n" + "="*80)
            logger.info("STAGE 4: PROLOG RULE INTEGRATION & INITIAL SYMBOLIC EVALUATION")
            logger.info("="*80)
            
            prolog_start = time.time()
            prolog_results = self._integrate_prolog_reasoning_and_evaluation(processed_data['test_samples'])
            prolog_time = time.time() - prolog_start
            
            pipeline_results['pipeline_stages']['prolog_integration'] = {
                'results': prolog_results,
                'duration_seconds': prolog_time,
                'status': 'completed'
            }
            
            logger.info(f"Prolog integration completed in {prolog_time:.2f} seconds")
            main_pbar.update(1)
            
            # Stage 5: Comprehensive Evaluation (End-to-End & Hybrid Performance)
            main_pbar.set_description("STAGE 5: COMPREHENSIVE EVALUATION")
            logger.info("\n" + "="*80)
            logger.info("STAGE 5: COMPREHENSIVE EVALUATION")
            logger.info("="*80)
            
            evaluation_start = time.time()
            logger.info("\n--- Starting Comprehensive End-to-End System Evaluation ---")
            
            serializable_prolog_results = prolog_results.get('reasoning_results', [])

            evaluation_results = self.components['model_evaluator'].evaluate_end_to_end_system(
                models_paths=pipeline_results['final_models'],
                test_samples=processed_data['test_samples'],
                prolog_reasoning_results=serializable_prolog_results
            )
            
            evaluation_time = time.time() - evaluation_start
            
            pipeline_results['pipeline_stages']['evaluation'] = {
                'results': evaluation_results,
                'duration_seconds': evaluation_time,
                'status': 'completed'
            }
            
            pipeline_results['evaluation_results'] = evaluation_results
            
            logger.info(f"Comprehensive evaluation completed in {evaluation_time:.2f} seconds")
            main_pbar.update(1)
            
            # Finalize pipeline
            total_time = time.time() - self.start_time
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['total_duration_seconds'] = total_time
            pipeline_results['status'] = 'completed'
            
            # Save final results
            results_file = self._save_pipeline_results(pipeline_results)
            pipeline_results['results_file'] = results_file
            
            # Generate final report
            self._generate_final_report(pipeline_results)
            
            logger.info("\n" + "="*80)
            logger.info("HYBEX-LAW TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total Duration: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"Results saved to: {results_file}")
            
            main_pbar.close()
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            pipeline_results['errors'].append({
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'stage': 'pipeline_execution'
            })
            
            pipeline_results['status'] = 'failed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            error_file = self._save_pipeline_results(pipeline_results)
            pipeline_results['results_file'] = error_file
            
            main_pbar.close()
            
            raise
    
    # Rest of the helper methods remain the same as the previous full correction
    
    def _load_processed_data(self, saved_files: Dict[str, str]) -> Dict[str, List]:
        """Load processed data from saved files"""
        logger.info("Loading processed data...")
        
        processed_data = {}
        
        for split_name in ['train', 'val', 'test']:
            file_path = saved_files.get(f'{split_name}_data_file')
            if file_path and Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed_data[f'{split_name}_samples'] = data
                    logger.info(f"Loaded {len(data)} {split_name} samples")
            else:
                logger.warning(f"Could not load {split_name} data from {file_path}. Skipping.")
                processed_data[f'{split_name}_samples'] = []
        
        # Ensure 'extracted_entities' are present in samples for Prolog processing
        for split in ['train_samples', 'val_samples', 'test_samples']:
            if processed_data.get(split):
                if not processed_data[split] or 'extracted_entities' not in processed_data[split][0]:
                    logger.warning(f"Samples in {split} are missing 'extracted_entities'. This might impact Prolog integration. "
                                    "Consider updating DataPreprocessor to always include them.")
                    for sample in processed_data[split][:10]:
                        if 'query' in sample and 'extracted_entities' not in sample:
                            sample['extracted_entities'] = self.components['data_processor'].extract_entities(sample['query'])
                            logger.debug(f"Extracted entities on the fly for sample ID {sample.get('sample_id', 'N/A')}")
                else:
                    logger.info(f"Samples in {split} contain 'extracted_entities'.")

        return processed_data
    
    def _integrate_prolog_reasoning_and_evaluation(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Integrate Prolog reasoning with test samples for evaluation purposes.
        This runs the Prolog engine over the test data."""
        logger.info("Integrating Prolog reasoning for evaluation...")
        
        cases_for_prolog_batch = []
        for sample in test_samples:
            if 'extracted_entities' in sample:
                cases_for_prolog_batch.append(sample['extracted_entities'])
            else:
                logger.warning(f"Sample ID {sample.get('sample_id', 'N/A')} missing 'extracted_entities'. Extracting on the fly.")
                extracted = self.components['data_processor'].extract_entities(sample.get('query', ''))
                cases_for_prolog_batch.append(extracted)

        if not cases_for_prolog_batch:
            logger.warning("No cases with extracted entities available for Prolog batch analysis.")
            return {
                'total_cases_evaluated': 0,
                'eligible_cases': 0,
                'eligibility_rate': 0,
                'average_confidence': 0,
                'results_file': 'N/A',
                'prolog_engine_available': self.components['prolog_engine'].prolog_available,
                'reasoning_results': []
            }

        reasoning_results_dataclasses = self.components['prolog_engine'].batch_legal_analysis(cases_for_prolog_batch)
        
        reasoning_results_dicts = [asdict(res) for res in reasoning_results_dataclasses]

        results_file = self.components['prolog_engine'].save_reasoning_results(reasoning_results_dicts, filename="prolog_batch_reasoning_results.json")
        
        eligible_count = sum(1 for r in reasoning_results_dicts if r.get('eligible', False))
        total_count = len(reasoning_results_dicts)
        
        integration_results = {
            'total_cases_evaluated': total_count,
            'eligible_cases': eligible_count,
            'eligibility_rate': eligible_count / total_count if total_count > 0 else 0,
            'average_confidence': np.mean([r.get('confidence', 0) for r in reasoning_results_dicts]) if reasoning_results_dicts else 0.0,
            'results_file': results_file,
            'prolog_engine_available': self.components['prolog_engine'].prolog_available,
            'reasoning_results': reasoning_results_dicts
        }
        
        logger.info(f"Prolog reasoning integration and batch evaluation complete:")
        logger.info(f"  - Cases evaluated: {total_count}")
        logger.info(f"  - Eligible cases: {eligible_count}")
        logger.info(f"  - Eligibility rate: {integration_results['eligibility_rate']*100:.1f}%")
        
        return integration_results

    def _save_pipeline_results(self, pipeline_results: Dict) -> str:
        """Save complete pipeline results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.RESULTS_DIR / f"pipeline_results_{timestamp}.json"
        
        serializable_results = self._make_serializable(pipeline_results)
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Pipeline results saved to {results_file}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}", exc_info=True)
            temp_file = self.config.RESULTS_DIR / f"pipeline_results_temp_{timestamp}.json"
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2)
                logger.warning(f"Saved to temporary file due to error: {temp_file}")
                return str(temp_file)
            except Exception as inner_e:
                logger.error(f"Failed to save even to temporary file: {inner_e}")
                return ""
    
    def _make_serializable(self, obj) -> Any:
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif hasattr(obj, '__dataclass_fields__'):
            return self._make_serializable(asdict(obj))
        elif hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_') and not callable(v)}
        else:
            return obj
    
    def _generate_final_report(self, pipeline_results: Dict):
        """Generate comprehensive final report"""
        logger.info("\n" + "="*80)
        logger.info("HYBEX-LAW TRAINING PIPELINE - FINAL REPORT")
        logger.info("="*80)
        
        total_time = pipeline_results.get('total_duration_seconds', 0)
        logger.info(f"\nPIPELINE OVERVIEW:")
        logger.info(f"  • Total Duration: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"  • Status: {pipeline_results.get('status', 'unknown')}")
        logger.info(f"  • Start Time: {pipeline_results.get('start_time', 'N/A')}")
        logger.info(f"  • End Time: {pipeline_results.get('end_time', 'N/A')}")
        
        logger.info(f"\nSTAGE PERFORMANCE:")
        stages = pipeline_results.get('pipeline_stages', {})
        for stage_name, stage_info in stages.items():
            duration = stage_info.get('duration_seconds', 0)
            status = stage_info.get('status', 'unknown')
            logger.info(f"  • {stage_name.replace('_', ' ').title()}: {duration:.2f}s ({status})")
        
        logger.info(f"\nNEURAL MODEL PERFORMANCE:")
        models = pipeline_results.get('final_models', {})
        for model_name, model_info in models.items():
            f1_score = model_info.get('best_f1', 0)
            logger.info(f"  • {model_name}: F1 = {f1_score:.4f}")
        
        eval_results = pipeline_results.get('evaluation_results', {})
        prolog_acc = eval_results.get('prolog_reasoning_accuracy', {})
        if prolog_acc:
            logger.info(f"\nPROLOG REASONING PERFORMANCE:")
            logger.info(f"  • Accuracy: {prolog_acc.get('accuracy', 0):.4f}")
            logger.info(f"  • Predictions: {prolog_acc.get('correct_predictions', 0)}/{prolog_acc.get('total_predictions', 0)}")
            logger.info(f"  • Eligibility Rate: {prolog_acc.get('eligibility_rate', 0):.3f}")
        
        logger.info(f"\nSYSTEM CONFIGURATION:")
        config = pipeline_results.get('config', {})
        logger.info(f"  • Base Model: {config.get('MODEL_CONFIG', {}).get('base_model', 'N/A')}")
        
        # FIX: Use correct nested keys from model_configs
        model_configs = config.get('model_configs', {})
        domain_config = model_configs.get('domain_classifier', {})
        
        # Use the domain_classifier config as the representative example for the report
        logger.info(f"  • Training Epochs (Domain Classifier): {domain_config.get('epochs', 'N/A')}")
        logger.info(f"  • Batch Size (Domain Classifier): {domain_config.get('batch_size', 'N/A')}")
        
        logger.info(f"\nGENERATED FILES:")
        logger.info(f"  • Pipeline Results: {pipeline_results.get('results_file', 'N/A')}")
        logger.info(f"  • Models Directory: {self.config.MODELS_DIR}")
        logger.info(f"  • Results Directory: {self.config.RESULTS_DIR}")
        logger.info(f"  • Logs Directory: {self.config.LOGS_DIR}")
        
        errors = pipeline_results.get('errors', [])
        if errors:
            logger.info(f"\nERRORS ENCOUNTERED:")
            for i, error in enumerate(errors, 1):
                logger.info(f"  • Error {i}: {error.get('error', 'Unknown error')}")
        
        logger.info("\n" + "="*80)
        logger.info("Report Generation Complete")
        logger.info("="*80)
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up training resources...")
        
        if 'prolog_engine' in self.components:
            self.components['prolog_engine'].cleanup()
        
        logger.info("Cleanup completed")

def main():
    """Main entry point for training pipeline"""
    trainer = None
    try:
        config = HybExConfig()
        trainer = TrainingOrchestrator(config)
        data_directory = "data"
        results = trainer.run_complete_training_pipeline(data_directory)
        
        trainer.cleanup()
        
        print("\nHybEx-Law training completed successfully!")
        print(f"Results saved to: {results.get('results_file', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        if trainer:
            trainer.cleanup()
        print(f"\nTraining pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()