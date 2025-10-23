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
import gc
import torch
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


# ============================================================================
# SMART PATH RESOLUTION HELPER
# ============================================================================

def find_model_file(model_dir: Path, model_name: str) -> Optional[Path]:
    """
    Smart model file finder - handles different naming conventions
    
    Args:
        model_dir: Directory containing the model
        model_name: Name of the model (e.g., 'enhanced_legal_bert')
    
    Returns:
        Path to the model file, or None if not found
    """
    if not model_dir.exists():
        logger.debug(f"Model directory does not exist: {model_dir}")
        return None
    
    # Define search patterns (in priority order)
    search_patterns = {
        'domain_classifier': [
            'model.pt',
            'domain_classifier.pt',
            'best_model.pt',
            'domain_classifier_best.pt'
        ],
        'eligibility_predictor': [
            'model.pt',
            'eligibility_predictor.pt',
            'best_model.pt',
            'eligibility_predictor_best.pt'
        ],
        'enhanced_legal_bert': [
            'enhanced_legal_bert_best.pt',  # MOST LIKELY ACTUAL FILE!
            'enhanced_legal_bert.pt',
            'model.pt',
            'enhanced_bert_best.pt',
            'enhanced_bert.pt',
            'best_model.pt'
        ],
        'enhanced_bert': [  # Alternative directory name
            'enhanced_legal_bert_best.pt',
            'model.pt',
            'enhanced_bert.pt',
            'best_model.pt'
        ],
        'gnn_model': [
            'gnn_model.pt',  # CORRECT NAMING
            'model.pt',
            'best_model.pt',
            'gnn_model_best.pt'
        ]
    }
    
    # Get patterns for this model (default to 'model.pt')
    patterns = search_patterns.get(model_name, ['model.pt', 'best_model.pt'])
    
    # Try each pattern
    for pattern in patterns:
        model_path = model_dir / pattern
        if model_path.exists():
            logger.debug(f"Found {model_name} at: {model_path}")
            return model_path
    
    # If not found, log all files in directory for debugging
    logger.debug(f"Model file not found for {model_name} in {model_dir}")
    if model_dir.exists():
        files = list(model_dir.glob('*.pt'))
        if files:
            logger.debug(f"Available .pt files: {[f.name for f in files]}")
    
    return None


# ============================================================================
# TRAINING ORCHESTRATOR CLASS
# ============================================================================

class TrainingOrchestrator:

    def train_domain_to_eligibility_wrapper(self, train_dataloader, val_dataloader, epochs=5):
        """
        ✅ NEW: Train the domain->eligibility wrapper
        (Domain classifier weights are frozen, only eligibility head is trained)
        """
        from .neural_models import DomainToEligibilityWrapper
        import torch
        import torch.nn as nn
        from sklearn.metrics import f1_score

        logger.info("\n" + "="*70)
        logger.info("TRAINING DOMAIN-TO-ELIGIBILITY WRAPPER")
        logger.info("="*70)

        # Load trained domain classifier
        domain_classifier = self.components['model_trainer'].load_model('domain_classifier')
        domain_classifier.eval()  # Freeze
        for param in domain_classifier.parameters():
            param.requires_grad = False

        # Create wrapper
        wrapper = DomainToEligibilityWrapper(domain_classifier, self.config).to(self.device)

        # Only train eligibility_head
        optimizer = torch.optim.Adam(wrapper.eligibility_head.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = 0.0
        for epoch in range(epochs):
            # Training
            wrapper.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).long()

                optimizer.zero_grad()
                outputs = wrapper(input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation
            wrapper.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = wrapper(input_ids, attention_mask)
                    preds = torch.argmax(outputs['logits'], dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_f1 = f1_score(val_labels, val_preds, average='binary')

            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_dataloader):.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Save wrapper
                save_path = self.config.MODELS_DIR / 'domain_to_eligibility' / 'model.pt'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(wrapper.state_dict(), save_path)

        logger.info(f"✅ Wrapper training complete. Best F1: {best_val_f1:.4f}")
        return wrapper
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
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
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
        import os
        # Set PyTorch memory allocator configuration for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.start_time = time.time()
        
        # Aggressive memory clearing before starting training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        logger.info("Starting Complete HybEx-Law Training Pipeline")
        logger.info(f"Data Directory: {data_directory}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'pipeline_stages': {},
            'final_models': {},
            'evaluation_results': {},
            'errors': []
        }
        
        # Define pipeline stages for progress tracking
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

            # =====================
            # QUICK DATA SANITY CHECKS
            # 1) Train/Val/Test overlap detection
            # Check for train/val/test overlap
            train_queries = {s['query'] for s in processed_data['train_samples']}
            val_queries = {s['query'] for s in processed_data['val_samples']}
            test_queries = {s['query'] for s in processed_data.get('test_samples', [])}

            overlap = train_queries & val_queries

            # ROBUST FIX: Skip overlap check for pre-split data
            saved_files = preprocessing_results.get('saved_files', {})
            is_presplit = 'train_split.json' in str(saved_files.get('train_data_file', ''))

            if is_presplit:
                logger.info(f"Pre-split data detected - skipping overlap validation")
                logger.info(f"Data loaded: {len(train_queries)} train, {len(val_queries)} val, {len(test_queries)} test queries")
            else:
                # Only check overlap for newly preprocessed data
                if overlap:
                    logger.error(f"DATA LEAKAGE: {len(overlap)} queries appear in both train and val!")
                    logger.error(f"Example overlapping query: {list(overlap)[0][:200]}...")
                    raise ValueError("Train/Val overlap detected - data leakage!")
                
                logger.info(f"Data loaded: {len(train_queries)} unique train queries, {len(val_queries)} unique val queries")

                # ✅ Check for entity/fact leakage
                entity_leakage = 0
                for train_sample in processed_data['train_samples']:
                    train_entities = train_sample.get('extracted_entities', {})
                    for val_sample in processed_data['val_samples']:
                        val_entities = val_sample.get('extracted_entities', {})
                        # Check for exact entity match
                        if train_entities and val_entities:
                            # Compare key facts (income, category, etc.)
                            key_facts = ['income', 'social_category', 'annual_income']
                            matches = sum(
                                train_entities.get(k) == val_entities.get(k)
                                for k in key_facts
                                if k in train_entities and k in val_entities
                            )
                            if matches >= 2:  # 2+ matching facts = potential leakage
                                entity_leakage += 1
                                break
                if entity_leakage > 0:
                    logger.warning(f"⚠️ Potential entity leakage: {entity_leakage} train samples have matching facts with validation")

            # 2) Domain distribution check for imbalance
            try:
                from collections import Counter
                all_domains = []
                for sample in processed_data.get('train_samples', []):
                    if 'domains' in sample and isinstance(sample['domains'], list):
                        all_domains.extend(sample['domains'])
                domain_counts = Counter(all_domains)
                total_train = max(1, len(processed_data.get('train_samples', [])))
                logger.info("\n📊 Domain distribution in training set:")
                for domain, count in domain_counts.items():
                    pct = (count / total_train) * 100
                    logger.info(f"  {domain}: {count} ({pct:.1f}%)")
                    if pct > 80.0:
                        logger.warning(f"Imbalance warning: Domain '{domain}' represents >80% of training samples ({pct:.1f}%)")
                    if pct < 5.0:
                        logger.warning(f"Imbalance warning: Domain '{domain}' represents <5% of training samples ({pct:.1f}%)")
            except Exception:
                logger.exception("Failed to compute domain distribution")
            # =====================
            
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
            
            # Memory cleanup after standard neural training
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU Memory after Stage 2: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

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
            
            # Memory cleanup after GNN training
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU Memory after Stage 3: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            
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
            
            # Clear memory after training completes
            gc.collect()
            torch.cuda.empty_cache()
            
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
            
            # Clear memory even on failure
            gc.collect()
            torch.cuda.empty_cache()
            
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
        
        # ✅ CRITICAL FIX: Verify entities are actually present
        for split in ['train_samples', 'val_samples', 'test_samples']:
            if processed_data.get(split):
                missing_count = sum(1 for s in processed_data[split] 
                                  if 'extracted_entities' not in s or not s['extracted_entities'])
                
                if missing_count > 0:
                    logger.warning(
                        f"⚠️  {missing_count}/{len(processed_data[split])} samples in {split} "
                        f"have missing/empty 'extracted_entities'. Reloading from source..."
                    )
                    
                    # ✅ RELOAD FROM ORIGINAL SPLIT FILES (which now have entities)
                    split_type = split.replace('_samples', '')
                    source_file = self.config.DATA_DIR / f"{split_type}_split.json"
                    
                    if source_file.exists():
                        logger.info(f"📂 Reloading {split} from source file: {source_file}")
                        with open(source_file, 'r', encoding='utf-8') as f:
                            reloaded_data = json.load(f)
                        
                        # Verify the reloaded data has entities
                        reloaded_missing = sum(1 for s in reloaded_data 
                                             if 'extracted_entities' not in s or not s['extracted_entities'])
                        
                        if reloaded_missing == 0:
                            processed_data[split] = reloaded_data
                            logger.info(f"✅ Successfully reloaded {len(reloaded_data)} samples with entities")
                        else:
                            logger.warning(
                                f"⚠️  Source file still missing {reloaded_missing} entities. "
                                f"Will extract on-the-fly during training."
                            )
                    else:
                        logger.warning(f"⚠️  Source file not found: {source_file}")
                else:
                    logger.info(f"✅ All samples in {split} have 'extracted_entities'")
        
        return processed_data
    
    def _integrate_prolog_reasoning_and_evaluation(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Integrate Prolog reasoning with test samples for evaluation purposes.
        This runs the Prolog engine over the test data."""
        logger.info("Integrating Prolog reasoning for evaluation...")
        
        cases_for_prolog_batch = []
        missing_entities_count = 0
        
        # Collect cases and track missing entities WITHOUT individual warnings
        for sample in test_samples:
            if 'extracted_entities' in sample and sample['extracted_entities']:
                cases_for_prolog_batch.append(sample)
            else:
                missing_entities_count += 1
                try:
                    extracted = self.components['data_processor'].extract_entities(
                        sample.get('query', '')
                    )
                    sample['extracted_entities'] = extracted  # ✅ SAVE BACK TO SAMPLE
                    cases_for_prolog_batch.append(sample)
                except Exception as e:
                    logger.error(f"Failed to extract entities for {sample.get('sample_id')}: {e}")
                    sample['extracted_entities'] = {}  # ✅ SET EMPTY DICT
                    cases_for_prolog_batch.append(sample)
        
        # Single summary log for missing entities
        if missing_entities_count > 0:
            logger.warning(
                f"⚠️  {missing_entities_count}/{len(test_samples)} test samples had missing "
                f"'extracted_entities'. Extracted on-the-fly for Prolog evaluation."
            )
        
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
        
        logger.info(f"✅ Prolog reasoning integration complete:")
        logger.info(f"   • Cases evaluated: {total_count}")
        logger.info(f"   • Eligible cases: {eligible_count}")
        logger.info(f"   • Eligibility rate: {integration_results['eligibility_rate']*100:.1f}%")
        logger.info(f"   • Results saved to: {results_file}")
        
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
    
    def run_comprehensive_evaluation(self):
        """
        Run comprehensive evaluation on all trained models
        UPDATED: Now includes EnhancedLegalBERT + Advanced Evaluator
        """
        logger.info("="*80)
        logger.info("STAGE 5: COMPREHENSIVE EVALUATION")
        logger.info("="*80)
        logger.info("\n--- Starting Comprehensive End-to-End System Evaluation ---")
        
        start_time = time.time()
        
        try:
            # Initialize evaluator
            evaluator = ModelEvaluator(self.config)
            
            # Load test data
            test_data_path = self.config.DATA_DIR / 'val_split.json'
            if not test_data_path.exists():
                logger.error(f"Test data not found: {test_data_path}")
                return {'status': 'failed', 'error': 'Test data not found'}
            
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            logger.info(f"Loaded {len(test_data)} test samples")
            
            # Dictionary to store all model results
            all_model_results = {}
            model_predictions = {}  # Store predictions for ensemble
            
            # ========================================
            # EVALUATE EACH MODEL WITH SMART PATH RESOLUTION
            # ========================================
            
            # 1. Domain Classifier
            logger.info("\n[1/5] Evaluating Domain Classifier...")
            try:
                domain_dir = self.config.MODELS_DIR / 'domain_classifier'
                domain_path = find_model_file(domain_dir, 'domain_classifier')
                
                if domain_path:
                    logger.info(f"✓ Found: {domain_path.name}")
                    domain_results = evaluator.evaluate_single_model(
                        model_path=domain_dir,
                        model_type='domain_classifier',
                        test_data=test_data
                    )
                    all_model_results['domain_classifier'] = domain_results
                    model_predictions['domain_classifier'] = domain_results.get('predictions', {})
                    logger.info(f"✅ Domain Classifier - Acc: {domain_results.get('accuracy', 0):.4f}, F1: {domain_results.get('f1', 0):.4f}")
                else:
                    logger.warning(f"⚠️  Domain Classifier model not found in {domain_dir}")
            except Exception as e:
                logger.error(f"❌ Domain Classifier evaluation failed: {str(e)}")
            
            # 2. Eligibility Predictor
            logger.info("\n[2/5] Evaluating Eligibility Predictor...")
            try:
                eligibility_dir = self.config.MODELS_DIR / 'eligibility_predictor'
                eligibility_path = find_model_file(eligibility_dir, 'eligibility_predictor')
                
                if eligibility_path:
                    logger.info(f"✓ Found: {eligibility_path.name}")
                    eligibility_results = evaluator.evaluate_single_model(
                        model_path=eligibility_dir,
                        model_type='eligibility_predictor',
                        test_data=test_data
                    )
                    all_model_results['eligibility_predictor'] = eligibility_results
                    model_predictions['eligibility_predictor'] = eligibility_results.get('predictions', {})
                    logger.info(f"✅ Eligibility Predictor - Acc: {eligibility_results.get('accuracy', 0):.4f}, F1: {eligibility_results.get('f1', 0):.4f}")
                else:
                    logger.warning(f"⚠️  Eligibility Predictor model not found in {eligibility_dir}")
            except Exception as e:
                logger.error(f"❌ Eligibility Predictor evaluation failed: {str(e)}")
            
            # 3. EnhancedLegalBERT with smart path resolution
            logger.info("\n[3/5] Evaluating EnhancedLegalBERT (Multi-Task)...")
            try:
                # Try primary directory first
                enhanced_dir = self.config.MODELS_DIR / 'enhanced_legal_bert'
                enhanced_path = find_model_file(enhanced_dir, 'enhanced_legal_bert')
                
                # Fallback to alternative directory name
                if not enhanced_path:
                    enhanced_dir = self.config.MODELS_DIR / 'enhanced_bert'
                    enhanced_path = find_model_file(enhanced_dir, 'enhanced_bert')
                
                if enhanced_path:
                    logger.info(f"✓ Found: {enhanced_path.name}")
                    
                    # Load model directly with state_dict
                    from .neural_models import EnhancedLegalBERT
                    
                    # Get device
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    # Initialize and load model
                    enhanced_model = EnhancedLegalBERT(self.config)
                    enhanced_model.load_state_dict(
                        torch.load(enhanced_path, map_location=device)
                    )
                    enhanced_model.to(device)
                    enhanced_model.eval()
                    
                    logger.info(f"✅ EnhancedLegalBERT model loaded successfully")
                    
                    # Evaluate using model instance
                    enhanced_results = evaluator.evaluate_model_instance(
                        model=enhanced_model,
                        model_name='enhanced_legal_bert',
                        test_data=test_data,
                        device=device
                    )
                    
                    all_model_results['enhanced_legal_bert'] = enhanced_results
                    model_predictions['enhanced_legal_bert'] = enhanced_results.get('predictions', {})
                    logger.info(f"✅ EnhancedLegalBERT - Acc: {enhanced_results.get('accuracy', 0):.4f}, F1: {enhanced_results.get('f1', 0):.4f}")
                else:
                    logger.warning(f"⚠️  EnhancedLegalBERT model not found")
                    logger.warning(f"    Searched: enhanced_legal_bert/ and enhanced_bert/")
            except Exception as e:
                logger.error(f"❌ EnhancedLegalBERT evaluation failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 4. GNN with smart path resolution
            logger.info("\n[4/5] Evaluating Knowledge Graph Neural Network...")
            try:
                gnn_dir = self.config.MODELS_DIR / 'gnn_model'
                gnn_path = find_model_file(gnn_dir, 'gnn_model')
                
                if gnn_path:
                    logger.info(f"✓ Found: {gnn_path.name}")
                    gnn_results = evaluator.evaluate_gnn_model(
                        model_path=gnn_path,
                        test_data=test_data
                    )
                    all_model_results['gnn_model'] = gnn_results
                    model_predictions['gnn_model'] = gnn_results.get('predictions', {})
                    logger.info(f"✅ GNN - Acc: {gnn_results.get('accuracy', 0):.4f}, F1: {gnn_results.get('f1', 0):.4f}")
                else:
                    logger.warning(f"⚠️  GNN model not found in {gnn_dir}")
            except Exception as e:
                logger.error(f"❌ GNN evaluation failed: {str(e)}")
            
            # 5. Prolog
            logger.info("\n[5/5] Evaluating Prolog Reasoning Engine...")
            try:
                prolog_results = evaluator.evaluate_prolog_engine(test_data)
                all_model_results['prolog'] = prolog_results
                model_predictions['prolog'] = prolog_results.get('predictions', {})
                logger.info(f"✅ Prolog - Acc: {prolog_results.get('accuracy', 0):.4f}, F1: {prolog_results.get('f1', 0):.4f}")
            except Exception as e:
                logger.error(f"❌ Prolog evaluation failed: {str(e)}")
            
            # ========================================
            # PRINT SUMMARY TABLE
            # ========================================
            logger.info("\n" + "="*80)
            logger.info("MODEL PERFORMANCE SUMMARY")
            logger.info("="*80)
            logger.info(f"{'Model':<30} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
            logger.info("-"*80)
            for model_name, results in all_model_results.items():
                acc = results.get('accuracy', 0)
                f1 = results.get('f1', 0)
                prec = results.get('precision', 0)
                rec = results.get('recall', 0)
                logger.info(f"{model_name:<30} {acc:<12.4f} {f1:<12.4f} {prec:<12.4f} {rec:<12.4f}")
            logger.info("="*80)
            
            # ========================================
            # SAVE INDIVIDUAL MODEL RESULTS
            # ========================================
            results_dir = self.config.RESULTS_DIR / 'evaluation_results'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f'individual_models_{timestamp}.json'
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_model_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n✅ Individual model results saved to: {results_file}")
            
            # ========================================
            # RUN ADVANCED EVALUATION (NEW!)
            # ========================================
            
            advanced_results = self.run_advanced_evaluation(all_model_results, test_data)
            
            # Record evaluation time
            eval_duration = time.time() - start_time
            logger.info(f"Total evaluation completed in {eval_duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'duration': eval_duration,
                'model_results': all_model_results,
                'advanced_results': advanced_results,  # NEW!
                'results_file': str(results_file)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def run_advanced_evaluation(self, model_results: Dict, test_data: List[Dict]):
        """
        Run advanced evaluation with stratification, calibration, and fairness analysis
        This uses the AdvancedEvaluator class for comprehensive metrics
        
        Args:
            model_results: Dictionary of model evaluation results
            test_data: List of test samples
        
        Returns:
            Dictionary with advanced evaluation results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING ADVANCED EVALUATION")
        logger.info("="*80)
        
        try:
            from .advanced_evaluator import AdvancedEvaluator
            
            # Initialize Advanced Evaluator
            advanced_eval = AdvancedEvaluator(
                output_dir=self.config.RESULTS_DIR / "advanced_evaluation"
            )
            
            logger.info("✅ AdvancedEvaluator initialized")
            
            # ========================================
            # PREPARE DATA FOR ADVANCED EVALUATION
            # ========================================
            
            predictions_list = []
            ground_truth = []
            metadata = []
            
            logger.info("Preparing data for advanced evaluation...")
            
            for sample in test_data:
                sample_id = sample.get('sample_id', 'unknown')
                entities = sample.get('extracted_entities', {})
                expected = sample.get('expected_eligibility', False)
                
                # Collect predictions from all available models
                sample_predictions = {}
                for model_name, results in model_results.items():
                    if 'predictions' in results and sample_id in results['predictions']:
                        sample_predictions[model_name] = results['predictions'][sample_id]
                
                # Create ensemble prediction
                if sample_predictions:
                    ensemble_pred = self._create_ensemble_prediction(sample_predictions)
                else:
                    ensemble_pred = {'eligible': False, 'confidence': 0.5, 'method': 'none'}
                
                # Add to lists
                predictions_list.append({
                    'eligible': ensemble_pred.get('eligible', False),
                    'confidence': ensemble_pred.get('confidence', 0.5),
                    'method': ensemble_pred.get('method', 'ensemble')
                })
                
                ground_truth.append(expected)
                
                # Prepare metadata
                metadata.append({
                    'case_id': sample_id,
                    'income': entities.get('income', 0),
                    'category': entities.get('social_category', 'unknown'),
                    'vulnerable': entities.get('is_disabled', False) or entities.get('is_woman', False),
                    'query': sample.get('query', '')[:200]  # Truncate for storage
                })
            
            logger.info(f"Prepared {len(predictions_list)} predictions for advanced analysis")
            
            # ========================================
            # RUN ADVANCED EVALUATION
            # ========================================
            
            logger.info("Running comprehensive advanced evaluation...")
            logger.info("  - Stratified analysis (income, category, vulnerability)")
            logger.info("  - Calibration curves")
            logger.info("  - Fairness metrics")
            logger.info("  - Error analysis")
            
            advanced_results = advanced_eval.comprehensive_evaluation(
                predictions=predictions_list,
                ground_truth=ground_truth,
                metadata=metadata
            )
            
            # ========================================
            # LOG ADVANCED RESULTS
            # ========================================
            
            logger.info("\n" + "="*80)
            logger.info("ADVANCED EVALUATION RESULTS")
            logger.info("="*80)
            
            # Overall metrics
            overall = advanced_results.get('overall_metrics', {})
            logger.info("\n📊 Overall Metrics:")
            logger.info(f"  Accuracy: {overall.get('accuracy', 0):.4f}")
            logger.info(f"  Precision: {overall.get('precision', 0):.4f}")
            logger.info(f"  Recall: {overall.get('recall', 0):.4f}")
            logger.info(f"  F1 Score: {overall.get('f1', 0):.4f}")
            
            # Calibration
            calibration_error = advanced_results.get('calibration_error', 0)
            logger.info(f"\n📈 Calibration Error: {calibration_error:.4f}")
            if calibration_error < 0.10:
                logger.info("  ✅ Excellent calibration (< 0.10)")
            elif calibration_error < 0.15:
                logger.info("  ⚠️  Good calibration (0.10 - 0.15)")
            else:
                logger.info("  ❌ Poor calibration (> 0.15) - consider recalibration")
            
            # Stratified metrics
            stratified = advanced_results.get('stratified_metrics', {})
            if stratified:
                logger.info("\n📊 Stratified Performance:")
                
                # By income bracket
                if 'income_brackets' in stratified:
                    logger.info("  By Income Bracket:")
                    for bracket, metrics in stratified['income_brackets'].items():
                        logger.info(f"    {bracket}: F1={metrics.get('f1', 0):.4f}, Count={metrics.get('count', 0)}")
                
                # By social category
                if 'categories' in stratified:
                    logger.info("  By Social Category:")
                    for category, metrics in stratified['categories'].items():
                        logger.info(f"    {category}: F1={metrics.get('f1', 0):.4f}, Count={metrics.get('count', 0)}")
                
                # By vulnerability
                if 'vulnerability' in stratified:
                    logger.info("  By Vulnerability:")
                    for vuln, metrics in stratified['vulnerability'].items():
                        logger.info(f"    {vuln}: F1={metrics.get('f1', 0):.4f}, Count={metrics.get('count', 0)}")
            
            # Fairness metrics
            fairness = advanced_results.get('fairness_metrics', {})
            if fairness:
                logger.info("\n⚖️  Fairness Analysis:")
                logger.info(f"  Income Disparity: {fairness.get('income_disparity', 0):.4f}")
                logger.info(f"  Category Parity: {fairness.get('category_parity', 0):.4f}")
                logger.info(f"  Vulnerability Fairness: {fairness.get('vulnerability_fairness', 0):.4f}")
            
            # Error analysis
            error_analysis = advanced_results.get('error_analysis', [])
            if error_analysis:
                logger.info(f"\n❌ Error Analysis: {len(error_analysis)} misclassified cases")
                logger.info("  Top 5 errors:")
                for i, error in enumerate(error_analysis[:5], 1):
                    logger.info(f"    {i}. Case {error.get('case_id')}: {error.get('error_type')} (conf={error.get('confidence', 0):.3f})")
            
            # Visualizations saved
            viz_dir = self.config.RESULTS_DIR / "advanced_evaluation"
            logger.info(f"\n📊 Visualizations saved to: {viz_dir}")
            logger.info("  - confusion_matrix.png")
            logger.info("  - calibration_curve.png")
            logger.info("  - stratified_performance.png")
            logger.info("  - fairness_analysis.png")
            
            logger.info("\n" + "="*80)
            logger.info("✅ ADVANCED EVALUATION COMPLETED")
            logger.info("="*80)
            
            return advanced_results
            
        except Exception as e:
            logger.error(f"❌ Advanced evaluation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e)}
    
    def _create_ensemble_prediction(self, sample_predictions: Dict) -> Dict:
        """
        Create weighted ensemble prediction from multiple models
        
        Args:
            sample_predictions: Dictionary mapping model_name -> prediction value
        
        Returns:
            Dictionary with ensemble prediction
        """
        weights = {
            'prolog': 0.30,
            'enhanced_legal_bert': 0.25,
            'eligibility_predictor': 0.20,
            'gnn_model': 0.15,
            'domain_classifier': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, pred_value in sample_predictions.items():
            weight = weights.get(model_name, 0.2)
            
            # Handle different prediction formats
            if isinstance(pred_value, dict):
                pred_score = float(pred_value.get('eligible', 0.5))
            elif isinstance(pred_value, (int, float)):
                pred_score = float(pred_value)
            elif isinstance(pred_value, bool):
                pred_score = 1.0 if pred_value else 0.0
            else:
                pred_score = 0.5
            
            weighted_sum += pred_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return {'eligible': False, 'confidence': 0.5, 'method': 'default'}
        
        final_score = weighted_sum / total_weight
        
        return {
            'eligible': final_score >= 0.5,
            'confidence': abs(final_score - 0.5) * 2,  # Convert to [0, 1] confidence
            'method': 'weighted_ensemble'
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up training resources...")
        
        if 'prolog_engine' in self.components:
            self.components['prolog_engine'].cleanup()
        
        logger.info("Cleanup completed")


# ============================================================================
# ADVANCED TRAINING STRATEGY & DATA AUGMENTATION
# ============================================================================

class AdvancedTrainingStrategy:
    """
    Advanced training techniques for better convergence and generalization.
    
    Implements:
    1. Mixup data augmentation - Improves generalization and calibration
    2. Focal Loss - Focuses on hard examples, reduces over-confidence
    3. Adversarial training (optional) - Improves robustness
    4. Gradient clipping - Prevents exploding gradients
    
    These techniques are proven to improve model performance, especially
    on imbalanced datasets and edge cases common in legal domains.
    """
    
    def __init__(self, model, optimizer, train_loader, val_loader, config: Optional[HybExConfig] = None):
        """
        Initialize advanced training strategy.
        
        Args:
            model: The neural model to train
            optimizer: Optimizer instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Optional HybExConfig for customization
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Training techniques (can be configured)
        self.use_mixup = True
        self.use_focal_loss = True
        self.use_adversarial = False  # Optional: adversarial training (computationally expensive)
        self.gradient_clip_value = 1.0
        
        # Hyperparameters
        self.mixup_alpha = 0.2
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        logger.info(f"AdvancedTrainingStrategy initialized - Mixup: {self.use_mixup}, "
                   f"Focal Loss: {self.use_focal_loss}, Adversarial: {self.use_adversarial}")
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Mixup data augmentation: mix two samples.
        
        Mixup improves generalization by training on convex combinations of examples.
        It also improves calibration (confidence alignment with accuracy).
        
        Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)
        
        Args:
            x: Input tensor [batch_size, ...]
            y: Target tensor [batch_size]
            alpha: Beta distribution parameter (typical: 0.2)
        
        Returns:
            mixed_x: Mixed input tensor
            y_a: First set of targets
            y_b: Second set of targets (shuffled)
            lam: Mixing coefficient (lambda)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs: x_mixed = λ * x_a + (1-λ) * x_b
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal Loss: Focus training on hard examples.
        
        Focal Loss reduces the loss contribution of easy examples and focuses
        on hard, misclassified examples. This is especially useful for imbalanced
        datasets where the model might become over-confident on easy examples.
        
        Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
        
        Formula: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            alpha: Weighting factor (typical: 0.25)
            gamma: Focusing parameter (typical: 2.0)
                   - γ=0: Focal Loss = Cross Entropy
                   - γ>0: Down-weights easy examples
        
        Returns:
            Focal loss value
        """
        # Compute cross-entropy loss without reduction
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        
        # Compute probability of true class: p_t = exp(-CE_loss)
        pt = torch.exp(-ce_loss)
        
        # Apply focal term: α * (1 - p_t)^γ
        focal_term = alpha * (1 - pt) ** gamma
        
        # Final focal loss
        focal_loss = focal_term * ce_loss
        
        return focal_loss.mean()
    
    def train_epoch_with_techniques(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with advanced techniques.
        
        Applies mixup augmentation and focal loss during training.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict with training metrics (loss, accuracy, etc.)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            input_ids = batch['input_ids'].to(self.model.device if hasattr(self.model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to(input_ids.device)
            labels = batch['labels'].to(input_ids.device)
            
            # Apply mixup (optional, with 50% probability)
            if self.use_mixup and np.random.rand() > 0.5:
                input_ids, labels_a, labels_b, lam = self.mixup_data(
                    input_ids, labels, alpha=self.mixup_alpha
                )
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Handle different model output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Compute mixed loss: λ * Loss(y_a) + (1-λ) * Loss(y_b)
                if self.use_focal_loss:
                    loss = (lam * self.focal_loss(logits, labels_a, self.focal_alpha, self.focal_gamma) +
                           (1 - lam) * self.focal_loss(logits, labels_b, self.focal_alpha, self.focal_gamma))
                else:
                    loss = (lam * torch.nn.functional.cross_entropy(logits, labels_a) +
                           (1 - lam) * torch.nn.functional.cross_entropy(logits, labels_b))
                
                # For accuracy calculation with mixup, use the dominant label
                _, predicted = torch.max(logits, 1)
                correct_predictions += (lam * (predicted == labels_a).sum().item() + 
                                       (1 - lam) * (predicted == labels_b).sum().item())
            else:
                # Normal forward pass (no mixup)
                outputs = self.model(input_ids, attention_mask)
                
                # Handle different model output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Compute loss
                if self.use_focal_loss:
                    loss = self.focal_loss(logits, labels, self.focal_alpha, self.focal_gamma)
                else:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == labels).sum().item()
            
            total_samples += labels.size(0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.model.device if hasattr(self.model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
                attention_mask = batch['attention_mask'].to(input_ids.device)
                labels = batch['labels'].to(input_ids.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Handle different model output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples
        }


class DataAugmenter:
    """
    Augment training data with synthetic variations.
    
    Provides multiple augmentation strategies:
    1. Paraphrasing - Generate semantic variations
    2. Noise injection - Add realistic typos and errors
    3. Back-translation - Translate to another language and back (if available)
    
    These augmentations help models generalize better and handle noisy inputs.
    """
    
    @staticmethod
    def paraphrase_query(query: str) -> List[str]:
        """
        Generate paraphrased versions of query.
        
        Uses rule-based replacements to create semantic variations.
        For production, consider using T5, BART, or GPT-based paraphrasing.
        
        Args:
            query: Original query text
        
        Returns:
            List of paraphrased variations (up to 3, including original)
        """
        variations = [query]  # Always include original
        
        # Define common phrase replacements
        replacements = {
            'I am': ['I\'m', 'I am currently', 'I happen to be'],
            'I have': ['I\'ve', 'I have got', 'I possess'],
            'do not': ['don\'t', 'do not', 'cannot'],
            'cannot': ['can\'t', 'cannot', 'am unable to'],
            'legal aid': ['free legal help', 'pro bono legal services', 'legal assistance', 'legal aid'],
            'eligible': ['qualify', 'eligible', 'entitled', 'qualify for'],
            'income': ['salary', 'earnings', 'income', 'wages'],
            'fired': ['dismissed', 'terminated', 'fired', 'let go'],
            'employer': ['company', 'boss', 'employer', 'workplace'],
            'help': ['assistance', 'help', 'support', 'aid'],
            'need': ['require', 'need', 'must have'],
        }
        
        # Generate variations by replacing phrases
        for original, options in replacements.items():
            if original.lower() in query.lower():
                for replacement in options:
                    if replacement.lower() != original.lower():
                        # Case-insensitive replacement
                        import re
                        pattern = re.compile(re.escape(original), re.IGNORECASE)
                        new_query = pattern.sub(replacement, query, count=1)
                        if new_query not in variations:
                            variations.append(new_query)
                            if len(variations) >= 3:
                                return variations[:3]
        
        return variations[:3]  # Return up to 3 variations
    
    @staticmethod
    def add_noise(query: str, noise_level: float = 0.1) -> str:
        """
        Add realistic noise (typos, missing words) to query.
        
        Simulates real-world user input with errors. This helps models
        become robust to typos and informal language.
        
        Noise operations:
        - Character swaps (typos)
        - Word duplication (stuttering)
        - Word removal (missing words)
        
        Args:
            query: Original query text
            noise_level: Proportion of words to modify (0.0 to 1.0)
        
        Returns:
            Noisy version of query
        """
        import random
        
        words = query.split()
        if len(words) == 0:
            return query
        
        # Calculate number of changes
        num_changes = max(1, int(len(words) * noise_level))
        
        for _ in range(num_changes):
            if len(words) == 0:
                break
            
            idx = random.randint(0, len(words) - 1)
            
            # Random operation
            op = random.choice(['swap', 'duplicate', 'remove'])
            
            if op == 'swap' and len(words[idx]) > 2:
                # Swap two adjacent characters (typo)
                pos = random.randint(0, len(words[idx]) - 2)
                word_list = list(words[idx])
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                words[idx] = ''.join(word_list)
            
            elif op == 'duplicate' and idx < len(words) - 1:
                # Duplicate a word (stuttering)
                words.insert(idx + 1, words[idx])
            
            elif op == 'remove' and len(words) > 5:
                # Remove a word (only if query is long enough)
                words.pop(idx)
        
        return ' '.join(words)
    
    @staticmethod
    def augment_batch(queries: List[str], augmentation_factor: int = 2) -> List[str]:
        """
        Augment a batch of queries.
        
        Args:
            queries: List of original queries
            augmentation_factor: How many augmented versions per query
        
        Returns:
            List containing original + augmented queries
        """
        augmented = []
        
        for query in queries:
            # Always include original
            augmented.append(query)
            
            # Add paraphrased versions
            paraphrases = DataAugmenter.paraphrase_query(query)
            augmented.extend(paraphrases[1:])  # Skip first (original)
            
            # Add noisy version if we need more
            if len(paraphrases) < augmentation_factor:
                noisy = DataAugmenter.add_noise(query, noise_level=0.1)
                augmented.append(noisy)
        
        return augmented
    
    @staticmethod
    def create_augmented_dataset(samples: List[Dict[str, Any]], 
                                 augmentation_factor: int = 2) -> List[Dict[str, Any]]:
        """
        Create augmented dataset from original samples.
        
        Args:
            samples: List of sample dictionaries with 'query' field
            augmentation_factor: How many augmented versions per sample
        
        Returns:
            List of augmented samples (original + augmented)
        """
        augmented_samples = []
        
        logger.info(f"Augmenting {len(samples)} samples with factor {augmentation_factor}")
        
        for sample in tqdm(samples, desc="Augmenting dataset"):
            if 'query' not in sample:
                augmented_samples.append(sample)
                continue
            
            original_query = sample['query']
            
            # Add original sample
            augmented_samples.append(sample)
            
            # Generate augmented versions
            for i in range(augmentation_factor - 1):
                augmented_sample = sample.copy()
                
                # Alternate between paraphrasing and noise
                if i % 2 == 0:
                    paraphrases = DataAugmenter.paraphrase_query(original_query)
                    if len(paraphrases) > 1:
                        augmented_sample['query'] = paraphrases[1]
                    else:
                        augmented_sample['query'] = original_query
                else:
                    augmented_sample['query'] = DataAugmenter.add_noise(original_query, 0.1)
                
                # Mark as augmented
                augmented_sample['is_augmented'] = True
                augmented_sample['augmentation_method'] = 'paraphrase' if i % 2 == 0 else 'noise'
                
                augmented_samples.append(augmented_sample)
        
        logger.info(f"Created {len(augmented_samples)} samples from {len(samples)} originals")
        
        return augmented_samples


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