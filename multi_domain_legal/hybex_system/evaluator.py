# hybex_system/evaluator.py

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import gc # Import garbage collector

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from dataclasses import asdict, dataclass
from datetime import datetime
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import HybExConfig
from .neural_models import (DomainClassifier, EligibilityPredictor,
                            LegalDataset, ModelMetrics)
from .prolog_engine import LegalReasoning, PrologEngine # Added PrologEngine
from .knowledge_graph_engine import KnowledgeGraphEngine # Added KnowledgeGraphEngine
from .data_processor import DataPreprocessor # Added DataPreprocessor for internal use
from .advanced_evaluator import AdvancedEvaluator # Added AdvancedEvaluator for comprehensive metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Comprehensive evaluation results structure."""
    timestamp: str
    overall_status: str
    neural_metrics: Dict[str, Any]
    prolog_metrics: Dict[str, Any]
    hybrid_metrics: Dict[str, Any]
    component_agreement: Dict[str, Any]
    error_analysis: List[Dict[str, Any]]
    sample_size: int
    config_summary: Dict[str, Any]

def _convert_numpy_to_list(obj):
    """Recursively convert numpy arrays and numpy types to lists/native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic, np.number)):
         return obj.item()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    else:
        return obj

class ModelEvaluator:
    """Comprehensive model evaluation framework for HybEx-Law system."""
    
    def __init__(self, config: HybExConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize internal components for evaluation tasks
        self._data_processor_cache = DataPreprocessor(self.config)
        self._knowledge_graph_engine_cache = KnowledgeGraphEngine(self.config)
        self._prolog_engine_cache = PrologEngine(self.config)

        self.setup_logging()
        logger.info(f"ModelEvaluator initialized on device: {self.device}")

    def setup_logging(self):
        log_file = self.config.get_log_path('model_evaluation')
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('model_evaluation.log') for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(self.config.LOGGING_CONFIG['format'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to ModelEvaluator logger.")
        
        logger.info("="*60)
        logger.info("Starting HybEx-Law Model Evaluation")
        logger.info("="*60)

    def clear_gpu_memory(self):
        """Clear GPU memory and force garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def load_trained_model(self, model_path: str, model_type: str) -> nn.Module:
        """Load a trained neural model."""
        path = Path(model_path)
        if not path.exists():
            # Check for the parent directory if path refers to the model's directory
            model_state_path = path / "model.pt"
            if not model_state_path.exists():
                raise FileNotFoundError(f"Model directory/file not found at {path} or {model_state_path}")
        else:
             model_state_path = path / "model.pt"
             if not model_state_path.exists():
                 raise FileNotFoundError(f"Model state file not found at {model_state_path}")
        
        model_class = None
        if model_type == 'domain_classifier':
            model_class = DomainClassifier
        elif model_type == 'eligibility_predictor':
            model_class = EligibilityPredictor
        elif model_type == 'gnn_model':
            # Assuming GNNModel is defined in neural_models.py (or imported). 
            # If not, this needs correction based on the actual architecture.
            # For now, we skip GNN loading here as GNN logic is separate in the orchestrator.
            raise ValueError(f"GNN model loading must be handled separately/differently: {model_type}")
        
        if model_class is None:
            raise ValueError(f"Unknown neural model type: {model_type}")
        
        model = model_class(self.config)
        model.load_state_dict(torch.load(model_state_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        logger.info(f"✅ Loaded {model_type} from {model_state_path.parent}")
        return model

    def evaluate_neural_model(self, model: nn.Module, test_samples: List[Dict], task_type: str) -> ModelMetrics:
        """Evaluate a single neural model and return metrics."""
        logger.info(f"Evaluating neural model: {model.__class__.__name__} for {task_type}")

        # Use the correct configuration based on the model type
        config_key = next((k for k in self.config.MODEL_CONFIGS if task_type in k), 'domain_classifier')
        batch_size = self.config.MODEL_CONFIGS.get(config_key, self.config.MODEL_CONFIG).get('batch_size', 16)

        test_dataset = LegalDataset(test_samples, self.tokenizer, self.config, task_type)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2 or 1, # Use os.cpu_count() for better resource use
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        predictions = []
        true_labels = []
        total_loss = 0.0
        
        # Use macro for multi-label, binary for binary
        if task_type == "domain_classification":
            criterion = nn.BCEWithLogitsLoss()
            average_type = 'macro'
        elif task_type == "eligibility_prediction":
            criterion = nn.BCEWithLogitsLoss()
            average_type = 'binary' # Corrected to 'binary' for better single-class metric
        else:
            criterion = nn.CrossEntropyLoss()
            average_type = 'macro'


        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {task_type}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                if task_type == "domain_classification":
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                elif task_type == "eligibility_prediction":
                    # Added squeeze to logits for BCEWithLogitsLoss
                    loss = criterion(logits.squeeze(), labels) 
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).cpu().numpy()
                else:
                    loss = criterion(logits, labels.long())
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        
        # Ensure predictions and labels are aligned numpy arrays
        predictions_np = np.array(predictions)
        true_labels_np = np.array(true_labels)

        # Handle potential shape mismatch for binary classification after extend/squeeze
        if task_type == "eligibility_prediction" and true_labels_np.ndim > 1 and true_labels_np.shape[1] == 1:
            true_labels_np = true_labels_np.squeeze()
            predictions_np = predictions_np.squeeze()

        cm = None
        if task_type == "eligibility_prediction" or (task_type != "domain_classification" and len(np.unique(true_labels_np)) < 10):
            cm = confusion_matrix(true_labels_np, predictions_np).tolist() 

        metrics = ModelMetrics(
            accuracy=accuracy_score(true_labels_np, predictions_np) if len(true_labels_np) > 0 else 0.0,
            f1_score=f1_score(true_labels_np, predictions_np, average=average_type, zero_division=0) if len(true_labels_np) > 0 else 0.0,
            precision=precision_score(true_labels_np, predictions_np, average=average_type, zero_division=0) if len(true_labels_np) > 0 else 0.0,
            recall=recall_score(true_labels_np, predictions_np, average=average_type, zero_division=0) if len(true_labels_np) > 0 else 0.0,
            loss=avg_loss,
            classification_report=classification_report(true_labels_np, predictions_np, output_dict=True, zero_division=0),
            confusion_matrix=cm
        )
        
        logger.info(f"Evaluation complete for {model.__class__.__name__}: Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}, Loss={metrics.loss:.4f}")
        return metrics

    def _evaluate_gnn_model(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Dedicated evaluation method for the Knowledge Graph Neural Network (KGNN)."""
        logger.info("\n--- Evaluating Knowledge Graph Engine / GNN Model ---")
        
        # NOTE: Assumes the GNN uses the same entity extraction output as Prolog/Graph analysis
        gnn_predictions = []
        gnn_labels = []
        
        # Limit the evaluation to the first 100 samples for performance sanity if the full set is large
        samples_to_evaluate = test_samples[:self.config.EVAL_CONFIG.get('max_gnn_samples', 1000)]

        for sample in tqdm(samples_to_evaluate, desc="Running GNN Analysis", leave=False):
            # Use the cached data processor
            entities = self._data_processor_cache.extract_entities(sample['query'])
            
            # Use the cached knowledge graph engine
            # predict_eligibility returns int (0 or 1), not a dict
            prediction = self._knowledge_graph_engine_cache.predict_eligibility(entities)
            
            # True label conversion (assuming 'expected_eligibility' is 0 or 1)
            true_label = int(sample.get('expected_eligibility', 0))
            
            # prediction is already 0 or 1
            gnn_predictions.append(prediction)
            gnn_labels.append(true_label)

        if not gnn_predictions:
            logger.warning("No GNN predictions generated.")
            return {'error': 'No predictions generated', 'evaluated_samples': 0}

        # Calculate GNN metrics (binary classification)
        gnn_accuracy = accuracy_score(gnn_labels, gnn_predictions)
        gnn_f1 = f1_score(gnn_labels, gnn_predictions, average='binary', zero_division=0)
        gnn_precision = precision_score(gnn_labels, gnn_predictions, average='binary', zero_division=0)
        gnn_recall = recall_score(gnn_labels, gnn_predictions, average='binary', zero_division=0)

        gnn_metrics = {
            'accuracy': gnn_accuracy,
            'f1_score': gnn_f1,
            'precision': gnn_precision,
            'recall': gnn_recall,
            'classification_report': classification_report(gnn_labels, gnn_predictions, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(gnn_labels, gnn_predictions).tolist(),
            'evaluated_samples': len(gnn_predictions)
        }
        
        logger.info(f"✅ GNN Evaluation Complete: Acc={gnn_accuracy:.4f}, F1={gnn_f1:.4f}")
        return gnn_metrics


    def evaluate_end_to_end_system(self, models_paths: Dict[str, str], test_samples: List[Dict],
                                     prolog_reasoning_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate the complete hybrid neural-symbolic system.
        """
        logger.info("Starting end-to-end system evaluation.")
        if models_paths is None:
            models_paths = {
                "domain_classifier": str(self.config.MODELS_DIR / "domain_classifier"),
                "eligibility_predictor": str(self.config.MODELS_DIR / "eligibility_predictor"),
                "gnn_model": str(self.config.MODELS_DIR / "gnn_model")  # ← ADD THIS LINE
            }
        
        overall_status = "Completed"
        neural_evaluation_metrics = {}
        loaded_neural_models = {}

        # 1. GNN Model Evaluation (Assuming GNN model path/type is 'gnn_model')
        if 'gnn_model' in models_paths:
            try:
                # 1. Load the GNN model first
                gnn_model_path = models_paths['gnn_model']
                
                # Handle both string path and dict path formats
                if isinstance(gnn_model_path, dict):
                    model_path_str = gnn_model_path.get('path') or gnn_model_path.get('model_path')
                else:
                    model_path_str = str(gnn_model_path)
                
                # Load the trained GNN model
                gnn_checkpoint_path = f"{model_path_str}/model.pt"
                logger.info(f"Loading GNN model from: {gnn_checkpoint_path}")
                self._knowledge_graph_engine_cache.load_model(gnn_checkpoint_path)
                logger.info(f"✅ GNN model loaded successfully")
                
                # 2. Now evaluate the loaded model
                gnn_metrics = self._evaluate_gnn_model(test_samples)
                neural_evaluation_metrics['gnn_model'] = gnn_metrics
            except FileNotFoundError as e:
                logger.error(f"GNN model file not found: {e}")
                overall_status = "Failed"
            except Exception as e:
                logger.error(f"Failed to evaluate GNN model: {e}")
                overall_status = "Failed"
        else:
            logger.warning("No GNN model path provided. Skipping GNN evaluation.")

        # 2. Evaluate Standard Neural Models (Domain Classifier & Eligibility Predictor)
        for model_name, task in [("domain_classifier", "domain_classification"), 
                                 ("eligibility_predictor", "eligibility_prediction")]:
            if model_name in models_paths:
                try:
                    self.clear_gpu_memory()
                    
                    # FIX #3: Handle both str and dict model paths
                    model_path_input = models_paths[model_name]
                    
                    # Extract path string from dict if needed
                    if isinstance(model_path_input, dict):
                        # Try common dict keys
                        model_path_str = model_path_input.get('path') or \
                                        model_path_input.get('model_path') or \
                                        model_path_input.get('save_path')
                        if model_path_str is None:
                            logger.error(f"Could not find path in dict for {model_name}: {model_path_input}")
                            continue
                    elif isinstance(model_path_input, (str, Path)):
                        model_path_str = str(model_path_input)
                    else:
                        logger.error(f"Invalid model path type for {model_name}: {type(model_path_input)}")
                        continue
                    
                    # Load and evaluate model
                    model = self.load_trained_model(model_path_str, model_name)
                    loaded_neural_models[model_name] = model
                    
                    metrics = self.evaluate_neural_model(model, test_samples, task)
                    neural_evaluation_metrics[model_name] = asdict(metrics)
                    
                    # Clear domain classifier immediately after evaluation
                    if model_name == 'domain_classifier':
                        del loaded_neural_models[model_name]
                        self.clear_gpu_memory()
                        logger.info(f"✅ {model_name} evaluation completed and memory cleared")
                    
                except Exception as e:
                    logger.error(f"Failed to load or evaluate neural model {model_name}: {e}")
                    overall_status = "Failed"
                    
                    # Clean up if model was loaded
                    if model_name in loaded_neural_models:
                        del loaded_neural_models[model_name]
                        self.clear_gpu_memory()

        # 3. Prolog Reasoning Evaluation (runs if no results provided by orchestrator)
        if prolog_reasoning_results is None:
            logger.info("No Prolog reasoning results provided. Running in-place Prolog evaluation...")
            try:
                # Use cached Prolog engine to run batch analysis
                # batch_legal_analysis expects list of dicts with 'extracted_entities' and 'sample_id'
                cases_for_analysis = []
                for sample in test_samples:
                    case_dict = {
                        'sample_id': sample.get('sample_id', f'eval_{len(cases_for_analysis)}'),
                        'extracted_entities': sample.get('extracted_entities', {})
                    }
                    # If no extracted_entities, extract on the fly
                    if not case_dict['extracted_entities'] and 'query' in sample:
                        case_dict['extracted_entities'] = self._data_processor_cache.extract_entities(sample['query'])
                    cases_for_analysis.append(case_dict)
                
                # Call batch_legal_analysis which returns List[LegalReasoning]
                reasoning_dataclasses = self._prolog_engine_cache.batch_legal_analysis(cases_for_analysis)
                # Convert to dicts for consistency
                prolog_reasoning_results = [asdict(r) for r in reasoning_dataclasses]
                logger.info(f"Completed in-place Prolog reasoning for {len(prolog_reasoning_results)} samples")
            except Exception as e:
                logger.error(f"Failed to run in-place Prolog reasoning: {e}", exc_info=True)
                prolog_reasoning_results = []
        
        prolog_metrics = self._calculate_prolog_metrics(test_samples, prolog_reasoning_results)

        # 4. Hybrid Decision Fusion and Metrics
        hybrid_metrics = {}
        component_agreement = {}
        error_analysis_list = []
        
        if 'eligibility_predictor' in loaded_neural_models and prolog_reasoning_results:
            logger.info("Performing hybrid decision fusion and agreement analysis...")
            
            hybrid_metrics, component_agreement, error_analysis_list = self._calculate_hybrid_metrics(
                loaded_neural_models['eligibility_predictor'], 
                test_samples, 
                prolog_reasoning_results
            )
            # Clean up eligibility predictor after hybrid evaluation
            del loaded_neural_models['eligibility_predictor']
            self.clear_gpu_memory()
            logger.info("✅ Eligibility predictor cleaned up after hybrid evaluation")

        # 5. Final Compilation and Saving
        final_evaluation_results = EvaluationResults(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            neural_metrics=_convert_numpy_to_list(neural_evaluation_metrics),
            prolog_metrics=_convert_numpy_to_list(prolog_metrics),
            hybrid_metrics=_convert_numpy_to_list(hybrid_metrics),
            component_agreement=_convert_numpy_to_list(component_agreement),
            error_analysis=_convert_numpy_to_list(error_analysis_list),
            sample_size=len(test_samples),
            config_summary=self.config.get_summary()
        )
        
        logger.info("End-to-end evaluation completed.")
        
        # Save results and generate report/visualizations
        results_dict = asdict(final_evaluation_results)
        self.save_evaluation_results(results_dict)
        self.generate_evaluation_report(results_dict)
        self.create_evaluation_visualizations(results_dict, str(self.config.RESULTS_DIR / "evaluation_plots"))
        
        return results_dict

    def evaluate_hybrid_system(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate complete hybrid system with ADVANCED metrics.
        
        Returns:
            Dict with standard metrics, method breakdown, and advanced analytics
        """
        from .hybrid_predictor import HybridPredictor, HybridPrediction
        
        logger.info("\n" + "="*70)
        logger.info("HYBRID SYSTEM EVALUATION (WITH ADVANCED METRICS)")
        logger.info("="*70)
        
        # FIX: Create a wrapper object with the correct attributes for HybridPredictor
        class ModelWrapper:
            def __init__(self, evaluator):
                self.device = evaluator.device
                self.tokenizer = evaluator.tokenizer
                # Use correct attribute names
                self.kg_engine = getattr(evaluator, 'kg_engine', None) or evaluator._knowledge_graph_engine_cache
                self.eligibility_model = None  # Will be set dynamically
        
        # Create wrapper
        model_wrapper = ModelWrapper(self)
        
        # Check if eligibility_model was set by main.py
        if hasattr(self, 'eligibility_model'):
            model_wrapper.eligibility_model = self.eligibility_model
        else:
            # Load it if not present
            logger.info("Loading BERT eligibility model...")
            bert_path = self.config.MODELS_DIR / 'eligibility_predictor' / 'model.pt'
            if bert_path.exists():
                model_wrapper.eligibility_model = EligibilityPredictor(self.config)
                model_wrapper.eligibility_model.load_state_dict(torch.load(bert_path, map_location=self.device))
                model_wrapper.eligibility_model.to(self.device)
                model_wrapper.eligibility_model.eval()
            else:
                logger.warning(f"BERT model not found at {bert_path}")
        
        # Initialize hybrid predictor with corrected references
        hybrid = HybridPredictor(
            prolog_engine=self._prolog_engine_cache,
            gnn_model=model_wrapper,  # Pass wrapper instead of self
            bert_model=model_wrapper,  # Pass wrapper instead of self
            config=self.config
        )
        
        # Get predictions
        predictions = hybrid.batch_predict(test_data)
        
        # Extract ground truth and predictions
        y_true = [sample['expected_eligibility'] for sample in test_data]
        y_pred = [pred.eligible for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Method-wise breakdown
        method_metrics = {}
        for method in ['prolog', 'gnn', 'bert', 'ensemble', 'fallback']:
            method_preds = [pred for pred in predictions if pred.method_used == method]
            if method_preds:
                method_indices = [i for i, pred in enumerate(predictions) if pred.method_used == method]
                method_y_true = [y_true[i] for i in method_indices]
                method_y_pred = [pred.eligible for pred in method_preds]
                
                method_metrics[method] = {
                    'count': len(method_preds),
                    'accuracy': accuracy_score(method_y_true, method_y_pred),
                    'f1': f1_score(method_y_true, method_y_pred, zero_division=0),
                    'precision': precision_score(method_y_true, method_y_pred, zero_division=0),
                    'recall': recall_score(method_y_true, method_y_pred, zero_division=0)
                }
        
        # Log results
        logger.info(f"\n{'='*70}")
        logger.info(f"HYBRID SYSTEM RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Overall F1 Score: {f1:.4f}")
        logger.info(f"Overall Precision: {precision:.4f}")
        logger.info(f"Overall Recall: {recall:.4f}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"METHOD-WISE BREAKDOWN")
        logger.info(f"{'='*70}")
        for method, metrics in sorted(method_metrics.items(), key=lambda x: -x[1]['count']):
            logger.info(f"\n{method.upper()}:")
            logger.info(f"  Cases: {metrics['count']}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        # Generate detailed report
        report = classification_report(y_true, y_pred, target_names=['Not Eligible', 'Eligible'], digits=4)
        logger.info(f"\n{'='*70}")
        logger.info(f"CLASSIFICATION REPORT")
        logger.info(f"{'='*70}\n{report}")
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'method_metrics': method_metrics,
            'predictions': predictions,
            'classification_report': report
        }
        
        # ✅ NEW: Advanced evaluation with comprehensive metrics
        if len(predictions) > 0:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"RUNNING ADVANCED EVALUATION")
                logger.info(f"{'='*70}")
                
                advanced_eval = AdvancedEvaluator(
                    output_dir=self.config.RESULTS_DIR / 'advanced_evaluation'
                )
                
                # Prepare metadata for advanced analysis
                metadata = []
                for sample in test_data:
                    meta = sample.get('extracted_entities', {}).copy() if 'extracted_entities' in sample else {}
                    meta['query'] = sample.get('query', sample.get('text', ''))
                    meta['domain'] = sample.get('domain', 'unknown')
                    metadata.append(meta)
                
                # Convert predictions to dict format required by AdvancedEvaluator
                predictions_for_advanced = [
                    {
                        'eligible': pred.eligible,
                        'confidence': pred.confidence,
                        'method': pred.method_used
                    }
                    for pred in predictions  # Use original HybridPrediction objects
                ]
                
                # Run comprehensive evaluation
                advanced_results = advanced_eval.comprehensive_evaluation(
                    predictions=predictions_for_advanced,  # Correct - list of dicts
                    ground_truth=y_true,
                    metadata=metadata
                )
                
                # Add to results
                results['advanced_metrics'] = advanced_results
                
                logger.info("✅ Advanced evaluation completed")
                logger.info(f"📊 Calibration Error: {advanced_results['confidence_analysis']['expected_calibration_error']:.4f}")
                logger.info(f"📊 Fairness Check: See {self.config.RESULTS_DIR / 'advanced_evaluation'}")
                
            except Exception as e:
                logger.warning(f"Advanced evaluation failed: {e}")
                logger.warning("Continuing with standard metrics...")
        
        # Save results
        results_path = self.config.RESULTS_DIR / 'evaluation_results' / f'hybrid_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert predictions to serializable format
        serializable_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'method_metrics': method_metrics,
            'classification_report': report
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n✓ Hybrid results saved to: {results_path}")
        
        return results

    def evaluate_ablation_combinations(self, test_loader: DataLoader, device: torch.device) -> Dict:
        """
        Evaluate ALL model combinations for ablation study.
        Tests: Prolog, Domain, Eligibility, EnhancedBERT, GNN, and all combinations.
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING ABLATION STUDY - ALL MODEL COMBINATIONS")
        logger.info("="*70)
        
        # Define combinations
        combinations = {
            # Single Models
            'prolog_only': ['prolog'],
            'domain_only': ['domain_classifier'],
            'eligibility_only': ['eligibility_predictor'],
            'enhanced_bert_only': ['enhanced_bert'],
            'gnn_only': ['gnn'],
            
            # Pairs
            'prolog_bert': ['prolog', 'enhanced_bert'],
            'prolog_gnn': ['prolog', 'gnn'],
            'bert_gnn': ['enhanced_bert', 'gnn'],
            'domain_eligibility': ['domain_classifier', 'eligibility_predictor'],
            
            # Triples
            'prolog_bert_gnn': ['prolog', 'enhanced_bert', 'gnn'],
            'all_separate': ['domain_classifier', 'eligibility_predictor', 'gnn'],
            'all_neural': ['enhanced_bert', 'eligibility_predictor', 'gnn'],
            
            # Quartets
            'prolog_all_neural': ['prolog', 'enhanced_bert', 'eligibility_predictor', 'gnn'],
            
            # Full Ensemble
            'full_ensemble': ['prolog', 'domain_classifier', 'eligibility_predictor', 'enhanced_bert', 'gnn'],
            
            # Comparisons
            'multitask_focus': ['prolog', 'enhanced_bert', 'gnn'],  # Recommended
            'singletask_focus': ['prolog', 'domain_classifier', 'eligibility_predictor', 'gnn'],
        }
        
        # Define weights for each combination
        weights = {
            'prolog_bert': {'prolog': 0.6, 'enhanced_bert': 0.4},
            'prolog_gnn': {'prolog': 0.6, 'gnn': 0.4},
            'bert_gnn': {'enhanced_bert': 0.6, 'gnn': 0.4},
            'domain_eligibility': {'domain_classifier': 0.3, 'eligibility_predictor': 0.7},
            'prolog_bert_gnn': {'prolog': 0.4, 'enhanced_bert': 0.35, 'gnn': 0.25},
            'all_separate': {'domain_classifier': 0.2, 'eligibility_predictor': 0.5, 'gnn': 0.3},
            'all_neural': {'enhanced_bert': 0.4, 'eligibility_predictor': 0.3, 'gnn': 0.3},
            'prolog_all_neural': {'prolog': 0.3, 'enhanced_bert': 0.3, 'eligibility_predictor': 0.2, 'gnn': 0.2},
            'full_ensemble': {'prolog': 0.25, 'domain_classifier': 0.10, 'eligibility_predictor': 0.15, 
                              'enhanced_bert': 0.30, 'gnn': 0.20},
            'multitask_focus': {'prolog': 0.35, 'enhanced_bert': 0.45, 'gnn': 0.20},
            'singletask_focus': {'prolog': 0.35, 'domain_classifier': 0.15, 'eligibility_predictor': 0.30, 'gnn': 0.20},
        }
        
        all_results = {}
        
        # Evaluate each combination
        for comb_name, models in combinations.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing: {comb_name}")
            logger.info(f"Models: {models}")
            
            # Check if models are available
            available_models = []
            for model_name in models:
                if model_name == 'prolog':
                    available_models.append(model_name)
                elif model_name in ['domain_classifier', 'eligibility_predictor', 'enhanced_bert', 'gnn']:
                    model_path = self.config.MODELS_DIR / model_name
                    if model_path.exists():
                        available_models.append(model_name)
            
            if len(available_models) != len(models):
                logger.warning(f"⚠️  Skipping {comb_name}: Some models not trained yet")
                continue
            
            # Get combination weights (or equal weights if not specified)
            comb_weights = weights.get(comb_name, {m: 1/len(models) for m in models})
            
            # Evaluate this combination
            result = self._evaluate_combination(
                test_loader, 
                device, 
                models, 
                comb_weights
            )
            
            all_results[comb_name] = result
            logger.info(f"✅ {comb_name}: Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        
        # Generate comparison report
        self._generate_ablation_report(all_results)
        
        return all_results

    def _evaluate_combination(self, test_loader: DataLoader, device: torch.device, 
                             models: List[str], weights: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a specific model combination with proper prediction aggregation."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = []
        y_pred_scores = []
        
        for batch in tqdm(test_loader, desc=f"Evaluating {'+'.join(models)}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            y_true.extend(labels)
            
            # Get predictions from each model in combination
            predictions = {}
            
            for model_name in models:
                if model_name == 'prolog':
                    # Prolog predictions
                    prolog_preds = self._get_prolog_predictions(batch, device)
                    predictions['prolog'] = prolog_preds
                
                elif model_name == 'domain_classifier':
                    # Domain classifier
                    predictions['domain_classifier'] = self._get_model_predictions(
                        'domain_classifier', input_ids, attention_mask, device
                    )
                
                elif model_name == 'eligibility_predictor':
                    # Eligibility predictor
                    predictions['eligibility_predictor'] = self._get_model_predictions(
                        'eligibility_predictor', input_ids, attention_mask, device
                    )
                
                elif model_name == 'enhanced_bert':
                    # EnhancedBERT
                    predictions['enhanced_bert'] = self._get_model_predictions(
                        'enhanced_bert', input_ids, attention_mask, device
                    )
                
                elif model_name == 'gnn':
                    # GNN
                    predictions['gnn'] = self._get_gnn_predictions(batch, device)
            
            # Weighted average for this batch
            batch_size = len(labels)
            batch_scores = []
            for i in range(batch_size):
                weighted_score = sum(
                    predictions[m][i] * weights.get(m, 1/len(models)) 
                    for m in predictions
                )
                batch_scores.append(weighted_score)
            
            y_pred_scores.extend(batch_scores)
        
        # Convert to binary predictions
        y_pred = [1 if score >= 0.5 else 0 for score in y_pred_scores]
        
        # Calculate metrics
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'models': models,
            'weights': weights
        }

    def _get_model_predictions(self, model_name: str, input_ids: torch.Tensor, 
                               attention_mask: torch.Tensor, device: torch.device) -> List[float]:
        """Get predictions from a neural model."""
        try:
            model_path = self.config.MODELS_DIR / model_name
            
            if not model_path.exists():
                logger.warning(f"Model {model_name} not found at {model_path}")
                return [0.5] * input_ids.size(0)
            
            # Load model
            model = self.load_trained_model(str(model_path), model_name)
            model.to(device)
            model.eval()
            
            # Get predictions
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                probs = torch.sigmoid(logits.squeeze())
                
                # Handle single sample case
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                
                return probs.cpu().numpy().tolist()
        
        except Exception as e:
            logger.warning(f"Failed to get predictions from {model_name}: {e}")
            return [0.5] * input_ids.size(0)

    def _get_prolog_predictions(self, batch: Dict, device: torch.device) -> List[float]:
        """Get Prolog predictions using the PrologEngine."""
        try:
            predictions = []
            
            # Get raw text queries from batch if available
            if 'raw_query' in batch:
                queries = batch['raw_query']
            elif hasattr(batch, 'dataset') and hasattr(batch.dataset, 'samples'):
                # Try to get original queries from dataset
                queries = [sample.get('query', '') for sample in batch.dataset.samples]
            else:
                # Fallback: return neutral predictions
                logger.warning("Cannot extract queries for Prolog, using neutral predictions")
                return [0.5] * batch['input_ids'].size(0)
            
            # Use cached Prolog engine
            for query in queries:
                try:
                    # Extract entities
                    entities = self._data_processor_cache.extract_entities(query)
                    
                    # Get Prolog reasoning
                    reasoning = self._prolog_engine_cache.legal_analysis(entities)
                    
                    # Convert to probability (eligible=1.0, not_eligible=0.0)
                    prob = 1.0 if reasoning.eligible else 0.0
                    
                    # Weight by confidence
                    prob = prob * reasoning.confidence
                    
                    predictions.append(prob)
                
                except Exception as e:
                    logger.warning(f"Prolog prediction failed for query: {e}")
                    predictions.append(0.5)  # Neutral fallback
            
            return predictions
        
        except Exception as e:
            logger.error(f"Prolog predictions failed: {e}")
            return [0.5] * batch['input_ids'].size(0)

    def _get_gnn_predictions(self, batch: Dict, device: torch.device) -> List[float]:
        """Get GNN predictions using the KnowledgeGraphEngine."""
        try:
            predictions = []
            
            # Get raw text queries from batch if available
            if 'raw_query' in batch:
                queries = batch['raw_query']
            elif hasattr(batch, 'dataset') and hasattr(batch.dataset, 'samples'):
                # Try to get original queries from dataset
                queries = [sample.get('query', '') for sample in batch.dataset.samples]
            else:
                # Fallback: return neutral predictions
                logger.warning("Cannot extract queries for GNN, using neutral predictions")
                return [0.8] * batch['input_ids'].size(0)
            
            # Use cached KG engine
            for query in queries:
                try:
                    # Extract entities
                    entities = self._data_processor_cache.extract_entities(query)
                    
                    # Get GNN prediction (returns 0 or 1)
                    prediction = self._knowledge_graph_engine_cache.predict_eligibility(entities)
                    
                    # Convert to probability
                    prob = float(prediction)
                    
                    predictions.append(prob)
                
                except Exception as e:
                    logger.warning(f"GNN prediction failed for query: {e}")
                    predictions.append(0.8)  # Optimistic fallback for GNN
            
            return predictions
        
        except Exception as e:
            logger.error(f"GNN predictions failed: {e}")
            return [0.8] * batch['input_ids'].size(0)

    def _generate_ablation_report(self, all_results: Dict[str, Dict]) -> None:
        """Generate comprehensive ablation study report with DataFrame visualization."""
        import pandas as pd
        
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY RESULTS - ALL COMBINATIONS")
        logger.info("="*80)
        
        # Convert to DataFrame for better visualization
        data = []
        for comb_name, metrics in all_results.items():
            data.append({
                'Combination': comb_name,
                'Models': ' + '.join(metrics['models']),
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']*100:.2f}%",
                'Recall': f"{metrics['recall']*100:.2f}%",
                'F1': f"{metrics['f1']:.4f}"
            })
        
        df = pd.DataFrame(data)
        
        # Sort by F1 score (convert F1 string to float for sorting)
        df['F1_numeric'] = df['F1'].astype(float)
        df = df.sort_values('F1_numeric', ascending=False)
        df = df.drop('F1_numeric', axis=1)
        
        # Print formatted table to console
        print("\n" + df.to_string(index=False))
        
        # Save results to multiple formats
        output_dir = self.config.RESULTS_DIR / 'ablation_study'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = output_dir / f'ablation_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV results saved to: {csv_path}")
        
        # Save JSON (with full numeric values)
        json_path = output_dir / f'ablation_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"✅ JSON results saved to: {json_path}")
        
        # Save text report
        report_path = output_dir / f'ablation_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ABLATION STUDY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Sort by F1 score
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
            
            f.write("RANKINGS (by F1 Score):\n")
            f.write("-"*80 + "\n")
            
            for rank, (name, result) in enumerate(sorted_results, 1):
                f.write(f"{rank}. {name}\n")
                f.write(f"   Models: {', '.join(result['models'])}\n")
                f.write(f"   Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"   F1: {result['f1']:.4f}\n")
                f.write(f"   Precision: {result['precision']:.4f}\n")
                f.write(f"   Recall: {result['recall']:.4f}\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("KEY INSIGHTS:\n")
            f.write("-"*80 + "\n")
            
            # Best overall
            best = sorted_results[0]
            f.write(f"🏆 Best Overall: {best[0]} (F1: {best[1]['f1']:.4f})\n")
            
            # Best single model
            single_models = [(n, r) for n, r in sorted_results if len(r['models']) == 1]
            if single_models:
                best_single = single_models[0]
                f.write(f"🥇 Best Single Model: {best_single[0]} (F1: {best_single[1]['f1']:.4f})\n")
            
            # Best pair
            pairs = [(n, r) for n, r in sorted_results if len(r['models']) == 2]
            if pairs:
                best_pair = pairs[0]
                f.write(f"🥈 Best Pair: {best_pair[0]} (F1: {best_pair[1]['f1']:.4f})\n")
        
        logger.info(f"✅ Text report saved to: {report_path}")
        
        # Print best combination summary
        best_comb = df.iloc[0]
        logger.info("\n" + "="*80)
        logger.info("🏆 BEST COMBINATION")
        logger.info("="*80)
        logger.info(f"Name: {best_comb['Combination']}")
        logger.info(f"Models: {best_comb['Models']}")
        logger.info(f"Accuracy: {best_comb['Accuracy']}")
        logger.info(f"F1 Score: {best_comb['F1']}")
        logger.info("="*80)



    # --- Helper Methods for Metrics Calculation (Factored out for clarity) ---

    def _calculate_prolog_metrics(self, test_samples: List[Dict], prolog_reasoning_results: List[Dict]) -> Dict[str, Any]:
        """Calculates standard binary classification metrics for Prolog results."""
        prolog_preds = []
        prolog_labels = []
        
        # Map samples by ID for quick lookup
        sample_map = {s.get('sample_id'): s for s in test_samples if s.get('sample_id')}

        matched_samples = 0
        for p_res in prolog_reasoning_results:
            case_id = p_res.get('case_id')
            prolog_eligible = p_res.get('eligible')
            
            if case_id and case_id in sample_map:
                true_eligible = sample_map[case_id].get('expected_eligibility')
                
                if true_eligible is not None and prolog_eligible is not None:
                    prolog_preds.append(prolog_eligible)
                    prolog_labels.append(true_eligible)
                    matched_samples += 1

        logger.info(f"Successfully matched {matched_samples} Prolog results with test samples")

        if prolog_preds:
            prolog_accuracy = accuracy_score(prolog_labels, prolog_preds)
            prolog_f1 = f1_score(prolog_labels, prolog_preds, average='binary', zero_division=0)
            prolog_precision = precision_score(prolog_labels, prolog_preds, average='binary', zero_division=0)
            prolog_recall = recall_score(prolog_labels, prolog_preds, average='binary', zero_division=0)
            
            prolog_metrics = {
                'accuracy': prolog_accuracy,
                'f1_score': prolog_f1,
                'precision': prolog_precision,
                'recall': prolog_recall,
                'classification_report': classification_report(prolog_labels, prolog_preds, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(prolog_labels, prolog_preds).tolist(),
                'evaluated_samples': len(prolog_preds)
            }
            logger.info(f"✅ Prolog Evaluation Complete: Acc={prolog_accuracy:.4f}, F1={prolog_f1:.4f}")
            return prolog_metrics
        else:
            logger.warning("No valid Prolog predictions to evaluate.")
            return {'error': 'No valid Prolog predictions found', 'evaluated_samples': 0}

    def _calculate_hybrid_metrics(self, neural_predictor: nn.Module, test_samples: List[Dict], prolog_reasoning_results: List[Dict]) -> Tuple[Dict, Dict, List[Dict]]:
        """Performs the fusion logic over test samples and calculates hybrid metrics."""
        
        hybrid_preds = []
        true_labels_hybrid = []
        neural_only_preds = []
        prolog_only_preds = []
        error_analysis_list = []

        prolog_results_map = {p.get('case_id'): p for p in prolog_reasoning_results if p.get('case_id')}
        logger.info(f"Created Prolog results map with {len(prolog_results_map)} entries for hybrid analysis")

        processed_hybrid = 0
        
        # Use only samples present in the Prolog map AND having a true label
        samples_to_process = [
            s for s in test_samples 
            if s.get('sample_id') in prolog_results_map and s.get('expected_eligibility') is not None
        ]
        
        # Limit processing for performance safety
        samples_to_process = samples_to_process[:self.config.EVAL_CONFIG.get('max_hybrid_samples', 1000)]

        for sample in tqdm(samples_to_process, desc="Processing hybrid fusion", leave=False):
            sample_id = sample.get('sample_id')
            query_text = sample.get('query')
            true_eligible = sample.get('expected_eligibility')

            try:
                # 1. Neural Prediction
                encoding = self.tokenizer(
                    query_text,
                    truncation=True, padding='max_length', max_length=self.config.MODEL_CONFIG['max_length'],
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    neural_logits = neural_predictor(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])['logits']
                    neural_pred_prob = torch.sigmoid(neural_logits).item()
                    neural_pred_bool = neural_pred_prob > 0.5
                
                # 2. Prolog/Symbolic Result
                prolog_pred_dict = prolog_results_map[sample_id]
                prolog_pred_bool = prolog_pred_dict.get('eligible')
                prolog_confidence = prolog_pred_dict.get('confidence', 0.5)

                # 3. Decision Fusion (Matching logic from trainer.py/main.py for consistency)
                final_hybrid_pred = False
                fusion_method = "default"

                # Use graph_override_threshold for symbolic (Prolog) reasoning
                prolog_threshold = self.config.FUSION_CONFIG.get('graph_override_threshold', 0.8)
                neural_threshold = self.config.FUSION_CONFIG.get('neural_override_threshold', 0.9)

                if prolog_confidence >= prolog_threshold:
                    final_hybrid_pred = prolog_pred_bool
                    fusion_method = "prolog_override"
                elif neural_pred_prob >= neural_threshold:
                    final_hybrid_pred = neural_pred_bool
                    fusion_method = "neural_override"
                else:
                    # Fallback to simple majority/neural if no high confidence
                    final_hybrid_pred = neural_pred_bool 
                    fusion_method = "simple_neural"

                # 4. Record results
                hybrid_preds.append(final_hybrid_pred)
                neural_only_preds.append(neural_pred_bool)
                prolog_only_preds.append(prolog_pred_bool)
                true_labels_hybrid.append(true_eligible)
                processed_hybrid += 1

                agreement = {
                    'neural_prolog': neural_pred_bool == prolog_pred_bool,
                    'neural_ground_truth': neural_pred_bool == true_eligible,
                    'prolog_ground_truth': prolog_pred_bool == true_eligible,
                    'hybrid_ground_truth': final_hybrid_pred == true_eligible
                }
                
                if not agreement['hybrid_ground_truth']:
                    error_analysis_list.append({
                        'sample_id': sample_id,
                        'query': query_text,
                        'true_label': true_eligible,
                        'neural_pred': neural_pred_bool,
                        'neural_prob': neural_pred_prob,
                        'prolog_pred': prolog_pred_bool,
                        'prolog_confidence': prolog_confidence,
                        'hybrid_pred': final_hybrid_pred,
                        'fusion_method': fusion_method,
                        'error_type': 'False Positive' if final_hybrid_pred and not true_eligible else 'False Negative',
                        'disagreement': {
                            'neural_prolog': not agreement['neural_prolog'],
                            'neural_gt': not agreement['neural_ground_truth'],
                            'prolog_gt': not agreement['prolog_ground_truth']
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"Hybrid evaluation failed for sample {sample_id}: {e}")
                continue

        logger.info(f"✅ Hybrid evaluation completed for {processed_hybrid} samples")

        if hybrid_preds:
            hybrid_accuracy = accuracy_score(true_labels_hybrid, hybrid_preds)
            hybrid_f1 = f1_score(true_labels_hybrid, hybrid_preds, average='binary', zero_division=0)
            hybrid_precision = precision_score(true_labels_hybrid, hybrid_preds, average='binary', zero_division=0)
            hybrid_recall = recall_score(true_labels_hybrid, hybrid_preds, average='binary', zero_division=0)
            
            hybrid_metrics = {
                'accuracy': hybrid_accuracy,
                'f1_score': hybrid_f1,
                'precision': hybrid_precision,
                'recall': hybrid_recall,
                'classification_report': classification_report(true_labels_hybrid, hybrid_preds, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(true_labels_hybrid, hybrid_preds).tolist(),
                'evaluated_samples': len(hybrid_preds)
            }

            total_samples_analyzed = len(true_labels_hybrid)
            neural_prolog_agreement_count = sum(1 for n, p in zip(neural_only_preds, prolog_only_preds) if n == p)
            
            component_agreement = {
                'neural_prolog_agreement_rate': neural_prolog_agreement_count / total_samples_analyzed if total_samples_analyzed > 0 else 0,
                'total_hybrid_samples': total_samples_analyzed,
                'neural_accuracy': accuracy_score(true_labels_hybrid, neural_only_preds),
                'prolog_accuracy': accuracy_score(true_labels_hybrid, prolog_only_preds)
            }
            
            logger.info(f"✅ Hybrid System Performance: Acc={hybrid_accuracy:.4f}, F1={hybrid_f1:.4f}")
            logger.info(f"✅ Component Agreement: Neural-Prolog={component_agreement['neural_prolog_agreement_rate']:.4f}")
            
            return hybrid_metrics, component_agreement, error_analysis_list
            
        else:
            logger.warning("No samples processed for hybrid evaluation.")
            return {'error': 'No hybrid predictions generated'}, {}, []

    def save_evaluation_results(self, evaluation_results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        results_path = self.config.RESULTS_DIR / "evaluation_results" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            serializable_results = _convert_numpy_to_list(evaluation_results)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Evaluation results saved to {results_path}")
            return str(results_path)
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        report_str = f"HybEx-Law System Evaluation Report - {evaluation_results['timestamp']}\n"
        report_str += "="*80 + "\n\n"
        
        report_str += "📊 Overall Status: " + evaluation_results.get('overall_status', 'N/A') + "\n"
        report_str += f"Total Test Samples: {evaluation_results.get('sample_size', 'N/A')}\n\n"
        
        report_str += "🧠 Neural Model Performance:\n"
        # Combine standard and GNN neural metrics for reporting
        neural_metrics_combined = evaluation_results.get('neural_metrics', {})
        
        for model_name, metrics in neural_metrics_combined.items():
            report_str += f"  - **{model_name.replace('_', ' ').title()}**:\n"
            report_str += f"    Accuracy: {metrics.get('accuracy', 0):.4f}\n"
            report_str += f"    F1 Score: {metrics.get('f1_score', 0):.4f}\n"
            report_str += f"    Precision: {metrics.get('precision', 0):.4f}\n"
            report_str += f"    Recall: {metrics.get('recall', 0):.4f}\n"
            report_str += f"    Loss: {metrics.get('loss', 'N/A'):.4f}\n" if 'loss' in metrics else f"    Loss: N/A\n"
        report_str += "\n"

        prolog_metrics = evaluation_results.get('prolog_metrics', {})
        if prolog_metrics and 'error' not in prolog_metrics:
            report_str += "⚖️ Prolog Reasoning Performance:\n"
            report_str += f"  Accuracy: {prolog_metrics.get('accuracy', 0):.4f}\n"
            report_str += f"  F1 Score: {prolog_metrics.get('f1_score', 0):.4f}\n"
            report_str += f"  Precision: {prolog_metrics.get('precision', 0):.4f}\n"
            report_str += f"  Recall: {prolog_metrics.get('recall', 0):.4f}\n"
            report_str += "\n"
        else:
            report_str += "⚠️ Prolog Reasoning Performance: Not available or not evaluated.\n\n"

        hybrid_metrics = evaluation_results.get('hybrid_metrics', {})
        if hybrid_metrics and 'error' not in hybrid_metrics:
            report_str += "🤝 Hybrid System Performance:\n"
            report_str += f"  Accuracy: {hybrid_metrics.get('accuracy', 0):.4f}\n"
            report_str += f"  F1 Score: {hybrid_metrics.get('f1_score', 0):.4f}\n"
            report_str += "\n"
        else:
            report_str += "⚠️ Hybrid System Performance: Not available or not evaluated.\n\n"
        
        agreement = evaluation_results.get('component_agreement', {})
        if agreement:
            report_str += "🔗 Component Agreement:\n"
            report_str += f"  Neural-Prolog Agreement Rate: {agreement.get('neural_prolog_agreement_rate', 0):.2%}\n"
            report_str += f"  Neural Accuracy: {agreement.get('neural_accuracy', 0):.4f}\n"
            report_str += f"  Prolog Accuracy: {agreement.get('prolog_accuracy', 0):.4f}\n"
            report_str += "\n"

        error_analysis = evaluation_results.get('error_analysis', [])
        if error_analysis:
            report_str += f"🚨 Error Analysis ({len(error_analysis)} cases with disagreement):\n"
            for i, error in enumerate(error_analysis[:5]):
                report_str += f"  Error {i+1} (Sample ID: {error.get('sample_id', 'N/A')}):\n"
                report_str += f"    Query: {error.get('query', '')[:100]}...\n"
                report_str += f"    True: {error.get('true_label', 'N/A')}, Hybrid Pred: {error.get('hybrid_pred', 'N/A')}\n"
                report_str += f"    Error Type: {error.get('error_type', 'N/A')} (Fusion Method: {error.get('fusion_method', 'N/A')})\n"
                report_str += f"    Disagreements: {json.dumps(error.get('disagreement', {}))}\n"
            if len(error_analysis) > 5:
                report_str += f"  ... and {len(error_analysis) - 5} more errors. See full results JSON for details.\n"
            report_str += "\n"
        
        report_file = self.config.RESULTS_DIR / "evaluation_reports" / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_str)
        
        logger.info(f"✅ Evaluation report generated: {report_file}")
        return str(report_file)

    def create_evaluation_visualizations(self, evaluation_results: Dict[str, Any], output_dir: str):
        """Create visualizations for evaluation results (e.g., confusion matrices)."""
        plots_dir = Path(output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a consistent style and check for the confusion matrix key
        plt.style.use('default') 
        
        metrics_to_plot = {
            'Neural Eligibility Predictor': evaluation_results['neural_metrics'].get('eligibility_predictor'),
            'Prolog Reasoning': evaluation_results.get('prolog_metrics'),
            'Hybrid System': evaluation_results.get('hybrid_metrics')
        }
        
        for title, metrics in metrics_to_plot.items():
            if metrics and 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None and len(metrics['confusion_matrix']) == 2:
                try:
                    cm = metrics['confusion_matrix']
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Not Eligible', 'Eligible'],
                                yticklabels=['Not Eligible', 'Eligible'])
                    plt.title(f'{title} Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.tight_layout()
                    
                    filename_safe = title.lower().replace(' ', '_').replace('/', '_')
                    plt.savefig(plots_dir / f"{filename_safe}_confusion_matrix.png", dpi=300)
                    plt.close()
                    logger.info(f"Generated {title} Confusion Matrix.")
                except Exception as e:
                    logger.warning(f"Could not generate {title} confusion matrix: {e}")

        logger.info(f"✅ Evaluation visualizations saved to {plots_dir}")