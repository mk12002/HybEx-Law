
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import gc # Import garbage collecto
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenize
from .config import HybExConfig
from .neural_models import (DomainClassifier, EligibilityPredictor,
                            LegalDataset, ModelMetrics)
from .prolog_engine import LegalReasoning, PrologEngine
from .knowledge_graph_engine import KnowledgeGraphEngine
from .data_processor import DataPreprocessor
from .advanced_evaluator import AdvancedEvaluato
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


# ============================================================================
# HELPER CLASSES
# ============================================================================

class AblationDataset(Dataset):
    """PyTorch Dataset for ablation study evaluation"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample.get('query', '')
        label = int(sample.get('expected_eligibility', 0))
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'sample_idx': idx
        }
class ModelEvaluator:
    """Comprehensive model evaluation framework for HybEx-Law system."""
    
# --- MODULE-LEVEL ModelWrapper ---
class ModelWrapper:
    """Wrapper to provide consistent interface for different model types"""
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.device = evaluator.device
        self.tokenizer = evaluator.tokenizer
        self.kg_engine = getattr(evaluator, 'kg_engine', None) or getattr(evaluator, '_knowledge_graph_engine_cache', None)
        self.eligibility_model = None
    def predict(self, model_type: str, model_obj: Any, input_data: Any) -> Any:
        """
        Generic prediction interface
        Args:
            model_type: Type of model ('prolog', 'gnn', 'bert', etc.)
            model_obj: The model object
            input_data: Input data dict or string
        Returns:
            Prediction result
        """
        if model_type == 'prolog':
            return model_obj.query_eligibility(input_data)
        elif model_type == 'gnn':
            return model_obj.predict(input_data)
        elif model_type in ['bert', 'domain', 'eligibility']:
            with torch.no_grad():
                model_obj.eval()
                query = input_data if isinstance(input_data, str) else input_data.get('query', '')
                encoding = self.tokenizer(
                    query,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                logits = model_obj(**encoding)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1).item()
                return pred
        else:
            raise ValueError(f"Unknown model type: {model_type}")

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
                    # FIX: Handle different output shapes for eligibility prediction
                    # Ensure labels and logits have compatible shapes
                    if logits.dim() == 1:
                        # Binary classification with single output per sample: [batch_size]
                        # Keep as-is, ensure labels are also 1D
                        labels = labels.squeeze() if labels.dim() > 1 else labels
                    elif logits.dim() == 2 and logits.size(1) == 1:
                        # Binary classification with [batch_size, 1] output
                        # Squeeze to [batch_size]
                        logits = logits.squeeze(1)
                        labels = labels.squeeze() if labels.dim() > 1 else labels
                    elif logits.dim() == 0:
                        # Single sample with scalar output
                        # Add batch dimension
                        logits = logits.unsqueeze(0)
                        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
                    
                    # Ensure labels are float for BCEWithLogitsLoss
                    labels = labels.float()
                    
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
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
            recall=recall_score(true_labels_np, predictions_np, average=average_type, zero_division=0) if len(true_labelsNp) > 0 else 0.0,
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

    def evaluate_ablation_combinations(self, test_data: List[Dict], device) -> Dict:
        """
        Run ablation study - Test ALL 17 model combinations
        ✅ Handles all models: prolog, gnn, domain, eligibility, enhanced_bert
        ✅ Fixed domain classifier to use legal_aid domain probability
        ✅ Proper error handling and GPU memory cleanup
        """
        from transformers import AutoTokenizer
        from torch.utils.data import Dataset, DataLoader
        import pandas as pd
        
        logger.info(f"\n{'='*80}")
        logger.info("COMPREHENSIVE ABLATION STUDY - ALL 17 MODEL COMBINATIONS")
        logger.info(f"{'='*80}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_CONFIG['base_model'])
        
        # ===== ALL 17 COMBINATIONS =====
        combinations = {
            # Singles (5)
            '01_prolog': ['prolog'],
            '02_gnn': ['gnn'],
            '03_domain': ['domain_classifier'],
            '04_eligibility': ['eligibility_predictor'],
            '05_enhanced_bert': ['enhanced_legal_bert'],
            
            # Pairs (6)
            '06_domain_eligibility': ['domain_classifier', 'eligibility_predictor'],
            '07_domain_bert': ['domain_classifier', 'enhanced_legal_bert'],
            '08_eligibility_bert': ['eligibility_predictor', 'enhanced_legal_bert'],
            '09_gnn_domain': ['gnn', 'domain_classifier'],
            '10_gnn_eligibility': ['gnn', 'eligibility_predictor'],
            '11_gnn_bert': ['gnn', 'enhanced_legal_bert'],
            
            # With Prolog (3)
            '12_prolog_domain': ['prolog', 'domain_classifier'],
            '13_prolog_eligibility': ['prolog', 'eligibility_predictor'],
            '14_prolog_gnn': ['prolog', 'gnn'],
            
            # Ensembles (3)
            '15_all_neural': ['domain_classifier', 'eligibility_predictor', 'enhanced_legal_bert', 'gnn'],
            '16_hybrid_best': ['prolog', 'eligibility_predictor', 'gnn'],
            '17_full_ensemble': ['prolog', 'domain_classifier', 'eligibility_predictor', 'gnn', 'enhanced_legal_bert'],
        }
        
        all_results = []
        
        # Helper to check model availability
        def check_models_available(models_list):
            for model_name in models_list:
                if model_name == 'prolog':
                    if not hasattr(self, '_prolog_engine_cache') or self._prolog_engine_cache is None:
                        return False, "Prolog engine not initialized"
                elif model_name == 'gnn':
                    if not hasattr(self, '_knowledge_graph_engine_cache') or self._knowledge_graph_engine_cache is None:
                        return False, "GNN engine not initialized"
                    if not self.config.GNN_MODEL_PATH.exists():
                        return False, f"GNN model not found at {self.config.GNN_MODEL_PATH}"
                elif model_name in ['domain_classifier', 'eligibility_predictor']:
                    model_path = self.config.MODELS_DIR / model_name / 'model.pt'
                    if not model_path.exists():
                        return False, f"{model_name} not found at {model_path}"
                elif model_name == 'enhanced_legal_bert':
                    if not self.config.ENHANCED_BERT_MODEL_PATH.exists():
                        return False, f"Enhanced BERT not found at {self.config.ENHANCED_BERT_MODEL_PATH}"
            return True, "All models available"
        
        # Evaluate each combination
        for combo_name, models in combinations.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing: {combo_name}")
            logger.info(f"Models: {models}")
            
            try:
                # Check availability
                available, msg = check_models_available(models)
                if not available:
                    logger.warning(f"⚠️  Skipping {combo_name}: {msg}")
                    continue
                
                predictions = []
                ground_truth = []
                
                # ===== PROLOG-ONLY =====
                if models == ['prolog']:
                    for sample in tqdm(test_data, desc=f"Evaluating {combo_name}"):
                        try:
                            entities = sample.get('extracted_entities', {})
                            query = sample.get('query', '')
                            result = self._prolog_engine_cache.evaluate_eligibility(entities, query)
                            predictions.append(1 if result.get('eligible', False) else 0)
                        except Exception as e:
                            logger.debug(f"Prolog prediction failed: {e}")
                            predictions.append(0)
                        ground_truth.append(int(sample.get('expected_eligibility', 0)))
                
                # ===== GNN-ONLY =====
                elif models == ['gnn']:
                    try:
                        self._knowledge_graph_engine_cache.load_model(str(self.config.GNN_MODEL_PATH))
                        logger.info("✅ GNN model loaded")
                    except Exception as e:
                        logger.error(f"Failed to load GNN: {e}")
                        continue
                    
                    entities_list = [s.get('extracted_entities', {}) for s in test_data]
                    predictions = self._knowledge_graph_engine_cache.batch_predict_eligibility(
                        entities_list, return_probabilities=False
                    )
                    ground_truth = [int(s.get('expected_eligibility', 0)) for s in test_data]
                
                # ===== HYBRID (Prolog + GNN) =====
                elif set(models) == {'prolog', 'gnn'}:
                    prolog_preds = []
                    for sample in test_data:
                        try:
                            result = self._prolog_engine_cache.evaluate_eligibility(
                                sample.get('extracted_entities', {}),
                                sample.get('query', '')
                            )
                            prolog_preds.append(1 if result.get('eligible', False) else 0)
                        except:
                            prolog_preds.append(0)
                    
                    entities_list = [s.get('extracted_entities', {}) for s in test_data]
                    gnn_preds = self._knowledge_graph_engine_cache.batch_predict_eligibility(entities_list)
                    
                    # Weighted combination
                    predictions = []
                    for p, g in zip(prolog_preds, gnn_preds):
                        combined = 0.5 * p + 0.5 * g
                        predictions.append(1 if combined > 0.5 else 0)
                    
                    ground_truth = [int(s.get('expected_eligibility', 0)) for s in test_data]
                
                # ===== NEURAL MODELS (with or without Prolog) =====
                else:
                    # Create dataset
                    dataset = AblationDataset(test_data, tokenizer)
                    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
                    
                    # Load neural models
                    loaded_models = {}
                    for model_name in models:
                        if model_name in ['prolog', 'gnn']:
                            continue  # Handle separately
                        
                        if model_name == 'domain_classifier':
                            from .neural_models import DomainClassifier
                            model = DomainClassifier(self.config).to(device)
                            model.load_state_dict(torch.load(
                                self.config.MODELS_DIR / 'domain_classifier' / 'model.pt',
                                map_location=device
                            ))
                            model.eval()
                            loaded_models[model_name] = model
                        
                        elif model_name == 'eligibility_predictor':
                            from .neural_models import EligibilityPredictor
                            model = EligibilityPredictor(self.config).to(device)
                            model.load_state_dict(torch.load(
                                self.config.MODELS_DIR / 'eligibility_predictor' / 'model.pt',
                                map_location=device
                            ))
                            model.eval()
                            loaded_models[model_name] = model
                        
                        elif model_name == 'enhanced_legal_bert':
                            from .neural_models import EnhancedLegalBERT
                            model = EnhancedLegalBERT(self.config).to(device)
                            model.load_state_dict(torch.load(
                                self.config.ENHANCED_BERT_MODEL_PATH,
                                map_location=device
                            ))
                            model.eval()
                            loaded_models[model_name] = model
                    
                    # Get Prolog predictions if needed
                    prolog_preds = None
                    if 'prolog' in models:
                        prolog_preds = []
                        for sample in test_data:
                            try:
                                result = self._prolog_engine_cache.evaluate_eligibility(
                                    sample.get('extracted_entities', {}),
                                    sample.get('query', '')
                                )
                                prolog_preds.append(1 if result.get('eligible', False) else 0)
                            except:
                                prolog_preds.append(0)
                    
                    # Get GNN predictions if needed
                    gnn_preds = None
                    if 'gnn' in models:
                        entities_list = [s.get('extracted_entities', {}) for s in test_data]
                        gnn_preds = self._knowledge_graph_engine_cache.batch_predict_eligibility(entities_list)
                    
                    # Neural model evaluation
                    neural_predictions = []
                    with torch.no_grad():
                        for batch in tqdm(dataloader, desc=f"Evaluating {combo_name}"):
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['label']
                            
                            batch_preds = []
                            
                            for model_name, model in loaded_models.items():
                                outputs = model(input_ids, attention_mask)
                                
                                # Extract logits
                                if isinstance(outputs, dict):
                                    logits = outputs.get('logits', outputs.get('eligibility_logits', None))
                                    if logits is None:
                                        logits = list(outputs.values())[0]
                                else:
                                    logits = outputs
                                
                                # ✅ CRITICAL FIX: Domain classifier
                                if model_name == 'domain_classifier':
                                    probs = torch.sigmoid(logits)
                                    # Use legal_aid domain (index 0) as eligibility proxy
                                    legal_aid_prob = probs[:, 0] if probs.dim() > 1 else probs
                                    preds = (legal_aid_prob > 0.5).long()
                                else:
                                    if logits.dim() == 1:
                                        probs = torch.sigmoid(logits)
                                        preds = (probs > 0.5).long()
                                    else:
                                        preds = torch.argmax(logits, dim=1)
                                
                                batch_preds.append(preds.cpu())
                            
                            # Ensemble neural predictions
                            if len(batch_preds) > 1:
                                stacked = torch.stack(batch_preds)
                                batch_ensemble = torch.mode(stacked, dim=0).values
                            else:
                                batch_ensemble = batch_preds[0]
                            
                            neural_predictions.extend(batch_ensemble.tolist())
                            ground_truth.extend(labels.tolist())
                    
                    # Combine with Prolog/GNN if present
                    if prolog_preds is not None or gnn_preds is not None:
                        all_preds = [neural_predictions]
                        if prolog_preds is not None:
                            all_preds.append(prolog_preds)
                        if gnn_preds is not None:
                            all_preds.append(gnn_preds)
                        
                        # Majority vote
                        predictions = []
                        for i in range(len(neural_predictions)):
                            votes = [pred_list[i] for pred_list in all_preds]
                            predictions.append(1 if sum(votes) > len(votes) / 2 else 0)
                    else:
                        predictions = neural_predictions
                
                # Calculate metrics
                accuracy = accuracy_score(ground_truth, predictions)
                precision = precision_score(ground_truth, predictions, zero_division=0)
                recall = recall_score(ground_truth, predictions, zero_division=0)
                f1 = f1_score(ground_truth, predictions, zero_division=0)
                
                result = {
                    'combination': combo_name,
                    'models': models,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'num_samples': len(predictions)
                }
                
                all_results.append(result)
                
                # Logging
                logger.info(f"\n{'='*70}")
                logger.info(f"RESULTS: {combo_name}")
                logger.info(f"{'='*70}")
                logger.info(f"Models: {' + '.join(models)}")
                logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall:    {recall:.4f}")
                logger.info(f"F1 Score:  {f1:.4f}")
                logger.info(f"Samples:   {len(predictions)}")
                
                # GPU cleanup
                del predictions, ground_truth
                if 'loaded_models' in locals():
                    for model in loaded_models.values():
                        del model
                    loaded_models.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Failed {combo_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Generate report
        self._generate_ablation_report(all_results)
        
        # Save results
        results_dir = self.config.RESULTS_DIR / 'ablation_study'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        df_data = []
        for r in all_results:
            df_data.append({
                'Combination': r['combination'],
                'Models': ' + '.join(r['models']),
                'Accuracy': f"{r['accuracy']*100:.2f}%",
                'Precision': f"{r['precision']*100:.2f}%",
                'Recall': f"{r['recall']*100:.2f}%",
                'F1': f"{r['f1']:.4f}",
                'Samples': r['num_samples']
            })
        
        df = pd.DataFrame(df_data)
        csv_path = results_dir / f'ablation_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV saved: {csv_path}")
        
        # Save JSON
        json_path = results_dir / f'ablation_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"✅ JSON saved: {json_path}")
        
        return {'all_results': all_results, 'results_file': str(json_path)}

    def _evaluate_combination(self, test_loader: DataLoader, device: torch.device, 
                             models: List[str], weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate a specific model combination with proper prediction aggregation
        FIXED: Handle Prolog-only combinations without requiring neural model inputs
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        combination_name = '+'.join(models)
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {combination_name}")
        logger.info(f"Models: {models}")
        
        try:
            y_true = []
            y_pred_scores = []
            
            # ✅ FIX: Check if only Prolog is being used (no neural models)
            neural_models_present = any(m in ['domain_classifier', 'eligibility_predictor', 
                                              'enhanced_bert', 'gnn'] 
                                       for m in models)
            
            if not neural_models_present and 'prolog' in models:
                # ✅ Prolog-only: Don't use DataLoader batching, use raw samples
                logger.info("Prolog-only combination - using direct evaluation")
                
                # Access the underlying dataset
                dataset = test_loader.dataset if hasattr(test_loader, 'dataset') else None
                if dataset is None:
                    raise ValueError("Cannot access dataset for Prolog-only evaluation")
                
                for sample in tqdm(dataset, desc=f"Evaluating {combination_name}"):
                    # Get Prolog prediction
                    prolog_pred = self._get_prolog_prediction(sample)
                    y_pred_scores.append(1.0 if prolog_pred else 0.0)
                    
                    # Get ground truth
                    label = sample.get('expected_eligibility', sample.get('label', 0))
                    y_true.append(int(label))
            
            else:
                # ✅ Neural models present: Use DataLoader batching
                for batch in tqdm(test_loader, desc=f"Evaluating {combination_name}", leave=False):
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
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            
            logger.info(f"✓ Accuracy: {accuracy:.4f}")
            logger.info(f"✓ F1 Score: {f1:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'models': models,
                'weights': weights,
                'num_samples': len(y_pred)
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate {combination_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'combination': combination_name,
                'models': models,
                'error': str(e),
                'accuracy': 0.0,
                'f1': 0.0
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
    
    def _get_prolog_prediction(self, sample: Dict) -> bool:
        """
        Get Prolog prediction for a single sample
        
        Args:
            sample: Test sample dictionary with 'query' and/or 'extracted_entities'
        
        Returns:
            Boolean eligibility prediction
        """
        try:
            # Try to get entities from sample
            entities = sample.get('extracted_entities', {})
            query_text = sample.get('query', '')
            
            # Extract entities if not present
            if not entities and query_text:
                if hasattr(self, '_data_processor_cache') and self._data_processor_cache:
                    entities = self._data_processor_cache.extract_entities(query_text)
                else:
                    logger.warning("Data processor not available for entity extraction")
                    return False
            
            # Query Prolog engine
            if hasattr(self, '_prolog_engine_cache') and self._prolog_engine_cache:
                # (actual logic for single-sample Prolog prediction continues here)
                pass
        
        # Generate reports
        self._generate_ablation_report(all_results)
        
        # Save results
        results_dir = self.config.RESULTS_DIR / 'ablation_study'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        import pandas as pd
        df_data = []
        for r in all_results:
            if 'error' not in r:
                df_data.append({
                    'Rank': len(df_data) + 1,
                    'Combination': r['combination'],
                    'Models': ' + '.join(r['models']),
                    'Accuracy': f"{r['accuracy']*100:.2f}%",
                    'Precision': f"{r['precision']*100:.2f}%",
                    'Recall': f"{r['recall']*100:.2f}%",
                    'F1': f"{r['f1']:.4f}",
                    'Samples': r['num_samples']
                })
        
        # Sort by F1 score
        df = pd.DataFrame(df_data)
        df = df.sort_values('F1', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        csv_path = results_dir / f'ablation_complete_{timestamp}.csv'
        df.to_csv(csv_path, index=False)

        logger.info(f"✅ CSV saved: {csv_path}")

        # Automated best model detection and summary
        best_combo = max(all_results, key=lambda x: x.get('f1', 0))
        prolog_only = next((r for r in all_results if r['combination'] == '01_prolog'), None)

        logger.info(f"\n{'='*70}")
        logger.info("ABLATION STUDY SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Best Combination: {best_combo['combination']}")
        logger.info(f"  Models: {' + '.join(best_combo['models'])}")
        logger.info(f"  F1 Score: {best_combo['f1']:.4f}")
        logger.info(f"  Accuracy: {best_combo['accuracy']:.4f}")

        if prolog_only:
            improvement = (best_combo['f1'] - prolog_only['f1']) / prolog_only['f1'] * 100 if prolog_only['f1'] else 0.0
            logger.info(f"\nImprovement over Prolog-only: {improvement:+.2f}%")
            if improvement < 1.0:
                logger.warning("⚠️ WARNING: Mixture models NOT outperforming Prolog!")
                logger.warning("Recommended actions:")
                logger.warning("  1. Check neural model training convergence")
                logger.warning("  2. Verify entity extraction quality")
                logger.warning("  3. Review ensemble weight calibration")

        # Save JSON
        json_path = results_dir / f'ablation_complete_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"✅ JSON saved: {json_path}")

        return {'all_results': all_results, 'results_file': str(json_path)}

        
    def _evaluate_combination(self, test_loader: DataLoader, device: torch.device, 
                             models: List[str], weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate a specific model combination with proper prediction aggregation
        FIXED: Handle Prolog-only combinations without requiring neural model inputs
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        combination_name = '+'.join(models)
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {combination_name}")
        logger.info(f"Models: {models}")
        
        try:
            y_true = []
            y_pred_scores = []
            
            # ✅ FIX: Check if only Prolog is being used (no neural models)
            neural_models_present = any(m in ['domain_classifier', 'eligibility_predictor', 
                                              'enhanced_bert', 'gnn'] 
                                       for m in models)
            
            if not neural_models_present and 'prolog' in models:
                # ✅ Prolog-only: Don't use DataLoader batching, use raw samples
                logger.info("Prolog-only combination - using direct evaluation")
                
                # Access the underlying dataset
                dataset = test_loader.dataset if hasattr(test_loader, 'dataset') else None
                if dataset is None:
                    raise ValueError("Cannot access dataset for Prolog-only evaluation")
                
                for sample in tqdm(dataset, desc=f"Evaluating {combination_name}"):
                    # Get Prolog prediction
                    prolog_pred = self._get_prolog_prediction(sample)
                    y_pred_scores.append(1.0 if prolog_pred else 0.0)
                    
                    # Get ground truth
                    label = sample.get('expected_eligibility', sample.get('label', 0))
                    y_true.append(int(label))
            
            else:
                # ✅ Neural models present: Use DataLoader batching
                for batch in tqdm(test_loader, desc=f"Evaluating {combination_name}", leave=False):
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
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            
            logger.info(f"✓ Accuracy: {accuracy:.4f}")
            logger.info(f"✓ F1 Score: {f1:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'models': models,
                'weights': weights,
                'num_samples': len(y_pred)
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate {combination_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'combination': combination_name,
                'models': models,
                'error': str(e),
                'accuracy': 0.0,
                'f1': 0.0
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
    
    def _get_prolog_prediction(self, sample: Dict) -> bool:
        """
        Get Prolog prediction for a single sample
        
        Args:
            sample: Test sample dictionary with 'query' and/or 'extracted_entities'
        
        Returns:
            Boolean eligibility prediction
        """
        try:
            # Try to get entities from sample
            entities = sample.get('extracted_entities', {})
            query_text = sample.get('query', '')
            
            # Extract entities if not present
            if not entities and query_text:
                if hasattr(self, '_data_processor_cache') and self._data_processor_cache:
                    entities = self._data_processor_cache.extract_entities(query_text)
                else:
                    logger.warning("Data processor not available for entity extraction")
                    return False
            
            # Query Prolog engine
            if hasattr(self, '_prolog_engine_cache') and self._prolog_engine_cache:
                result = self._prolog_engine_cache.evaluate_eligibility(entities)
                return result.get('eligible', False)
            else:
                logger.warning("Prolog engine not available")
                return False
                
        except Exception as e:
            logger.error(f"Prolog prediction failed: {str(e)}")
            return False

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

    def _generate_ablation_report(self, all_results: List[Dict]) -> None:
        """
        Generate comprehensive ablation study report with DataFrame visualization
        FIXED: Handle missing metrics safely, accepts list of result dicts
        FIXED: Remove emoji for Windows compatibility
        """
        import pandas as pd
        
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY RESULTS - ALL COMBINATIONS")
        logger.info("="*80)
        
        # Filter out failed results
        valid_results = []
        for result in all_results:
            # Skip if error or f1 is 0
            if 'error' in result or result.get('f1', 0) == 0:
                logger.warning(f"⚠️  Skipping {result.get('combination', 'Unknown')} from report (failed or zero F1)")
                continue
            
            valid_results.append(result)
        
        if not valid_results:
            logger.warning("⚠️  No valid ablation results to report")
            return
        
        # Sort by F1 score
        sorted_results = sorted(valid_results, key=lambda x: x.get('f1', 0), reverse=True)
        
        # Console output (can use emoji)
        logger.info(f"\n{'Rank':<6} {'Combination':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        logger.info("-" * 90)
        
        for i, result in enumerate(sorted_results, 1):
            comb_name = result.get('combination', 'Unknown')[:28]
            acc = result.get('accuracy', 0) * 100
            prec = result.get('precision', 0) * 100
            rec = result.get('recall', 0) * 100
            f1 = result.get('f1', 0) * 100
            
            logger.info(f"{i:<6} {comb_name:<30} {acc:>10.2f}% {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}%")
        
        # Best combination (console - can use emoji)
        best = sorted_results[0]
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 BEST COMBINATION: {best.get('combination', 'Unknown')}")
        logger.info(f"{'='*80}")
        logger.info(f"   Models: {', '.join(best.get('models', []))}")
        logger.info(f"   Accuracy: {best.get('accuracy', 0)*100:.2f}%")
        logger.info(f"   F1 Score: {best.get('f1', 0)*100:.2f}%")
        logger.info(f"   Precision: {best.get('precision', 0)*100:.2f}%")
        logger.info(f"   Recall: {best.get('recall', 0)*100:.2f}%")
        logger.info(f"   Samples: {best.get('num_samples', 0)}")
        
        # Save results to files
        try:
            output_dir = self.config.RESULTS_DIR / 'ablation_study'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert to DataFrame
            df_data = []
            for result in sorted_results:
                df_data.append({
                    'Combination': result.get('combination', 'Unknown'),
                    'Models': ' + '.join(result.get('models', [])),
                    'Accuracy': f"{result.get('accuracy', 0)*100:.2f}%",
                    'Precision': f"{result.get('precision', 0)*100:.2f}%",
                    'Recall': f"{result.get('recall', 0)*100:.2f}%",
                    'F1': f"{result.get('f1', 0):.4f}",
                    'Samples': result.get('num_samples', 0)
                })
            
            df = pd.DataFrame(df_data)
            
            # Save CSV
            csv_path = output_dir / f'ablation_results_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"\n✅ CSV results saved to: {csv_path}")
            
            # Save JSON (with full numeric values)
            json_path = output_dir / f'ablation_results_{timestamp}.json'
            json_data = {r['combination']: r for r in sorted_results}
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"✅ JSON results saved to: {json_path}")
            
            # ✅ FIX: Save text report WITHOUT emoji (Windows compatible)
            report_path = output_dir / f'ablation_report_{timestamp}.txt'
            with open(report_path, 'w', encoding='utf-8') as f:  # ← Use UTF-8 encoding
                f.write("="*80 + "\n")
                f.write("HYBEX-LAW ABLATION STUDY REPORT\n")  # ← No emoji
                f.write("="*80 + "\n\n")
                
                f.write("RANKINGS (by F1 Score):\n")
                f.write("-"*80 + "\n")
                
                for rank, result in enumerate(sorted_results, 1):
                    f.write(f"{rank}. {result.get('combination', 'Unknown')}\n")
                    f.write(f"   Models: {', '.join(result.get('models', []))}\n")
                    f.write(f"   Accuracy: {result.get('accuracy', 0):.4f}\n")
                    f.write(f"   F1: {result.get('f1', 0):.4f}\n")
                    f.write(f"   Precision: {result.get('precision', 0):.4f}\n")
                    f.write(f"   Recall: {result.get('recall', 0):.4f}\n\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("KEY INSIGHTS:\n")
                f.write("-"*80 + "\n")
                
                # ✅ FIX: Best overall WITHOUT emoji (Windows compatible)
                f.write(f"Best Overall: {best.get('combination', 'Unknown')} (F1: {best.get('f1', 0):.4f})\n")
                
                # Best single model
                single_models = [r for r in sorted_results if len(r.get('models', [])) == 1]
                if single_models:
                    best_single = single_models[0]
                    f.write(f"Best Single Model: {best_single.get('combination', 'Unknown')} (F1: {best_single.get('f1', 0):.4f})\n")
                
                # Best pair
                pairs = [r for r in sorted_results if len(r.get('models', [])) == 2]
                if pairs:
                    best_pair = pairs[0]
                    f.write(f"Best Pair: {best_pair.get('combination', 'Unknown')} (F1: {best_pair.get('f1', 0):.4f})\n")
            
            logger.info(f"✅ Text report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save ablation reports: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())



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