# hybex_system/evaluator.py

import json
import logging
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
from .prolog_engine import LegalReasoning

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
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
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
        
        self.setup_logging()
        logger.info(f"ModelEvaluator initialized on device: {self.device}")

    def setup_logging(self):
        log_file = self.config.get_log_path('model_evaluation')
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('model_evaluation.log') for h in logger.handlers):
            # Corrected: Add the 'encoding' parameter
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
        import gc
        gc.collect()

    def load_trained_model(self, model_path: str, model_type: str) -> nn.Module:
        """Load a trained neural model."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found at {path}")
        
        model_state_path = path / "model.pt"
        if not model_state_path.exists():
            raise FileNotFoundError(f"Model state file not found at {model_state_path}")
        
        model_class = None
        if model_type == 'domain_classifier':
            model_class = DomainClassifier
        elif model_type == 'eligibility_predictor':
            model_class = EligibilityPredictor
        
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_class(self.config)
        model.load_state_dict(torch.load(model_state_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        logger.info(f"✅ Loaded {model_type} from {model_path}")
        return model

    def evaluate_neural_model(self, model: nn.Module, test_samples: List[Dict], task_type: str) -> ModelMetrics:
        """Evaluate a single neural model and return metrics."""
        logger.info(f"Evaluating neural model: {model.__class__.__name__} for {task_type}")

        # Use the correct configuration based on the model type
        if task_type == "domain_classification":
            batch_size = self.config.MODEL_CONFIGS['domain_classifier']['batch_size']
        elif task_type == "eligibility_prediction":
            batch_size = self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size']
        else:
            batch_size = self.config.MODEL_CONFIG['batch_size'] # Fallback

        test_dataset = LegalDataset(test_samples, self.tokenizer, self.config, task_type)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size, # Correctly pass the resolved batch_size
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        predictions = []
        true_labels = []
        total_loss = 0.0
        
        if task_type == "domain_classification":
            criterion = nn.BCEWithLogitsLoss()
        elif task_type == "eligibility_prediction":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating model", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                if task_type == "domain_classification":
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                elif task_type == "eligibility_prediction":
                    loss = criterion(logits.squeeze(), labels)
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).cpu().numpy()
                else:
                    loss = criterion(logits, labels.long())
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        
        cm = None
        if task_type == "eligibility_prediction":
            cm = confusion_matrix(true_labels, predictions).tolist() 

        metrics = ModelMetrics(
            accuracy=accuracy_score(true_labels, predictions) if len(true_labels) > 0 else 0.0,
            f1_score=f1_score(true_labels, predictions, average='macro', zero_division=0) if len(true_labels) > 0 else 0.0,
            precision=precision_score(true_labels, predictions, average='macro', zero_division=0) if len(true_labels) > 0 else 0.0,
            recall=recall_score(true_labels, predictions, average='macro', zero_division=0) if len(true_labels) > 0 else 0.0,
            loss=avg_loss,
            classification_report=classification_report(true_labels, predictions, output_dict=True, zero_division=0),
            confusion_matrix=cm
        )
        
        logger.info(f"Evaluation complete for {model.__class__.__name__}: Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}, Loss={metrics.loss:.4f}")
        return metrics

    def evaluate_end_to_end_system(self, models_paths: Dict[str, str], test_samples: List[Dict],
                                 prolog_reasoning_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate the complete hybrid neural-symbolic system.
        ...
        """
        logger.info("Starting end-to-end system evaluation.")
        if models_paths is None:
            models_paths = {
                "domain_classifier": str(self.config.MODELS_DIR / "domain_classifier"),
                "eligibility_predictor": str(self.config.MODELS_DIR / "eligibility_predictor")
            }
        
        overall_status = "Completed"
        
        # If no Prolog reasoning results provided, run Prolog reasoning on test samples
        if prolog_reasoning_results is None:
            logger.info("No Prolog reasoning results provided. Running Prolog evaluation on test samples...")
            try:
                from .prolog_engine import PrologEngine
                prolog_engine = PrologEngine(self.config)
                prolog_reasoning_results = []
                
                for i, sample in enumerate(tqdm(test_samples[:100], desc="Running Prolog reasoning", leave=False)):  # Limit to first 100 for performance
                    try:
                        # Extract entities from the sample - reuse data processor if available
                        if not hasattr(self, '_data_processor_cache'):
                            from .data_processor import DataPreprocessor
                            self._data_processor_cache = DataPreprocessor(self.config)
                        
                        extracted_entities = self._data_processor_cache.extract_entities(sample['query'])
                        
                        # Run Prolog reasoning
                        reasoning_result = prolog_engine.comprehensive_legal_analysis(extracted_entities)
                        
                        prolog_reasoning_results.append({
                            'case_id': sample.get('sample_id', f"sample_{i}"),
                            'eligible': reasoning_result.eligible,
                            'confidence': reasoning_result.confidence,
                            'reasoning': reasoning_result.primary_reason
                        })
                        
                    except Exception as e:
                        logger.warning(f"Prolog reasoning failed for sample {i}: {e}")
                        continue
                        
                logger.info(f"Completed Prolog reasoning for {len(prolog_reasoning_results)} samples")
                prolog_engine.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to run Prolog reasoning: {e}")
                prolog_reasoning_results = []
        
        try:
            loaded_neural_models = {}
            neural_evaluation_metrics = {}
            
            # --- Evaluate Domain Classifier ---
            model_name = 'domain_classifier'
            if model_name in models_paths:
                try:
                    self.clear_gpu_memory()
                    model_path_str = models_paths[model_name]
                    loaded_neural_models[model_name] = self.load_trained_model(model_path_str, model_name)
                    task = 'domain_classification'
                    neural_evaluation_metrics[model_name] = asdict(self.evaluate_neural_model(
                        loaded_neural_models[model_name], test_samples, task
                    ))
                    # Aggressively clear memory after evaluation
                    del loaded_neural_models[model_name]
                    self.clear_gpu_memory()
                    logger.info("✅ Domain classifier evaluation completed and memory cleared")
                except Exception as e:
                    logger.error(f"Failed to load or evaluate neural model {model_name}: {e}")
                    overall_status = "Failed"
                    if model_name in loaded_neural_models: del loaded_neural_models[model_name]
                    self.clear_gpu_memory()

            # --- Evaluate Eligibility Predictor ---
            model_name = 'eligibility_predictor'
            if model_name in models_paths:
                try:
                    self.clear_gpu_memory()
                    model_path_str = models_paths[model_name]
                    loaded_neural_models[model_name] = self.load_trained_model(model_path_str, model_name)
                    task = 'eligibility_prediction'
                    neural_evaluation_metrics[model_name] = asdict(self.evaluate_neural_model(
                        loaded_neural_models[model_name], test_samples, task
                    ))
                    logger.info("✅ Eligibility predictor evaluation completed")
                    
                except Exception as e:
                    logger.error(f"Failed to load or evaluate neural model {model_name}: {e}")
                    overall_status = "Failed"
                    if model_name in loaded_neural_models: del loaded_neural_models[model_name]
                    self.clear_gpu_memory()
            
            prolog_metrics = {}
            if prolog_reasoning_results:
                logger.info("Evaluating Prolog reasoning accuracy...")
                logger.info(f"Processing {len(prolog_reasoning_results)} Prolog reasoning results...")
                prolog_preds = []
                prolog_labels = []
                
                sample_map = {s.get('sample_id'): s for s in test_samples if s.get('sample_id')}
                logger.info(f"Created sample map with {len(sample_map)} test samples")

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
                    else:
                        logger.debug(f"Prolog result for case_id {case_id} not found in test samples. Skipping.")

                logger.info(f"Successfully matched {matched_samples} Prolog results with test samples")

                if prolog_preds:
                    prolog_accuracy = accuracy_score(prolog_labels, prolog_preds)
                    prolog_f1 = f1_score(prolog_labels, prolog_preds, zero_division=0)
                    prolog_precision = precision_score(prolog_labels, prolog_preds, zero_division=0)
                    prolog_recall = recall_score(prolog_labels, prolog_preds, zero_division=0)
                    
                    prolog_metrics = {
                        'accuracy': prolog_accuracy,
                        'f1_score': prolog_f1,
                        'precision': prolog_precision,
                        'recall': prolog_recall,
                        'classification_report': classification_report(prolog_labels, prolog_preds, output_dict=True, zero_division=0),
                        'confusion_matrix': confusion_matrix(prolog_labels, prolog_preds).tolist(),
                        'evaluated_samples': len(prolog_preds)
                    }
                    logger.info(f"✅ Prolog Evaluation Complete: Acc={prolog_accuracy:.4f}, F1={prolog_f1:.4f}, Precision={prolog_precision:.4f}, Recall={prolog_recall:.4f}")
                else:
                    logger.warning("No valid Prolog predictions to evaluate.")
                    prolog_metrics = {
                        'error': 'No valid Prolog predictions found',
                        'evaluated_samples': 0
                    }
            else:
                logger.warning("⚠️ No Prolog reasoning results available for evaluation")

            hybrid_metrics = {}
            component_agreement = {}
            error_analysis_list = []
            
            if 'eligibility_predictor' in loaded_neural_models and prolog_reasoning_results:
                logger.info("Performing hybrid decision fusion and agreement analysis...")
                logger.info(f"Running hybrid evaluation with neural model and {len(prolog_reasoning_results)} Prolog results")
                neural_predictor = loaded_neural_models['eligibility_predictor']
                
                hybrid_preds = []
                true_labels_hybrid = []
                neural_only_preds = []
                prolog_only_preds = []

                prolog_results_map = {p.get('case_id'): p for p in prolog_reasoning_results if p.get('case_id')}
                logger.info(f"Created Prolog results map with {len(prolog_results_map)} entries")

                processed_hybrid = 0
                for sample in tqdm(test_samples[:100], desc="Processing hybrid evaluation", leave=False):  # Limit for performance
                    sample_id = sample.get('sample_id')
                    query_text = sample.get('query')
                    true_eligible = sample.get('expected_eligibility')

                    if sample_id not in prolog_results_map:
                        continue

                    try:
                        encoding = self.tokenizer(
                            query_text,
                            truncation=True, padding='max_length', max_length=self.config.MODEL_CONFIG['max_length'],
                            return_tensors='pt'
                        ).to(self.device)
                        
                        with torch.no_grad():
                            neural_logits = neural_predictor(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])['logits']
                            neural_pred_prob = torch.sigmoid(neural_logits).item()
                            neural_pred_bool = neural_pred_prob > 0.5
                        
                        prolog_pred_dict = prolog_results_map[sample_id]
                        prolog_pred_bool = prolog_pred_dict.get('eligible')
                        prolog_confidence = prolog_pred_dict.get('confidence', 0.5)
                        
                        neural_only_preds.append(neural_pred_bool)
                        prolog_only_preds.append(prolog_pred_bool)
                        true_labels_hybrid.append(true_eligible)

                        final_hybrid_pred = False
                        fusion_method = "default"

                        # Safe access to configuration with fallbacks
                        prolog_threshold = getattr(self.config, 'PROLOG_CONFIG', {}).get('min_confidence_for_override', 0.95)
                        neural_threshold = getattr(self.config, 'NEURAL_CONFIG', {}).get('min_confidence_for_override', 0.90)

                        if prolog_confidence >= prolog_threshold:
                            final_hybrid_pred = prolog_pred_bool
                            fusion_method = "prolog_override"
                        elif neural_pred_prob >= neural_threshold:
                            final_hybrid_pred = neural_pred_bool
                            fusion_method = "neural_override"
                        else:
                            final_hybrid_pred = neural_pred_bool

                        hybrid_preds.append(final_hybrid_pred)
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
                    hybrid_f1 = f1_score(true_labels_hybrid, hybrid_preds, zero_division=0)
                    hybrid_precision = precision_score(true_labels_hybrid, hybrid_preds, zero_division=0)
                    hybrid_recall = recall_score(true_labels_hybrid, hybrid_preds, zero_division=0)
                    
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
                    
                else:
                    logger.warning("No samples processed for hybrid evaluation.")
                    hybrid_metrics = {'error': 'No hybrid predictions generated'}
                    
                # Clean up eligibility predictor after hybrid evaluation
                if 'eligibility_predictor' in loaded_neural_models:
                    del loaded_neural_models['eligibility_predictor']
                    self.clear_gpu_memory()
                    logger.info("✅ Eligibility predictor cleaned up after hybrid evaluation")

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
                return asdict(final_evaluation_results)

        except Exception as e:
            logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
            overall_status = "Failed"
                
            error_result = EvaluationResults(
                timestamp=datetime.now().isoformat(),
                overall_status=overall_status,
                neural_metrics={},
                prolog_metrics={},
                hybrid_metrics={},
                component_agreement={},
                error_analysis=[{'message': str(e), 'details': 'An unexpected error occurred.'}],
                sample_size=len(test_samples),
                config_summary=self.config.get_summary()
            )
            return asdict(error_result)

    def save_evaluation_results(self, evaluation_results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        results_path = self.config.RESULTS_DIR / "evaluation_results" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Corrected: Use helper function to ensure all content is serializable
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
        for model_name, metrics in evaluation_results['neural_metrics'].items():
            report_str += f"  - {model_name.replace('_', ' ').title()}:\n"
            report_str += f"    Accuracy: {metrics.get('accuracy', 0):.4f}\n"
            report_str += f"    F1 Score (Macro): {metrics.get('f1_score', 0):.4f}\n"
            report_str += f"    Precision (Macro): {metrics.get('precision', 0):.4f}\n"
            report_str += f"    Recall (Macro): {metrics.get('recall', 0):.4f}\n"
            report_str += f"    Loss: {metrics.get('loss', 0):.4f}\n"
        report_str += "\n"

        prolog_metrics = evaluation_results.get('prolog_metrics', {})
        if prolog_metrics:
            report_str += "⚖️ Prolog Reasoning Performance:\n"
            report_str += f"  Accuracy: {prolog_metrics.get('accuracy', 0):.4f}\n"
            report_str += f"  F1 Score: {prolog_metrics.get('f1_score', 0):.4f}\n"
            report_str += f"  Precision: {prolog_metrics.get('precision', 0):.4f}\n"
            report_str += f"  Recall: {prolog_metrics.get('recall', 0):.4f}\n"
            report_str += "\n"
        else:
            report_str += "⚠️ Prolog Reasoning Performance: Not available or not evaluated.\n\n"

        hybrid_metrics = evaluation_results.get('hybrid_metrics', {})
        if hybrid_metrics:
            report_str += "🤝 Hybrid System Performance:\n"
            report_str += f"  Accuracy: {hybrid_metrics.get('accuracy', 0):.4f}\n"
            report_str += f"  F1 Score: {hybrid_metrics.get('f1_score', 0):.4f}\n"
            report_str += "\n"
        else:
            report_str += "⚠️ Hybrid System Performance: Not available or not evaluated.\n\n"
        
        agreement = evaluation_results.get('component_agreement', {})
        if agreement:
            report_str += "🔗 Component Agreement:\n"
            report_str += f"  Neural-Prolog Agreement Rate: {agreement.get('neural_prolog_agreement_rate', 0):.2%}\n"
            report_str += "\n"

        error_analysis = evaluation_results.get('error_analysis', [])
        if error_analysis:
            report_str += f"Error Analysis ({len(error_analysis)} cases with disagreement):\n"
            for i, error in enumerate(error_analysis[:5]):
                report_str += f"  Error {i+1} (Sample ID: {error.get('sample_id', 'N/A')}):\n"
                report_str += f"    Query: {error.get('query', '')[:100]}...\n"
                report_str += f"    True: {error.get('true_label', 'N/A')}, Neural Pred: {error.get('neural_pred', 'N/A')}, Prolog Pred: {error.get('prolog_pred', 'N/A')}, Hybrid Pred: {error.get('hybrid_pred', 'N/A')}\n"
                report_str += f"    Type: {error.get('error_type', 'N/A')}\n"
                report_str += f"    Disagreements: {json.dumps(error.get('disagreement', {}))}\n"
            if len(error_analysis) > 5:
                report_str += f"  ... and {len(error_analysis) - 5} more errors. See full results JSON for details.\n"
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
        
        plt.style.use('seaborn-v0_8')

        if 'eligibility_predictor' in evaluation_results['neural_metrics']:
            try:
                cm = evaluation_results['neural_metrics']['eligibility_predictor']['confusion_matrix']
                if cm is not None and len(cm) == 2:
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Not Eligible', 'Eligible'],
                                yticklabels=['Not Eligible', 'Eligible'])
                    plt.title('Neural Eligibility Predictor Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.tight_layout()
                    plt.savefig(plots_dir / "eligibility_predictor_confusion_matrix.png")
                    plt.close()
                    logger.info("Generated Neural Eligibility Predictor Confusion Matrix.")
            except Exception as e:
                logger.warning(f"Could not generate neural eligibility confusion matrix: {e}")
        
        if 'prolog_metrics' in evaluation_results and 'confusion_matrix' in evaluation_results['prolog_metrics']:
            try:
                cm_prolog = evaluation_results['prolog_metrics']['confusion_matrix']
                if cm_prolog is not None and len(cm_prolog) == 2:
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm_prolog, annot=True, fmt='d', cmap='Greens',
                                xticklabels=['Not Eligible', 'Eligible'],
                                yticklabels=['Not Eligible', 'Eligible'])
                    plt.title('Prolog Reasoning Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.tight_layout()
                    plt.savefig(plots_dir / "prolog_confusion_matrix.png")
                    plt.close()
                    logger.info("Generated Prolog Reasoning Confusion Matrix.")
            except Exception as e:
                logger.warning(f"Could not generate Prolog confusion matrix: {e}")

        logger.info(f"✅ Evaluation visualizations saved to {plots_dir}")