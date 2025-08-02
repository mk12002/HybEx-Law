# hybex_system/evaluator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from .config import HybExConfig
from .neural_models import DomainClassifier, EligibilityPredictor, LegalDataset, ModelMetrics # Import actual model classes
from .prolog_engine import PrologEngine, LegalReasoning # Import PrologEngine and its result dataclass
from .data_processor import DataPreprocessor # To help with entity extraction if needed for evaluation

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

class ModelEvaluator:
    """Comprehensive model evaluation framework for HybEx-Law system."""
    
    def __init__(self, config: HybExConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # Or add a specific pad token
        
        self.setup_logging()
        logger.info(f"ModelEvaluator initialized on device: {self.device}")

    def setup_logging(self):
        log_file = self.config.get_log_path('model_evaluation')
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('model_evaluation.log') for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(self.config.LOGGING_CONFIG['format'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to ModelEvaluator logger.")
        
        logger.info("="*60)
        logger.info("Starting HybEx-Law Model Evaluation")
        logger.info("="*60)

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
        # Add other model types if needed (e.g., EntityExtractor if it's evaluated independently)
        
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
        
        test_dataset = LegalDataset(test_samples, self.tokenizer, self.config, task_type)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        predictions = []
        true_labels = []
        total_loss = 0.0
        
        # Loss function for consistent evaluation
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
                else: # Placeholder for other tasks if any
                    loss = criterion(logits, labels.long())
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        metrics = ModelMetrics(
            accuracy=accuracy_score(true_labels, predictions) if len(true_labels) > 0 else 0.0,
            f1_score=f1_score(true_labels, predictions, average='macro', zero_division=0) if len(true_labels) > 0 else 0.0,
            precision=precision_score(true_labels, predictions, average='macro', zero_division=0) if len(true_labels) > 0 else 0.0,
            recall=recall_score(true_labels, predictions, average='macro', zero_division=0) if len(true_labels) > 0 else 0.0,
            loss=avg_loss,
            classification_report=classification_report(true_labels, predictions, output_dict=True, zero_division=0),
            confusion_matrix=confusion_matrix(true_labels.argmax(axis=1), predictions.argmax(axis=1)) if task_type == "domain_classification" else confusion_matrix(true_labels, predictions) # For multi-label confusion matrix, adjust based on how labels are stored.
        )
        
        logger.info(f"Evaluation complete for {model.__class__.__name__}: Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}, Loss={metrics.loss:.4f}")
        return metrics

    def evaluate_end_to_end_system(self, models_paths: Dict[str, str], test_samples: List[Dict],
                                 prolog_reasoning_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate the complete hybrid neural-symbolic system.
        This includes neural model performance, symbolic reasoning accuracy, and hybrid agreement.
        """
        logger.info("Starting end-to-end system evaluation.")
        
        # 1. Load Neural Models
        loaded_neural_models = {}
        neural_evaluation_metrics = {}
        for model_name, model_path_str in tqdm(models_paths.items(), desc="Loading models", leave=False):
            try:
                # model_path_str might be the directory or the model.pt file
                # The load_trained_model expects the directory
                model_dir_path = Path(model_path_str)
                
                # Check if model_path_str is directly the .pt file or a directory
                if model_dir_path.is_file() and model_dir_path.suffix == '.pt':
                    # If it's the .pt file, get its parent directory
                    model_dir_path = model_dir_path.parent
                
                model_type = model_name # domain_classifier or eligibility_predictor
                loaded_neural_models[model_name] = self.load_trained_model(str(model_dir_path), model_type)
                
                # Evaluate each neural model
                if model_name == 'domain_classifier':
                    task = 'domain_classification'
                    # Prepare labels for domain classification, which are multi-hot encoded.
                    # Need to adapt LegalDataset for evaluation of raw query samples without pre-converted labels.
                    # Or, ensure test_samples from data_processor already have `domains` list.
                    
                    # Assuming test_samples are already preprocessed with 'domains' list and 'expected_eligibility' bool
                    neural_evaluation_metrics['domain_classifier'] = self.evaluate_neural_model(
                        loaded_neural_models['domain_classifier'], test_samples, task
                    )
                elif model_name == 'eligibility_predictor':
                    task = 'eligibility_prediction'
                    neural_evaluation_metrics['eligibility_predictor'] = self.evaluate_neural_model(
                        loaded_neural_models['eligibility_predictor'], test_samples, task
                    )
            except Exception as e:
                logger.error(f"Failed to load or evaluate neural model {model_name}: {e}")
        
        # 2. Evaluate Prolog Reasoning Accuracy (if results are provided)
        prolog_metrics = {}
        if prolog_reasoning_results:
            logger.info("Evaluating Prolog reasoning accuracy...")
            # Align Prolog results with test_samples for ground truth comparison
            prolog_preds = []
            prolog_labels = []
            
            # Create a map for quick lookup
            sample_map = {s.get('sample_id'): s for s in test_samples if s.get('sample_id')}

            for p_res in prolog_reasoning_results:
                case_id = p_res.get('case_id')
                prolog_eligible = p_res.get('eligible')
                
                if case_id and case_id in sample_map:
                    true_eligible = sample_map[case_id].get('expected_eligibility')
                    
                    if true_eligible is not None and prolog_eligible is not None:
                        prolog_preds.append(prolog_eligible)
                        prolog_labels.append(true_eligible)
                else:
                    logger.warning(f"Prolog result for case_id {case_id} not found in test samples. Skipping.")

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
                    'classification_report': classification_report(prolog_labels, prolog_preds, output_dict=True, zero_division=0)
                }
                logger.info(f"Prolog Evaluation: Acc={prolog_accuracy:.4f}, F1={prolog_f1:.4f}")
            else:
                logger.warning("No valid Prolog predictions to evaluate.")

        # 3. Hybrid Decision Making & Agreement Analysis
        hybrid_metrics = {}
        component_agreement = {}
        error_analysis_list = []
        
        if 'eligibility_predictor' in loaded_neural_models and prolog_reasoning_results:
            logger.info("Performing hybrid decision fusion and agreement analysis...")
            neural_predictor = loaded_neural_models['eligibility_predictor']
            
            hybrid_preds = []
            true_labels_hybrid = []
            neural_only_preds = []
            prolog_only_preds = []

            # Match samples to Prolog results
            prolog_results_map = {p.get('case_id'): p for p in prolog_reasoning_results if p.get('case_id')}

            for sample in tqdm(test_samples, desc="Processing hybrid evaluation", leave=False):
                sample_id = sample.get('sample_id')
                query_text = sample.get('query')
                true_eligible = sample.get('expected_eligibility')

                if sample_id not in prolog_results_map:
                    logger.warning(f"Sample ID {sample_id} not found in Prolog results for hybrid analysis. Skipping.")
                    continue

                # Get neural prediction for this sample
                encoding = self.tokenizer(
                    query_text,
                    truncation=True, padding='max_length', max_length=self.config.MODEL_CONFIG['max_length'],
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    neural_logits = neural_predictor(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])['logits']
                    neural_pred_prob = torch.sigmoid(neural_logits).item()
                    neural_pred_bool = neural_pred_prob > 0.5
                
                # Get Prolog prediction for this sample
                prolog_pred_dict = prolog_results_map[sample_id]
                prolog_pred_bool = prolog_pred_dict.get('eligible')
                
                # Store individual predictions for comparison
                neural_only_preds.append(neural_pred_bool)
                prolog_only_preds.append(prolog_pred_bool)
                true_labels_hybrid.append(true_eligible)

                # Hybrid Fusion Logic (Example: Weighted or Rule-based Override)
                # Simple fusion: Prolog overrides if high confidence, else neural is weighted.
                # A more advanced fusion would be a separate model or a more complex rule set.
                
                final_hybrid_pred = False
                fusion_method = "default"

                prolog_confidence = prolog_pred_dict.get('confidence', 0.5)

                if prolog_confidence >= self.config.PROLOG_CONFIG['min_confidence_for_override']: # Assuming such a config
                    final_hybrid_pred = prolog_pred_bool
                    fusion_method = "prolog_override"
                elif neural_pred_prob >= self.config.NEURAL_CONFIG['min_confidence_for_override']: # Assuming such a config
                     final_hybrid_pred = neural_pred_bool
                     fusion_method = "neural_override"
                else: # Simple majority vote or neural default
                    final_hybrid_pred = neural_pred_bool # Default to neural if no strong override

                hybrid_preds.append(final_hybrid_pred)

                # Analyze agreement and errors
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

            if hybrid_preds:
                hybrid_accuracy = accuracy_score(true_labels_hybrid, hybrid_preds)
                hybrid_f1 = f1_score(true_labels_hybrid, hybrid_preds, zero_division=0)
                hybrid_metrics = {
                    'accuracy': hybrid_accuracy,
                    'f1_score': hybrid_f1,
                    'classification_report': classification_report(true_labels_hybrid, hybrid_preds, output_dict=True, zero_division=0)
                }

                # Calculate overall agreement stats
                total_samples_analyzed = len(true_labels_hybrid)
                neural_prolog_agreement_count = sum(1 for n, p in zip(neural_only_preds, prolog_only_preds) if n == p)
                
                component_agreement = {
                    'neural_prolog_agreement_rate': neural_prolog_agreement_count / total_samples_analyzed if total_samples_analyzed > 0 else 0,
                    'total_hybrid_samples': total_samples_analyzed
                }
            else:
                logger.warning("No samples processed for hybrid evaluation.")

        # Consolidate all results
        final_evaluation_results = EvaluationResults(
            timestamp=datetime.now().isoformat(),
            overall_status="Completed",
            neural_metrics=asdict(neural_evaluation_metrics) if isinstance(neural_evaluation_metrics, ModelMetrics) else neural_evaluation_metrics,
            prolog_metrics=prolog_metrics,
            hybrid_metrics=hybrid_metrics,
            component_agreement=component_agreement,
            error_analysis=error_analysis_list,
            sample_size=len(test_samples),
            config_summary=self.config.get_summary()
        )
        
        logger.info("End-to-end evaluation completed.")
        return asdict(final_evaluation_results) # Return as dict for serialization

    def save_evaluation_results(self, evaluation_results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        results_path = self.config.RESULTS_DIR / "evaluation_results" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
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
        
        # Neural Metrics
        report_str += "🧠 Neural Model Performance:\n"
        for model_name, metrics in evaluation_results['neural_metrics'].items():
            report_str += f"  - {model_name.replace('_', ' ').title()}:\n"
            report_str += f"    Accuracy: {metrics.get('accuracy', 0):.4f}\n"
            report_str += f"    F1 Score (Macro): {metrics.get('f1_score', 0):.4f}\n"
            report_str += f"    Precision (Macro): {metrics.get('precision', 0):.4f}\n"
            report_str += f"    Recall (Macro): {metrics.get('recall', 0):.4f}\n"
            report_str += f"    Loss: {metrics.get('loss', 0):.4f}\n"
            # Optional: Add classification report details if desired
            # report_str += f"    Classification Report:\n{json.dumps(metrics.get('classification_report', {}), indent=2)}\n"
        report_str += "\n"

        # Prolog Metrics
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

        # Hybrid Metrics
        hybrid_metrics = evaluation_results.get('hybrid_metrics', {})
        if hybrid_metrics:
            report_str += "🤝 Hybrid System Performance:\n"
            report_str += f"  Accuracy: {hybrid_metrics.get('accuracy', 0):.4f}\n"
            report_str += f"  F1 Score: {hybrid_metrics.get('f1_score', 0):.4f}\n"
            report_str += "\n"
        else:
            report_str += "⚠️ Hybrid System Performance: Not available or not evaluated.\n\n"
        
        # Component Agreement
        agreement = evaluation_results.get('component_agreement', {})
        if agreement:
            report_str += "🔗 Component Agreement:\n"
            report_str += f"  Neural-Prolog Agreement Rate: {agreement.get('neural_prolog_agreement_rate', 0):.2%}\n"
            report_str += "\n"

        # Error Analysis (top 5 errors)
        error_analysis = evaluation_results.get('error_analysis', [])
        if error_analysis:
            report_str += f"Error Analysis ({len(error_analysis)} cases with disagreement):\n"
            for i, error in enumerate(error_analysis[:5]): # Show top 5 errors
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

        # Confusion Matrix for Eligibility Predictor (Binary)
        if 'eligibility_predictor' in evaluation_results['neural_metrics']:
            try:
                cm = evaluation_results['neural_metrics']['eligibility_predictor']['confusion_matrix']
                if cm is not None and len(cm) == 2: # Ensure it's a binary confusion matrix
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
        
        # Confusion Matrix for Prolog (Binary)
        if 'prolog_metrics' in evaluation_results and 'classification_report' in evaluation_results['prolog_metrics']:
            try:
                # Need true_labels and preds for confusion matrix
                # Re-extract them from test_samples and prolog_reasoning_results
                prolog_preds = []
                prolog_labels = []
                prolog_results_map = {p.get('case_id'): p for p in evaluation_results.get('prolog_reasoning_results', []) if p.get('case_id')} # Assumes this key exists in the input
                for sample in evaluation_results.get('test_samples', []): # Assumes test_samples are available
                    sample_id = sample.get('sample_id')
                    if sample_id in prolog_results_map:
                        prolog_preds.append(prolog_results_map[sample_id].get('eligible'))
                        prolog_labels.append(sample.get('expected_eligibility'))
                
                if prolog_preds:
                    cm_prolog = confusion_matrix(prolog_labels, prolog_preds)
                    if len(cm_prolog) == 2:
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