# hybex_system/neural_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import os # Added for os.makedirs if needed, though pathlib covers most
from tqdm import tqdm

from .config import HybExConfig

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Dataclass to hold model evaluation metrics."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    loss: float = 0.0
    classification_report: Dict[str, Any] = None
    confusion_matrix: Any = None # Keep Any type for compatibility, handle None for multi-label

class DomainClassifier(nn.Module):
    """Multi-label classification model for legal domains."""
    def __init__(self, config: HybExConfig):
        super().__init__()
        self.config = config
        self.num_labels = len(config.ENTITY_CONFIG['domains'])
        self.base_model = AutoModel.from_pretrained(config.MODEL_CONFIG['base_model'])
        self.dropout = nn.Dropout(config.MODEL_CONFIG['dropout_prob'])
        self.classifier = nn.Linear(self.base_model.config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] # CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {'logits': logits}

class EligibilityPredictor(nn.Module):
    """Binary classification model for legal aid eligibility."""
    def __init__(self, config: HybExConfig):
        super().__init__()
        self.config = config
        self.base_model = AutoModel.from_pretrained(config.MODEL_CONFIG['base_model'])
        self.dropout = nn.Dropout(config.MODEL_CONFIG['dropout_prob'])
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 1) # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] # CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {'logits': logits.squeeze(-1)} # Squeeze for BCEWithLogitsLoss

class LegalDataset(Dataset):
    """PyTorch Dataset for legal text processing"""
    
    def __init__(self, samples: List[Dict], tokenizer, config: HybExConfig, task_type: str = "domain_classification"):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.task_type = task_type
        self.max_length = config.MODEL_CONFIG['max_length']

        # Ensure that `domains` are always lists for multi-label tasks
        if self.task_type == "domain_classification":
            for sample in self.samples:
                if 'domains' not in sample or not isinstance(sample['domains'], list):
                    sample['domains'] = [] # Ensure it's a list for multi-hot encoding

        logger.info(f"Created {task_type} dataset with {len(samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input text
        encoding = self.tokenizer(
            sample['query'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare labels based on task type
        if self.task_type == "domain_classification":
            # Multi-label classification: labels are multi-hot encoded
            labels = torch.zeros(len(self.config.ENTITY_CONFIG['domains']), dtype=torch.float)
            for i, domain in enumerate(self.config.ENTITY_CONFIG['domains']):
                if domain in sample['domains']:
                    labels[i] = 1.0
        elif self.task_type == "eligibility_prediction":
            # Binary classification: labels are single float (0.0 or 1.0)
            labels = torch.tensor(float(sample['expected_eligibility']), dtype=torch.float)
        else:
            labels = torch.tensor(0.0, dtype=torch.float) # Placeholder for other tasks

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'sample_id': sample.get('sample_id', idx)
        }

class ModelTrainer:
    """Orchestrates the training of neural models."""

    def __init__(self, config: HybExConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])
        
        # Corrected: Ensure pad_token is set appropriately for BERT-style models
        if self.tokenizer.pad_token is None:
            # Check if a default pad_token_id exists for the model's vocabulary
            if self.tokenizer.pad_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
            else:
                # Fallback: Add a new pad token if none exists and no ID is defined
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added a new [PAD] token to the tokenizer.")
        
        self.setup_logging()
        logger.info(f"ModelTrainer initialized on device: {self.device}")

    def setup_logging(self):
        log_file = self.config.get_log_path('neural_training')
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('neural_training.log') for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(self.config.LOGGING_CONFIG['format'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to ModelTrainer logger.")
        logger.info("="*60)
        logger.info("Starting HybEx-Law Neural Model Training")
        logger.info("="*60)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.config.MODEL_CONFIGS.get(model_name, self.config.MODEL_CONFIGS['domain_classifier'])

    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   model_name: str, task_type: str) -> Dict[str, Any]:
        """Train a model with comprehensive monitoring."""
        
        model_config = self.get_model_config(model_name)
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=model_config['early_stopping_patience'] // 2
        )
        
        if task_type == "domain_classification":
            criterion = nn.BCEWithLogitsLoss()
        elif task_type == "eligibility_prediction":
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type for training: {task_type}")

        best_val_f1 = -1.0
        best_model_state = None
        patience_counter = 0

        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'train_metrics': [],
            'val_metrics': []
        }

        logger.info(f"Starting training for {model_name} for {model_config['epochs']} epochs.")
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(model_config['epochs']), desc=f"Training {model_name}", unit="epoch")
        
        for epoch in epoch_pbar:
            model.train()
            total_train_loss = 0
            train_preds, train_labels = [], []

            # Progress bar for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")
            
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                if task_type == "eligibility_prediction":
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                elif task_type == "domain_classification":
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['gradient_clip_val'])
                optimizer.step()

                total_train_loss += loss.item()
                
                # Update progress bar with current loss
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_metrics = self.calculate_metrics(train_preds, train_labels, task_type)
            train_metrics.loss = avg_train_loss
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_metrics'].append(train_metrics)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Validation
            model.eval()
            total_val_loss = 0
            val_preds, val_labels = [], []
            
            # Progress bar for validation batches
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False, unit="batch")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']

                    if task_type == "eligibility_prediction":
                        loss = criterion(logits, labels)
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    elif task_type == "domain_classification":
                        loss = criterion(logits, labels)
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    
                    total_val_loss += loss.item()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
                    
                    # Update validation progress bar
                    val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

            avg_val_loss = total_val_loss / len(val_loader)
            val_metrics = self.calculate_metrics(val_preds, val_labels, task_type)
            val_metrics.loss = avg_val_loss
            training_history['val_losses'].append(avg_val_loss)
            training_history['val_metrics'].append(val_metrics)
            
            # Update main epoch progress bar with key metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'train_f1': f'{train_metrics.f1_score:.4f}',
                'val_f1': f'{val_metrics.f1_score:.4f}'
            })
            
            logger.info(
                f"Epoch {epoch+1}/{model_config['epochs']} | "
                f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_metrics.f1_score:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_metrics.f1_score:.4f}"
            )

            scheduler.step(val_metrics.f1_score) # Step LR scheduler based on validation F1

            # Early stopping logic
            if val_metrics.f1_score > best_val_f1:
                best_val_f1 = val_metrics.f1_score
                best_model_state = model.state_dict()
                patience_counter = 0
                logger.info(f"New best validation F1 score: {best_val_f1:.4f}. Saving model state.")
            else:
                patience_counter += 1
                logger.info(f"Validation F1 did not improve. Patience: {patience_counter}/{model_config['early_stopping_patience']}")
                if patience_counter >= model_config['early_stopping_patience']:
                    logger.info("Early stopping triggered.")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with F1 score: {best_val_f1:.4f}")
        else:
            logger.warning(f"No best model state found for {model_name}. Using last epoch's state.")
        
        return {
            'model': model,
            'training_history': training_history,
            'best_f1_score': best_val_f1,
            'final_epoch': epoch + 1
        }
    
    def calculate_metrics(self, predictions: List[Any], labels: List[Any], task_type: str) -> ModelMetrics:
        """Calculate comprehensive evaluation metrics."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        if task_type == "domain_classification":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            precision = precision_score(labels, predictions, average='macro', zero_division=0)
            recall = recall_score(labels, predictions, average='macro', zero_division=0)
            
            try:
                target_names = self.config.ENTITY_CONFIG['domains']
                class_report = classification_report(labels, predictions, target_names=target_names, output_dict=True, zero_division=0)
            except Exception as e:
                logger.warning(f"Could not generate classification report for domain classification: {e}")
                class_report = {}
            
            cm = None
            
        else:
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, zero_division=0)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            
            try:
                target_names = ['Not Eligible', 'Eligible']
                class_report = classification_report(labels, predictions, target_names=target_names, output_dict=True, zero_division=0)
            except Exception as e:
                logger.warning(f"Could not generate classification report for eligibility prediction: {e}")
                class_report = {}

            # Changed to store the NumPy array directly
            cm = confusion_matrix(labels, predictions)
            
        return ModelMetrics(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            loss=0.0,
            classification_report=class_report,
            confusion_matrix=cm
        )
    
    def save_model(self, model: nn.Module, model_name: str, training_results: Dict) -> str:
        """Save trained model with comprehensive metadata."""
        model_dir = self.config.MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        tokenizer_path = model_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        history_path = model_dir / "training_history.json"
        
        serializable_history = {
            'train_losses': training_results['training_history']['train_losses'],
            'val_losses': training_results['training_history']['val_losses'],
            'learning_rates': training_results['training_history']['learning_rates'],
            'train_metrics': [],
            'val_metrics': [],
            'best_f1_score': training_results['best_f1_score'],
            'final_epoch': training_results['final_epoch'],
            'model_config': self.config.MODEL_CONFIG # Capture model config
        }
        
        # Convert ModelMetrics objects to dictionaries for serialization
        for metrics in training_results['training_history']['train_metrics']:
            serializable_history['train_metrics'].append(asdict(metrics))
        
        for metrics in training_results['training_history']['val_metrics']:
            serializable_history['val_metrics'].append(asdict(metrics))
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        info_path = model_dir / "model_info.json"
        model_info = {
            'model_name': model_name,
            'model_class': model.__class__.__name__,
            'base_model': self.config.MODEL_CONFIG['base_model'],
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': str(self.device),
            'save_timestamp': datetime.now().isoformat(),
            'config_summary': self.config.get_summary()
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved model {model_name} to {model_dir}")
        return str(model_dir)

    def create_training_plots(self, training_results: Dict, model_name: str):
        """Create comprehensive training visualization plots."""
        history = training_results['training_history']
        
        plots_dir = self.config.RESULTS_DIR / "training_plots" / model_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss', linewidth=2)
        plt.plot(history['val_losses'], label='Validation Loss', linewidth=2)
        plt.title(f'{model_name} - Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Learning rate schedule
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rates'], linewidth=2, color='orange')
        plt.title(f'{model_name} - Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_and_lr.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Metrics curves
        # Ensure that 'train_metrics' and 'val_metrics' are lists of ModelMetrics objects
        train_metrics = [ModelMetrics(**m) if isinstance(m, dict) else m for m in history['train_metrics']]
        val_metrics = [ModelMetrics(**m) if isinstance(m, dict) else m for m in history['val_metrics']]
        
        metrics_names = ['accuracy', 'f1_score', 'precision', 'recall']
        
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics_names):
            plt.subplot(2, 2, i + 1)
            
            train_values = [getattr(m, metric) for m in train_metrics]
            val_values = [getattr(m, metric) for m in val_metrics]
            
            plt.plot(train_values, label=f'Training {metric.replace("_", " ").title()}', linewidth=2)
            plt.plot(val_values, label=f'Validation {metric.replace("_", " ").title()}', linewidth=2)
            plt.title(f'{model_name} - {metric.replace("_", " ").title()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace("_", " ").title())
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training plots to {plots_dir}")

    def train_all_models(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict[str, Any]:
        """Train all required neural models."""
        logger.info(f"Training Domain Classifier and Eligibility Predictor models.")
        
        trained_models_info = {}
        
        # Train Domain Classifier
        logger.info("\n--- Training Domain Classifier ---")
        domain_classifier = DomainClassifier(self.config)
        domain_train_dataset = LegalDataset(train_samples, self.tokenizer, self.config, task_type="domain_classification")
        domain_val_dataset = LegalDataset(val_samples, self.tokenizer, self.config, task_type="domain_classification")
        
        domain_train_loader = DataLoader(domain_train_dataset, batch_size=self.config.MODEL_CONFIGS['domain_classifier']['batch_size'], shuffle=True)
        domain_val_loader = DataLoader(domain_val_dataset, batch_size=self.config.MODEL_CONFIGS['domain_classifier']['batch_size'], shuffle=False)
        
        domain_results = self.train_model(domain_classifier, domain_train_loader, domain_val_loader, "domain_classifier", "domain_classification")
        saved_domain_model_path = self.save_model(domain_results['model'], "domain_classifier", domain_results)
        self.create_training_plots(domain_results, "domain_classifier")
        
        trained_models_info["domain_classifier"] = {
            "path": Path(saved_domain_model_path), # Store as Path object
            "best_f1": domain_results['best_f1_score']
        }
        
        # Train Eligibility Predictor
        logger.info("\n--- Training Eligibility Predictor ---")
        eligibility_predictor = EligibilityPredictor(self.config)
        eligibility_train_dataset = LegalDataset(train_samples, self.tokenizer, self.config, task_type="eligibility_prediction")
        eligibility_val_dataset = LegalDataset(val_samples, self.tokenizer, self.config, task_type="eligibility_prediction")
        
        eligibility_train_loader = DataLoader(eligibility_train_dataset, batch_size=self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size'], shuffle=True)
        eligibility_val_loader = DataLoader(eligibility_val_dataset, batch_size=self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size'], shuffle=False)
        
        eligibility_results = self.train_model(eligibility_predictor, eligibility_train_loader, eligibility_val_loader, "eligibility_predictor", "eligibility_prediction")
        saved_eligibility_model_path = self.save_model(eligibility_results['model'], "eligibility_predictor", eligibility_results)
        self.create_training_plots(eligibility_results, "eligibility_predictor")
        
        trained_models_info["eligibility_predictor"] = {
            "path": Path(saved_eligibility_model_path), # Store as Path object
            "best_f1": eligibility_results['best_f1_score']
        }
        
        logger.info("All neural models trained successfully.")
        return trained_models_info