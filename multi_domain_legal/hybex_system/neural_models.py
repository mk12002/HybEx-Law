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
import os 
from tqdm import tqdm

from .config import HybExConfig
from .knowledge_graph_engine import KnowledgeGraphEngine # Import the GNN engine

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
    # Changed type hint to Optional[List[List[int]]] for binary classification consistency
    confusion_matrix: Optional[List[List[int]]] = None 

class DomainClassifier(nn.Module):
    """Multi-label classification model for legal domains."""
    def __init__(self, config: HybExConfig):
        super().__init__()
        self.config = config
        model_config = config.get_model_config('domain_classifier') # <-- Get specific config
        self.num_labels = len(config.ENTITY_CONFIG['domains'])
        self.base_model = AutoModel.from_pretrained(model_config['model_name'])
        self.dropout = nn.Dropout(model_config['dropout_prob']) 
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
        model_config = config.get_model_config('eligibility_predictor') # <-- Get specific config
        self.base_model = AutoModel.from_pretrained(model_config['model_name'])
        self.dropout = nn.Dropout(model_config['dropout_prob'])
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 1) # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] # CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {'logits': logits.squeeze(-1)} # Squeeze for BCEWithLogitsLoss

class LegalDataset(Dataset):
    """PyTorch Dataset for legal text processing"""
    def __init__(self, samples: List[Dict], tokenizer, config: HybExConfig, task_type: str = "domain_classification", model_config: Dict[str, Any] = None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.task_type = task_type
        self.max_length = model_config.get('max_length', 512) if model_config else 512 

        # Ensure that `domains` are always lists for multi-label tasks
        if self.task_type == "domain_classification":
            for sample in self.samples:
                # Use .get() for safety
                if not isinstance(sample.get('domains'), list):
                    sample['domains'] = [] 

        logger.info(f"Created {task_type} dataset with {len(samples)} samples")

        # DEBUG: Log first few samples to detect leakage during dataset construction
        try:
            logger.debug("\n🔍 DEBUG: First 5 training samples:")
            for i, sample in enumerate(self.samples[:5]):
                logger.debug(f"\nSample {i}:")
                logger.debug(f"  Query: {sample.get('query', 'MISSING')[:100]}...")
                logger.debug(f"  Domains: {sample.get('domains', 'MISSING')}")
                logger.debug(f"  Eligibility: {sample.get('expected_eligibility', 'MISSING')}")
                logger.debug(f"  Keys: {list(sample.keys())}")

            # Quick check for leaked fields that shouldn't be present in neural inputs
            leaked_keys = ['extracted_facts', 'prolog_facts', 'user_demographics', 'income', 'social_category']
            for key in leaked_keys:
                if key in self.samples[0]:
                    logger.warning(f"⚠️  WARNING: Found potential data leakage key: '{key}' in samples[0]")
        except Exception:
            # Don't fail dataset construction on debug printing issues
            logger.exception("Debug print failed in LegalDataset.__init__")

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
            # Ensure safe conversion, handle potential missing key
            eligibility_value = float(sample.get('expected_eligibility', 0.0))
            labels = torch.tensor(eligibility_value, dtype=torch.float)
        elif self.task_type == "multi_task":
            # For EnhancedLegalBERT: need both eligibility and domain labels
            eligibility_value = float(sample.get('expected_eligibility', 0.0))
            labels = torch.tensor(eligibility_value, dtype=torch.long)  # 0 or 1 for CrossEntropyLoss
        else:
            labels = torch.tensor(0.0, dtype=torch.float) # Placeholder for other tasks

        # Prepare domain information for multi-task learning
        domain_labels = torch.zeros(len(self.config.ENTITY_CONFIG['domains']), dtype=torch.float)
        domain_indices = []
        for i, domain in enumerate(self.config.ENTITY_CONFIG['domains']):
            if domain in sample.get('domains', []):
                domain_labels[i] = 1.0
                domain_indices.append(i)
        
        # Primary domain index (for domain-specific projection)
        primary_domain_idx = domain_indices[0] if domain_indices else 0

        return {
            # Squeeze dim 1 (batch dimension from `return_tensors='pt'`)
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'domains': torch.tensor(primary_domain_idx, dtype=torch.long),  # Single domain index
            'domain_labels': domain_labels,  # Multi-hot for domain classification
            'sample_id': sample.get('sample_id', idx)
        }

class ModelTrainer:
    """Orchestrates the training of neural models."""

    def __init__(self, config: HybExConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])

        # Corrected: Ensure pad_token is set appropriately for BERT-style models
        if self.tokenizer.pad_token is None:
            # Check if a default pad_token_id exists for the model's vocabulary
            if self.tokenizer.pad_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
            else:
                # Fallback: Add a new pad token if none exists and no ID is defined
                # Note: This requires resizing the model's token embeddings later if needed.
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added a new [PAD] token to the tokenizer.")

        self.setup_logging()
        logger.info(f"ModelTrainer initialized on device: {self.device}")

    def setup_logging(self):
        log_file = self.config.get_log_path('neural_training')
        # Robust check against multiple handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('neural_training.log') for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
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
        # Use .get with a safe fallback
        return self.config.MODEL_CONFIGS.get(model_name, self.config.MODEL_CONFIGS.get('domain_classifier', {}))

    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    model_name: str, task_type: str) -> Dict[str, Any]:
        """Train a model with comprehensive monitoring, gradient accumulation, and mixed precision."""

        model_config = self.get_model_config(model_name)
        model.to(self.device)
        
        # Gradient accumulation settings (simulate larger batch size)
        ACCUMULATION_STEPS = 4  # Effective batch = 2 * 4 = 8
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=model_config['early_stopping_patience'] // 2
        )

        # Mixed precision training for memory optimization
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler() if self.device.type == 'cuda' else None

        # Use weights for BCEWithLogitsLoss if class imbalance is an issue
        if task_type in ["domain_classification", "eligibility_prediction"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type for training: {task_type}")

        best_val_f1 = -1.0
        best_model_state = None
        patience_counter = 0
        final_epoch = 0

        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'train_metrics': [],
            'val_metrics': []
        }

        logger.info(f"Starting training for {model_name} with gradient accumulation (steps={ACCUMULATION_STEPS})")
        if scaler:
            logger.info("Mixed precision training enabled (FP16)")

        # Progress bar for epochs
        epoch_pbar = tqdm(range(model_config['epochs']), desc=f"Training {model_name}", unit="epoch")

        for epoch in epoch_pbar:
            final_epoch = epoch + 1
            model.train()
            total_train_loss = 0
            train_preds, train_labels = [], []

            optimizer.zero_grad()  # Zero gradients at start

            # Progress bar for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Use autocast for mixed precision if CUDA available
                if scaler:
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs['logits']
                        loss = criterion(logits, labels)
                        loss = loss / ACCUMULATION_STEPS  # Normalize for accumulation
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    loss = criterion(logits, labels)
                    loss = loss / ACCUMULATION_STEPS
                
                # Predictions
                if task_type == "eligibility_prediction":
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                elif task_type == "domain_classification":
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update weights every ACCUMULATION_STEPS or on last batch
                if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['gradient_clip_val'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['gradient_clip_val'])
                        optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * ACCUMULATION_STEPS  # De-normalize for logging

                # Update progress bar with current loss
                train_pbar.set_postfix({'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}'})

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

                    loss = criterion(logits, labels)

                    if task_type == "eligibility_prediction":
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    elif task_type == "domain_classification":
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
                # Deep copy the state dictionary to prevent modification by subsequent training steps
                best_model_state = model.state_dict().copy() 
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
            'final_epoch': final_epoch
        }

    def calculate_metrics(self, predictions: List[Any], labels: List[Any], task_type: str) -> ModelMetrics:
        """Calculate comprehensive evaluation metrics."""
        # Convert to numpy arrays, handle list of lists vs 1D array conversion
        labels_np = np.array(labels)
        predictions_np = np.array(predictions)

        # Reshape labels/predictions if they ended up as (N, 1) instead of (N,) for binary
        if task_type == "eligibility_prediction" and labels_np.ndim > 1 and labels_np.shape[1] == 1:
            labels_np = labels_np.squeeze()
            predictions_np = predictions_np.squeeze()

        if task_type == "domain_classification":
            # Multi-label metrics: use macro or samples average, macro is standard here
            accuracy = accuracy_score(labels_np, predictions_np) # Subset accuracy
            f1 = f1_score(labels_np, predictions_np, average='macro', zero_division=0)
            precision = precision_score(labels_np, predictions_np, average='macro', zero_division=0)
            recall = recall_score(labels_np, predictions_np, average='macro', zero_division=0)

            try:
                target_names = self.config.ENTITY_CONFIG['domains']
                class_report = classification_report(labels_np, predictions_np, target_names=target_names, output_dict=True, zero_division=0)
            except Exception as e:
                logger.warning(f"Could not generate classification report for domain classification: {e}")
                class_report = {}
            
            # Multi-label CM is complex, return None
            cm = None 

        elif task_type == "eligibility_prediction":
            # Binary metrics: use 'binary' average
            accuracy = accuracy_score(labels_np, predictions_np)
            f1 = f1_score(labels_np, predictions_np, average='binary', zero_division=0)
            precision = precision_score(labels_np, predictions_np, average='binary', zero_division=0)
            recall = recall_score(labels_np, predictions_np, average='binary', zero_division=0)

            try:
                target_names = ['Not Eligible', 'Eligible']
                class_report = classification_report(labels_np, predictions_np, target_names=target_names, output_dict=True, zero_division=0)
            except Exception as e:
                logger.warning(f"Could not generate classification report for eligibility prediction: {e}")
                class_report = {}

            # Store the NumPy array as a list of lists for JSON serialization
            cm = confusion_matrix(labels_np, predictions_np).tolist()
        
        else:
            # Fallback for unexpected task type
            accuracy, f1, precision, recall, class_report, cm = 0.0, 0.0, 0.0, 0.0, {}, None

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
        # Save model's state dictionary only
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
            'model_config': self.get_model_config(model_name) # Capture model config specifically
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
            'config_summary': self.config.get_summary() # Assuming HybExConfig has get_summary()
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

        # Using a standard style for broad compatibility
        plt.style.use('default') 

        # 1. Loss curves and Learning rate schedule
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss', linewidth=2)
        plt.plot(history['val_losses'], label='Validation Loss', linewidth=2)
        plt.title(f'{model_name} - Loss Curves', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rates'], linewidth=2, color='orange')
        plt.title(f'{model_name} - Learning Rate Schedule', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Learning Rate (log scale)', fontsize=10)
        plt.yscale('log')
        plt.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "loss_and_lr.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Metrics curves
        # Robustly convert list of dicts back to ModelMetrics objects for plotting
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
            plt.title(f'{model_name} - {metric.replace("_", " ").title()}', fontsize=12)
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel(metric.replace("_", " ").title(), fontsize=10)
            plt.legend()
            plt.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training plots to {plots_dir}")

    def train_gnn_model_component(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict[str, Any]:
        """Trains the Knowledge Graph Neural Network (KGNN) component - FIXED"""
        logger.info("--- Starting Knowledge Graph Engine (GNN) Training ---")
        try:
            from .knowledge_graph_engine import KnowledgeGraphEngine
            
            # Initialize GNN engine
            kgengine = KnowledgeGraphEngine(self.config, prolog_engine=None)
            
            # Verify samples have extracted entities
            train_with_entities = [s for s in train_samples if s.get('extracted_entities')]
            val_with_entities = [s for s in val_samples if s.get('extracted_entities')]
            
            if len(train_with_entities) == 0:
                logger.error("No training samples with extracted_entities. Cannot train GNN.")
                return {
                    'gnn_model': {'path': None, 'best_f1': 0.0},
                    'gnn_training': {'status': 'failed', 'reason': 'No samples with entities'}
                }
            
            logger.info(f"Training GNN with {len(train_with_entities)} train, {len(val_with_entities)} val samples")
            
            # Train the GNN
            result = kgengine.train_gnn(
                train_data=train_with_entities,
                val_data=val_with_entities if val_with_entities else None,
                epochs=50
            )
            
            # Save the model
            model_dir = self.config.MODELS_DIR / 'gnn_model'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'gnn_model.pt'
            kgengine.save_model(str(model_path))
            
            logger.info(f"✅ GNN model saved to {model_path}")
            
            return {
                'gnn_model': {
                    'path': str(model_path),
                    'best_f1': result.get('best_val_f1', 0.0)
                },
                'gnn_training': {
                    'status': 'success',
                    'num_nodes': kgengine.graph.number_of_nodes(),
                    'num_edges': kgengine.graph.number_of_edges()
                }
            }
        
        except Exception as e:
            logger.error(f"GNN model training failed: {e}", exc_info=True)
            return {
                'gnn_model': {'path': None, 'best_f1': 0.0},
                'gnn_training': {'status': 'failed', 'reason': str(e)}
            }
    
    def train_all_models(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict[str, Any]:
        """Train all required neural models (Domain Classifier and Eligibility Predictor)."""
        logger.info(f"Training Domain Classifier and Eligibility Predictor models.")

        trained_models_info = {}

        # Train Domain Classifier
        logger.info("\n--- Training Domain Classifier ---")
        domain_classifier = DomainClassifier(self.config)
        domain_config = self.config.MODEL_CONFIGS['domain_classifier']
        domain_train_dataset = LegalDataset(train_samples, self.tokenizer, self.config, task_type="domain_classification", model_config=domain_config)
        domain_val_dataset = LegalDataset(val_samples, self.tokenizer, self.config, task_type="domain_classification", model_config=domain_config)

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
        eligibility_config = self.config.MODEL_CONFIGS['eligibility_predictor']
        eligibility_train_dataset = LegalDataset(train_samples, self.tokenizer, self.config, task_type="eligibility_prediction", model_config=eligibility_config)
        eligibility_val_dataset = LegalDataset(val_samples, self.tokenizer, self.config, task_type="eligibility_prediction", model_config=eligibility_config)

        eligibility_train_loader = DataLoader(eligibility_train_dataset, batch_size=self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size'], shuffle=True)
        eligibility_val_loader = DataLoader(eligibility_val_dataset, batch_size=self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size'], shuffle=False)

        eligibility_results = self.train_model(eligibility_predictor, eligibility_train_loader, eligibility_val_loader, "eligibility_predictor", "eligibility_prediction")
        saved_eligibility_model_path = self.save_model(eligibility_results['model'], "eligibility_predictor", eligibility_results)
        self.create_training_plots(eligibility_results, "eligibility_predictor")

        trained_models_info["eligibility_predictor"] = {
            "path": Path(saved_eligibility_model_path), # Store as Path object
            "best_f1": eligibility_results['best_f1_score']
        }

        # Train EnhancedLegalBERT (multi-task model)
        logger.info("\n--- Training EnhancedLegalBERT (Multi-Task Model) ---")
        enhanced_results = self._train_enhanced_legal_bert(train_samples, val_samples)
        
        trained_models_info["enhanced_legal_bert"] = {
            "path": Path(enhanced_results['model_path']),
            "best_f1": enhanced_results['best_f1']
        }

        logger.info("All neural models (standard + enhanced) trained successfully.")
        return trained_models_info
    
    def _train_enhanced_legal_bert(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict[str, Any]:
        """Train the EnhancedLegalBERT model with multi-task learning."""
        logger.info("Initializing EnhancedLegalBERT with multi-task learning...")
        
        # Get config parameters (with defaults if not defined)
        enhanced_config = self.config.MODEL_CONFIGS.get('enhanced_legal_bert', {
            'batch_size': 8,
            'epochs': 15,
            'learning_rate': 2e-5,
            'max_length': 512,
            'early_stopping_patience': 5
        })
        
        # Initialize model
        model = EnhancedLegalBERT(
            model_name='nlpaueb/legal-bert-base-uncased',
            num_domains=len(self.config.ENTITY_CONFIG['domains']),
            dropout=0.3
        ).to(self.device)
        
        # Initialize trainer
        trainer = EnhancedLegalBERTTrainer(model, device=self.device)
        
        # Create datasets for multi-task learning
        train_dataset = LegalDataset(
            train_samples, 
            self.tokenizer, 
            self.config, 
            task_type="multi_task",
            model_config=enhanced_config
        )
        val_dataset = LegalDataset(
            val_samples, 
            self.tokenizer, 
            self.config, 
            task_type="multi_task",
            model_config=enhanced_config
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=enhanced_config['batch_size'], 
            shuffle=True,
            num_workers=0  # Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=enhanced_config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Training set: {len(train_dataset)} samples")
        logger.info(f"Validation set: {len(val_dataset)} samples")
        logger.info(f"Batch size: {enhanced_config['batch_size']}")
        logger.info(f"Epochs: {enhanced_config['epochs']}")
        
        # Training loop
        best_f1 = 0.0
        best_epoch = 0
        best_model_state = None
        patience_counter = 0
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_f1': [],
            'val_f1': [],
            'eligibility_losses': [],
            'domain_losses': [],
            'confidence_losses': []
        }
        
        epoch_pbar = tqdm(range(enhanced_config['epochs']), desc="Training EnhancedLegalBERT", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            epoch_losses = {
                'total_loss': [],
                'eligibility_loss': [],
                'domain_loss': [],
                'confidence_loss': []
            }
            train_preds = []
            train_labels = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")
            
            for batch in train_pbar:
                losses = trainer.train_step(batch)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
                
                train_pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'elig': f"{losses['eligibility_loss']:.4f}"
                })
                
                # Collect predictions for metrics
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    domains = batch['domains'].cpu().numpy()
                    
                    logits, _, _ = model(input_ids, attention_mask, domains, return_confidence=False)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    train_preds.extend(preds)
                    train_labels.extend(labels)
            
            # Calculate training metrics
            avg_train_loss = np.mean(epoch_losses['total_loss'])
            train_f1 = f1_score(train_labels, train_preds, average='binary')
            
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_f1'].append(train_f1)
            training_history['eligibility_losses'].append(np.mean(epoch_losses['eligibility_loss']))
            training_history['domain_losses'].append(np.mean(epoch_losses['domain_loss']))
            training_history['confidence_losses'].append(np.mean(epoch_losses['confidence_loss']))
            
            # Validation phase
            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False, unit="batch"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    domains = batch['domains'].cpu().numpy()
                    
                    logits, _, _ = model(input_ids, attention_mask, domains, return_confidence=False)
                    
                    # Calculate loss
                    loss = trainer.eligibility_criterion(logits, labels)
                    val_losses.append(loss.item())
                    
                    # Predictions
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate validation metrics
            avg_val_loss = np.mean(val_losses)
            val_f1 = f1_score(val_labels, val_preds, average='binary')
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
            val_recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
            
            training_history['val_losses'].append(avg_val_loss)
            training_history['val_f1'].append(val_f1)
            
            logger.info(f"\nEpoch {epoch+1}/{enhanced_config['epochs']}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
            logger.info(f"  Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            logger.info(f"  Component Losses - Eligibility: {training_history['eligibility_losses'][-1]:.4f}, "
                       f"Domain: {training_history['domain_losses'][-1]:.4f}, "
                       f"Confidence: {training_history['confidence_losses'][-1]:.4f}")
            
            # Early stopping and model checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logger.info(f"  ✅ New best model! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  No improvement. Patience: {patience_counter}/{enhanced_config['early_stopping_patience']}")
            
            if patience_counter >= enhanced_config['early_stopping_patience']:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            epoch_pbar.set_postfix({
                'best_f1': f"{best_f1:.4f}",
                'val_f1': f"{val_f1:.4f}"
            })
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"\n✅ Restored best model from epoch {best_epoch} (F1: {best_f1:.4f})")
        
        # Save model
        save_dir = self.config.MODELS_DIR / 'enhanced_legal_bert'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / 'enhanced_legal_bert_best.pt'
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_bert_state_dict': trainer.optimizer_bert.state_dict(),
            'optimizer_head_state_dict': trainer.optimizer_head.state_dict(),
            'best_f1': best_f1,
            'training_history': training_history,
            'config': enhanced_config
        }, model_path)
        
        logger.info(f"✅ EnhancedLegalBERT saved to {model_path}")
        
        # Create training plots
        self._create_enhanced_training_plots(training_history, 'enhanced_legal_bert')
        
        return {
            'model': model,
            'model_path': str(model_path),
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'training_history': training_history
        }
    
    def _create_enhanced_training_plots(self, history: Dict, model_name: str):
        """Create training plots for EnhancedLegalBERT."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Loss plot
            axes[0, 0].plot(history['train_losses'], label='Train Loss', marker='o')
            axes[0, 0].plot(history['val_losses'], label='Val Loss', marker='s')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # F1 Score plot
            axes[0, 1].plot(history['train_f1'], label='Train F1', marker='o')
            axes[0, 1].plot(history['val_f1'], label='Val F1', marker='s')
            axes[0, 1].set_title('Training and Validation F1 Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Component losses
            axes[1, 0].plot(history['eligibility_losses'], label='Eligibility Loss', marker='o')
            axes[1, 0].plot(history['domain_losses'], label='Domain Loss', marker='s')
            axes[1, 0].plot(history['confidence_losses'], label='Confidence Loss', marker='^')
            axes[1, 0].set_title('Multi-Task Component Losses')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Best F1 marker
            best_idx = np.argmax(history['val_f1'])
            axes[1, 1].bar(['Best Val F1'], [history['val_f1'][best_idx]], color='green', alpha=0.7)
            axes[1, 1].axhline(y=history['val_f1'][best_idx], color='r', linestyle='--', 
                             label=f'Best: {history["val_f1"][best_idx]:.4f}')
            axes[1, 1].set_title('Best Validation F1 Score')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_dir = self.config.RESULTS_DIR / 'training_plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f'{model_name}_training_curves.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create training plots: {e}")
            logger.exception("Plot creation error")


# ============================================================================
# ENHANCED LEGAL-BERT ARCHITECTURE
# ============================================================================

class EnhancedLegalBERT(nn.Module):
    """
    Enhanced BERT model with:
    1. Legal domain adaptation
    2. Multi-task learning (eligibility + domain classification)
    3. Attention pooling
    4. Uncertainty estimation
    """
    
    def __init__(self, model_name='nlpaueb/legal-bert-base-uncased', 
                 num_domains=5, dropout=0.3):
        super().__init__()
        
        # Load pre-trained legal BERT (better than vanilla BERT for legal text)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size  # 768
        
        # Freeze bottom 6 layers (keep legal knowledge, fine-tune top layers)
        for param in self.bert.encoder.layer[:6].parameters():
            param.requires_grad = False
        
        # Attention-based pooling (better than CLS token alone)
        self.attention_weights = nn.Linear(hidden_size, 1)
        
        # Multi-head attention for better representation
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout
        )
        
        # Domain-specific projection layers
        self.domain_projection = nn.ModuleDict({
            'legalaid': nn.Linear(hidden_size, hidden_size),
            'familylaw': nn.Linear(hidden_size, hidden_size),
            'consumerprotection': nn.Linear(hidden_size, hidden_size),
            'employmentlaw': nn.Linear(hidden_size, hidden_size),
            'fundamentalrights': nn.Linear(hidden_size, hidden_size)
        })
        
        # Main classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2)  # Binary: eligible/not_eligible
        )
        
        # Auxiliary task: Domain classification (multi-task learning)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output: 0-1 confidence score
        )
        
    def attention_pooling(self, hidden_states, attention_mask):
        """
        Attention-weighted pooling over sequence.
        Better than just using [CLS] token.
        """
        # hidden_states: [batch, seq_len, hidden]
        # attention_mask: [batch, seq_len]
        
        # Calculate attention scores
        attention_scores = self.attention_weights(hidden_states).squeeze(-1)
        # attention_scores: [batch, seq_len]
        
        # Mask padded tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float('-inf')
        )
        
        # Softmax to get attention distribution
        attention_probs = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        # attention_probs: [batch, seq_len, 1]
        
        # Weighted sum
        pooled = torch.sum(hidden_states * attention_probs, dim=1)
        # pooled: [batch, hidden]
        
        return pooled
    
    def forward(self, input_ids, attention_mask, domains=None, return_confidence=False):
        """
        Forward pass with multi-task learning.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            domains: [batch] - domain indices for each sample
            return_confidence: Whether to return confidence scores
        
        Returns:
            logits: [batch, 2] - eligibility scores
            domain_logits: [batch, num_domains] - domain classification scores
            confidence: [batch] - confidence scores (if return_confidence=True)
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states from last layer
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Attention pooling
        pooled = self.attention_pooling(hidden_states, attention_mask)
        # pooled: [batch, hidden]
        
        # Multi-head self-attention for better representation
        pooled_attn, _ = self.multihead_attn(
            pooled.unsqueeze(0),  # [1, batch, hidden]
            pooled.unsqueeze(0),
            pooled.unsqueeze(0)
        )
        pooled_attn = pooled_attn.squeeze(0)  # [batch, hidden]
        
        # Combine original and attention-enhanced representations
        combined = pooled + pooled_attn
        
        # Domain-specific projection (if domains provided)
        if domains is not None:
            domain_enhanced = []
            for i, domain_idx in enumerate(domains):
                domain_name = ['legalaid', 'familylaw', 'consumerprotection', 
                               'employmentlaw', 'fundamentalrights'][domain_idx]
                proj = self.domain_projection[domain_name](combined[i:i+1])
                domain_enhanced.append(proj)
            combined = torch.cat(domain_enhanced, dim=0)
        
        # Main task: Eligibility classification
        logits = self.classifier(combined)
        
        # Auxiliary task: Domain classification
        domain_logits = self.domain_classifier(combined)
        
        # Confidence estimation
        confidence = None
        if return_confidence:
            confidence = self.confidence_head(combined).squeeze(-1)
        
        return logits, domain_logits, confidence


# ============================================================================
# ENHANCED TRAINER WITH MULTI-TASK LEARNING
# ============================================================================

class EnhancedLegalBERTTrainer:
    """Enhanced trainer with better training strategies."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Separate optimizers for different parts
        self.optimizer_bert = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if 'bert' in n],
            lr=2e-5, weight_decay=0.01
        )
        
        self.optimizer_head = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if 'bert' not in n],
            lr=5e-5, weight_decay=0.01
        )
        
        # Learning rate schedulers
        self.scheduler_bert = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_bert, T_0=10, T_mult=2
        )
        
        self.scheduler_head = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_head, T_0=10, T_mult=2
        )
        
        # Loss functions
        self.eligibility_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.2, 1.0]).to(device)  # Slightly weight NOT ELIGIBLE more
        )
        self.domain_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
    def train_step(self, batch):
        """Enhanced training step with multi-task learning."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        eligibility_labels = batch['labels'].to(self.device)
        domain_labels = batch['domains'].to(self.device)
        
        # Forward pass
        logits, domain_logits, confidence = self.model(
            input_ids, attention_mask, 
            domains=domain_labels.cpu().numpy(),
            return_confidence=True
        )
        
        # Calculate losses
        eligibility_loss = self.eligibility_criterion(logits, eligibility_labels)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)
        
        # Confidence target: 1.0 if prediction is correct, 0.5 if wrong
        preds = torch.argmax(logits, dim=-1)
        confidence_target = (preds == eligibility_labels).float()
        confidence_loss = self.confidence_criterion(confidence, confidence_target)
        
        # Combined loss (weighted)
        total_loss = (
            0.7 * eligibility_loss +  # Main task
            0.2 * domain_loss +        # Auxiliary task
            0.1 * confidence_loss      # Calibration
        )
        
        # Backward pass
        self.optimizer_bert.zero_grad()
        self.optimizer_head.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimize
        self.optimizer_bert.step()
        self.optimizer_head.step()
        
        # Update learning rates
        self.scheduler_bert.step()
        self.scheduler_head.step()
        
        return {
            'total_loss': total_loss.item(),
            'eligibility_loss': eligibility_loss.item(),
            'domain_loss': domain_loss.item(),
            'confidence_loss': confidence_loss.item()
        }