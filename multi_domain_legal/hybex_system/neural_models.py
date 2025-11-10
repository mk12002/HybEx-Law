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

    def forward(self, input_ids, attention_mask, return_dict=False):
        """
        [FIXED 11/09/2025]
        - This forward signature expects positional args.
        - The "return_dict" arg is for compatibility with the trainer.
        - Always return a dictionary to standardize output for all callers.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] # CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Always return a dictionary for consistent output
        return {'logits': logits}

class DomainToEligibilityWrapper(nn.Module):
    """
    Wrapper to use Domain Classifier for eligibility prediction
    Uses domain confidence as eligibility proxy
    """
    def __init__(self, domain_classifier, config):
        super().__init__()
        self.domain_classifier = domain_classifier
        self.config = config
        if hasattr(domain_classifier, 'classifier'):
            input_dim = domain_classifier.classifier.out_features
        elif hasattr(domain_classifier, 'output_layer'):
            input_dim = domain_classifier.output_layer.out_features
        else:
            try:
                dummy_input = torch.randn(1, 512).to(next(domain_classifier.parameters()).device)
                dummy_output = domain_classifier(
                    input_ids=torch.ones(1, 512, dtype=torch.long).to(dummy_input.device),
                    attention_mask=torch.ones(1, 512, dtype=torch.long).to(dummy_input.device)
                )
                if isinstance(dummy_output, dict):
                    input_dim = dummy_output['logits'].shape[-1]
                else:
                    input_dim = dummy_output.shape[-1]
            except:
                logger.warning("Could not infer domain classifier output dim, using 7")
                input_dim = 7
        self.eligibility_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary eligibility
        )

    def forward(self, input_ids, attention_mask, return_dict=False):
        """
        [FIXED 11/09/2025]
        - This forward signature expects positional args.
        - Always return a dictionary to standardize output for all callers.
        """
        # Get domain predictions (frozen)
        with torch.no_grad():
            domain_output = self.domain_classifier(input_ids, attention_mask, return_dict=True)
            domain_logits = domain_output['logits']

        # Map to eligibility
        eligibility_logits = self.eligibility_head(domain_logits)
        
        # Always return a dictionary for consistent output
        return {'logits': eligibility_logits}

class EligibilityPredictor(nn.Module):
    """Binary classification model for legal aid eligibility."""
    def __init__(self, config: HybExConfig):
        super().__init__()
        self.config = config
        model_config = config.get_model_config('eligibility_predictor') # <-- Get specific config
        self.base_model = AutoModel.from_pretrained(model_config['model_name'])
        self.dropout = nn.Dropout(model_config['dropout_prob'])
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 1) # Binary classification

    def forward(self, input_ids, attention_mask, return_dict=False):
        """
        [FIXED 11/09/2025]
        - This forward signature expects positional args.
        - The "return_dict" arg is for compatibility with the trainer.
        - Always return a dictionary to standardize output for all callers.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] # CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(-1)
        
        # Always return a dictionary for consistent output
        return {'logits': logits}

class EnhancedLegalBERT(nn.Module):
    """
    Enhanced Multi-task Legal BERT model with:
    1. Legal domain adaptation
    2. Multi-task learning (eligibility + domain classification)
    3. Attention pooling
    4. Uncertainty estimation
    5. Domain-specific projections
    """
    def __init__(self, config: HybExConfig):
        super().__init__()
        
        self.config_obj = config
        model_name = config.MODEL_CONFIG.get('base_model', 'nlpaueb/legal-bert-base-uncased')
        num_domains = len(config.ENTITY_CONFIG.get('domains', []))
        if num_domains == 0:
            logger.warning("No domains found in config, defaulting to 5")
            num_domains = 5
            
        enhanced_config = config.MODEL_CONFIGS.get('enhanced_legal_bert', {})
        dropout = enhanced_config.get('dropout_prob', 0.3)
        freeze_layers = enhanced_config.get('freeze_bottom_layers', 6)
        num_heads = enhanced_config.get('num_attention_heads', 8)

        self.bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.bert_config.hidden_size
        
        if freeze_layers > 0:
            logger.info(f"Freezing bottom {freeze_layers} BERT layers")
            for param in self.bert.encoder.layer[:freeze_layers].parameters():
                param.requires_grad = False
        
        self.attention_weights = nn.Linear(hidden_size, 1)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.domain_names = config.ENTITY_CONFIG.get('domains', ['legal_aid', 'family_law', 'consumer_protection', 'employment_law', 'fundamental_rights'])
        self.domain_projection = nn.ModuleDict({
            domain: nn.Linear(hidden_size, hidden_size) for domain in self.domain_names
        })
        
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
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.num_domains = num_domains
    
    def attention_pooling(self, hidden_states, attention_mask):
        """
        Attention-weighted pooling over sequence.
        """
        attention_scores = self.attention_weights(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float('-inf')
        )
        attention_probs = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(hidden_states * attention_probs, dim=1)
        return pooled
    
    def forward(self, input_ids, attention_mask, domains=None, return_dict=False, return_confidence=False):
        """
        Forward pass with multi-task learning.
        [NOTE] This model *is* designed to accept keyword arguments,
        so its call signature is different from the simpler nn.Modules.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state
        pooled = self.attention_pooling(hidden_states, attention_mask)
        
        pooled_attn, _ = self.multihead_attn(
            pooled.unsqueeze(1),
            pooled.unsqueeze(1),
            pooled.unsqueeze(1),
            key_padding_mask=None
        )
        pooled_attn = pooled_attn.squeeze(1)
        
        combined = pooled + pooled_attn
        
        if domains is not None:
            domain_enhanced = []
            
            if isinstance(domains, torch.Tensor):
                domains_list = domains.cpu().tolist()
            else:
                domains_list = domains
                
            for i, domain_idx in enumerate(domains_list):
                domain_idx_int = int(domain_idx)
                domain_name = self.domain_names[domain_idx_int % len(self.domain_names)] 
                proj = self.domain_projection[domain_name](combined[i:i+1])
                domain_enhanced.append(proj)
            combined = torch.cat(domain_enhanced, dim=0)
        
        logits = self.classifier(combined)
        domain_logits = self.domain_classifier(combined)
        
        confidence = None
        if return_confidence:
            confidence = self.confidence_head(combined).squeeze(-1)
        
        if return_dict:
            return {
                'logits': logits,
                'domain_logits': domain_logits,
                'eligibility_logits': logits,  # Alias for compatibility
                'confidence': confidence
            }
        return logits, domain_logits, confidence

class LegalDataset(Dataset):
    """PyTorch Dataset for legal text processing"""
    def __init__(self, samples: List[Dict], tokenizer, config: HybExConfig, task_type: str = "domain_classification", model_config: Dict[str, Any] = None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.task_type = task_type
        self.max_length = model_config.get('max_length', 512) if model_config else 512 

        if self.task_type == "domain_classification":
            for sample in self.samples:
                if not isinstance(sample.get('domains'), list):
                    sample['domains'] = [] 

        logger.info(f"Created {task_type} dataset with {len(samples)} samples")

        try:
            logger.debug("\n🔍 DEBUG: First 5 training samples:")
            for i, sample in enumerate(self.samples[:5]):
                logger.debug(f"\nSample {i}:")
                logger.debug(f"  Query: {sample.get('query', 'MISSING')[:100]}...")
                logger.debug(f"  Domains: {sample.get('domains', 'MISSING')}")
                logger.debug(f"  Eligibility: {sample.get('expected_eligibility', 'MISSING')}")
                logger.debug(f"  Keys: {list(sample.keys())}")

            leaked_keys = ['extracted_facts', 'prolog_facts', 'user_demographics', 'income', 'social_category']
            for key in leaked_keys:
                if key in self.samples[0]:
                    logger.warning(f"⚠️  WARNING: Found potential data leakage key: '{key}' in samples[0]")
        except Exception:
            logger.exception("Debug print failed in LegalDataset.__init__")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample['query'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        if self.task_type == "domain_classification":
            labels = torch.zeros(len(self.config.ENTITY_CONFIG['domains']), dtype=torch.float)
            for i, domain in enumerate(self.config.ENTITY_CONFIG['domains']):
                if domain in sample['domains']:
                    labels[i] = 1.0
        elif self.task_type == "eligibility_prediction":
            eligibility_value = float(sample.get('expected_eligibility', 0.0))
            labels = torch.tensor(eligibility_value, dtype=torch.float)
        elif self.task_type == "multi_task":
            eligibility_value = float(sample.get('expected_eligibility', 0.0))
            labels = torch.tensor(eligibility_value, dtype=torch.long)
        else:
            labels = torch.tensor(0.0, dtype=torch.float)

        domain_labels = torch.zeros(len(self.config.ENTITY_CONFIG['domains']), dtype=torch.float)
        domain_indices = []
        for i, domain in enumerate(self.config.ENTITY_CONFIG['domains']):
            if domain in sample.get('domains', []):
                domain_labels[i] = 1.0
                domain_indices.append(i)
        
        primary_domain_idx = domain_indices[0] if domain_indices else 0

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'domains': torch.tensor(primary_domain_idx, dtype=torch.long),
            'domain_labels': domain_labels,
            'sample_id': sample.get('sample_id', idx)
        }

class ModelTrainer:
    """Orchestrates the training of neural models."""

    def __init__(self, config: HybExConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])

        if self.tokenizer.pad_token is None:
            if self.tokenizer.pad_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
            else:
                added = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if added:
                    logger.info("Added a new [PAD] token to the tokenizer (will require model.resize_token_embeddings after model creation).")
                else:
                    logger.warning("Attempted to add [PAD] token but tokenizer reported 0 added tokens.")

        self.setup_logging()
        logger.info(f"ModelTrainer initialized on device: {self.device}")

    def setup_logging(self):
        log_file = self.config.get_log_path('neural_training')
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
        return self.config.MODEL_CONFIGS.get(model_name, self.config.MODEL_CONFIGS.get('domain_classifier', {}))

    def _resize_model_embeddings_if_tokenizer_changed(self, model):
        """
        If the tokenizer was updated (added tokens), ensure the model embedding matrix is resized.
        """
        try:
            if hasattr(self.tokenizer, 'added_tokens_encoder') and len(self.tokenizer.added_tokens_encoder) > 0:
                new_vocab_size = len(self.tokenizer)
                if hasattr(model, 'resize_token_embeddings'):
                    model.resize_token_embeddings(new_vocab_size)
                    logger.info(f"Resized model embeddings to {new_vocab_size} tokens after tokenizer change.")
        except Exception as e:
            logger.warning(f"Could not resize model embeddings automatically: {e}")

    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    model_name: str, task_type: str) -> Dict[str, Any]:
        """Train a model with comprehensive monitoring, gradient accumulation, and mixed precision."""

        model_config = self.get_model_config(model_name)
        model.to(self.device)
        
        ACCUMULATION_STEPS = 4
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=model_config['early_stopping_patience'] // 2
        )

        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        if task_type == "domain_classification":
            criterion = nn.BCEWithLogitsLoss()
        elif task_type == "eligibility_prediction":
            try:
                labels = [s.get('expected_eligibility', 0) for s in train_loader.dataset.samples]
                pos_count = sum(labels)
                neg_count = len(labels) - pos_count
                
                pos_weight_value = neg_count / max(1, pos_count)
                
                pos_weight = torch.tensor([pos_weight_value], device=self.device)
                logger.info(f"Using pos_weight for eligibility: {pos_weight.item():.2f} (Pos: {pos_count}, Neg: {neg_count})")
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            except Exception as e:
                logger.warning(f"Could not calculate pos_weight, using unweighted loss. Error: {e}")
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
        if scaler.is_enabled():
            logger.info("Mixed precision training enabled (FP16)")

        epoch_pbar = tqdm(range(model_config['epochs']), desc=f"Training {model_name}", unit="epoch")

        for epoch in epoch_pbar:
            final_epoch = epoch + 1
            model.train()
            total_train_loss = 0
            train_preds, train_labels = [], []

            optimizer.zero_grad()

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    # Apply return_dict=True to the model call
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    logits = outputs['logits']
                    loss = criterion(logits, labels)
                    loss = loss / ACCUMULATION_STEPS
                
                if task_type == "eligibility_prediction":
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                elif task_type == "domain_classification":
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

                scaler.scale(loss).backward()

                if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['gradient_clip_val'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * ACCUMULATION_STEPS

                train_pbar.set_postfix({'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}'})

                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())

            avg_train_loss = total_train_loss / len(train_loader)
            train_metrics = self.calculate_metrics(train_preds, train_labels, task_type)
            train_metrics.loss = avg_train_loss
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_metrics'].append(train_metrics)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            model.eval()
            total_val_loss = 0
            val_preds, val_labels = [], []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False, unit="batch")

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    logits = outputs['logits']

                    loss = criterion(logits, labels)

                    if task_type == "eligibility_prediction":
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    elif task_type == "domain_classification":
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

                    total_val_loss += loss.item()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

                    val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

            avg_val_loss = total_val_loss / len(val_loader)
            val_metrics = self.calculate_metrics(val_preds, val_labels, task_type)
            val_metrics.loss = avg_val_loss
            training_history['val_losses'].append(avg_val_loss)
            training_history['val_metrics'].append(val_metrics)

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

            scheduler.step(val_metrics.f1_score)

            if val_metrics.f1_score > best_val_f1:
                best_val_f1 = val_metrics.f1_score
                best_model_state = model.state_dict().copy() 
                patience_counter = 0
                logger.info(f"New best validation F1 score: {best_val_f1:.4f}. Saving model state.")
            else:
                patience_counter += 1
                logger.info(f"Validation F1 did not improve. Patience: {patience_counter}/{model_config['early_stopping_patience']}")
                if patience_counter >= model_config['early_stopping_patience']:
                    logger.info("Early stopping triggered.")
                    break

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
        labels_np = np.array(labels)
        predictions_np = np.array(predictions)

        if task_type == "eligibility_prediction" and labels_np.ndim > 1 and labels_np.shape[1] == 1:
            labels_np = labels_np.squeeze()
            predictions_np = predictions_np.squeeze()

        if task_type == "domain_classification":
            accuracy = accuracy_score(labels_np, predictions_np)
            f1 = f1_score(labels_np, predictions_np, average='macro', zero_division=0)
            precision = precision_score(labels_np, predictions_np, average='macro', zero_division=0)
            recall = recall_score(labels_np, predictions_np, average='macro', zero_division=0)

            try:
                target_names = self.config.ENTITY_CONFIG['domains']
                class_report = classification_report(labels_np, predictions_np, target_names=target_names, output_dict=True, zero_division=0)
            except Exception as e:
                logger.warning(f"Could not generate classification report for domain classification: {e}")
                class_report = {}
            
            cm = None 

        elif task_type == "eligibility_prediction":
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

            cm = confusion_matrix(labels_np, predictions_np).tolist()
        
        else:
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
            'model_config': self.get_model_config(model_name)
        }

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

        plt.style.use('default') 

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
            
            kgengine = KnowledgeGraphEngine(self.config, prolog_engine=None)
            
            train_with_entities = [s for s in train_samples if s.get('extracted_entities')]
            val_with_entities = [s for s in val_samples if s.get('extracted_entities')]
            
            if len(train_with_entities) == 0:
                logger.error("No training samples with extracted_entities. Cannot train GNN.")
                return {
                    'gnn_model': {'path': None, 'best_f1': 0.0},
                    'gnn_training': {'status': 'failed', 'reason': 'No samples with entities'}
                }
            
            logger.info(f"Training GNN with {len(train_with_entities)} train, {len(val_with_entities)} val samples")
            
            result = kgengine.train_gnn(
                train_data=train_with_entities,
                val_data=val_with_entities if val_with_entities else None,
                epochs=50
            )
            
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
        self._resize_model_embeddings_if_tokenizer_changed(domain_classifier.base_model)
        
        domain_config = self.config.MODEL_CONFIGS['domain_classifier']
        domain_train_dataset = LegalDataset(train_samples, self.tokenizer, self.config, task_type="domain_classification", model_config=domain_config)
        domain_val_dataset = LegalDataset(val_samples, self.tokenizer, self.config, task_type="domain_classification", model_config=domain_config)

        domain_train_loader = DataLoader(domain_train_dataset, batch_size=self.config.MODEL_CONFIGS['domain_classifier']['batch_size'], shuffle=True)
        domain_val_loader = DataLoader(domain_val_dataset, batch_size=self.config.MODEL_CONFIGS['domain_classifier']['batch_size'], shuffle=False)

        domain_results = self.train_model(domain_classifier, domain_train_loader, domain_val_loader, "domain_classifier", "domain_classification")
        saved_domain_model_path = self.save_model(domain_results['model'], "domain_classifier", domain_results)
        self.create_training_plots(domain_results, "domain_classifier")

        trained_models_info["domain_classifier"] = {
            "path": Path(saved_domain_model_path),
            "best_f1": domain_results['best_f1_score']
        }

        # Train Eligibility Predictor
        logger.info("\n--- Training Eligibility Predictor ---")
        eligibility_predictor = EligibilityPredictor(self.config)
        self._resize_model_embeddings_if_tokenizer_changed(eligibility_predictor.base_model)
        
        eligibility_config = self.config.MODEL_CONFIGS['eligibility_predictor']
        eligibility_train_dataset = LegalDataset(train_samples, self.tokenizer, self.config, task_type="eligibility_prediction", model_config=eligibility_config)
        eligibility_val_dataset = LegalDataset(val_samples, self.tokenizer, self.config, task_type="eligibility_prediction", model_config=eligibility_config)

        eligibility_train_loader = DataLoader(eligibility_train_dataset, batch_size=self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size'], shuffle=True)
        eligibility_val_loader = DataLoader(eligibility_val_dataset, batch_size=self.config.MODEL_CONFIGS['eligibility_predictor']['batch_size'], shuffle=False)

        eligibility_results = self.train_model(eligibility_predictor, eligibility_train_loader, eligibility_val_loader, "eligibility_predictor", "eligibility_prediction")
        saved_eligibility_model_path = self.save_model(eligibility_results['model'], "eligibility_predictor", eligibility_results)
        self.create_training_plots(eligibility_results, "eligibility_predictor")

        trained_models_info["eligibility_predictor"] = {
            "path": Path(saved_eligibility_model_path),
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
        
        enhanced_config = self.config.MODEL_CONFIGS.get('enhanced_legal_bert', {
            'batch_size': 8,
            'epochs': 15,
            'learning_rate': 2e-5,
            'max_length': 512,
            'early_stopping_patience': 5
        })
        
        model = EnhancedLegalBERT(self.config).to(self.device)
        self._resize_model_embeddings_if_tokenizer_changed(model.bert)
        
        trainer = EnhancedLegalBERTTrainer(model, device=self.device)
        
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
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=enhanced_config['batch_size'], 
            shuffle=True,
            num_workers=0
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
                
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    domains = batch['domains'].cpu().numpy()
                    
                    logits, _, _ = model(input_ids, attention_mask, domains, return_confidence=False)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    train_preds.extend(preds)
                    train_labels.extend(labels)
            
            avg_train_loss = np.mean(epoch_losses['total_loss'])
            train_f1 = f1_score(train_labels, train_preds, average='binary')
            
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_f1'].append(train_f1)
            training_history['eligibility_losses'].append(np.mean(epoch_losses['eligibility_loss']))
            training_history['domain_losses'].append(np.mean(epoch_losses['domain_loss']))
            training_history['confidence_losses'].append(np.mean(epoch_losses['confidence_loss']))
            
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
                    
                    loss = trainer.eligibility_criterion(logits, labels)
                    val_losses.append(loss.item())
                    
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
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
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"\n✅ Restored best model from epoch {best_epoch} (F1: {best_f1:.4f})")
        
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
# (Moved to before ModelTrainer)
# ============================================================================

# ============================================================================
# ENHANCED TRAINER WITH MULTI-TASK LEARNING
# (Moved to before ModelTrainer)
# ============================================================================

class EnhancedLegalBERTTrainer:
    """Enhanced trainer with better training strategies."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer_bert = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if 'bert' in n],
            lr=2e-5, weight_decay=0.01
        )
        
        self.optimizer_head = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if 'bert' not in n],
            lr=5e-5, weight_decay=0.01
        )
        
        self.scheduler_bert = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_bert, T_0=10, T_mult=2
        )
        
        self.scheduler_head = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_head, T_0=10, T_mult=2
        )
        
        self.eligibility_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.2, 1.0]).to(device)
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
        
        logits, domain_logits, confidence = self.model(
            input_ids, attention_mask, 
            domains=domain_labels.cpu().numpy(),
            return_confidence=True
        )
        
        eligibility_loss = self.eligibility_criterion(logits, eligibility_labels)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)
        
        preds = torch.argmax(logits, dim=-1)
        confidence_target = (preds == eligibility_labels).float()
        confidence_loss = self.confidence_criterion(confidence, confidence_target)
        
        total_loss = (
            0.7 * eligibility_loss +
            0.2 * domain_loss +
            0.1 * confidence_loss
        )
        
        self.optimizer_bert.zero_grad()
        self.optimizer_head.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer_bert.step()
        self.optimizer_head.step()
        
        self.scheduler_bert.step()
        self.scheduler_head.step()
        
        return {
            'total_loss': total_loss.item(),
            'eligibility_loss': eligibility_loss.item(),
            'domain_loss': domain_loss.item(),
            'confidence_loss': confidence_loss.item()
        }