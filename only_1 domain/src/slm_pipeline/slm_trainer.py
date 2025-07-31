"""
SLM Trainer - Training and fine-tuning utilities for Small Language Models.

This module provides comprehensive training infrastructure for fine-tuning
SLMs on legal fact extraction tasks.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for SLM fine-tuning."""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "models/slm_legal_facts"
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_wandb: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True


class LegalFactDataset(Dataset):
    """Dataset class for legal fact extraction training."""
    
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer, 
        max_length: int = 512,
        system_prompt: str = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of training examples with 'query' and 'facts' keys
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            system_prompt: System prompt for fact extraction
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Preprocess all examples
        self.processed_data = self._preprocess_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} examples")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for training."""
        return """You are a legal fact extraction specialist. Extract structured legal facts from natural language queries and output them in Prolog format.

Extract these facts when present:
- applicant(user).
- income_monthly(user, Amount).
- case_type(user, 'Type').
- is_woman(user, Boolean).
- is_sc_st(user, Boolean).
- is_child(user, Boolean).
- is_disabled(user, Boolean).

Output only the Prolog facts, one per line."""
    
    def _preprocess_data(self) -> List[Dict[str, Any]]:
        """Preprocess all training examples."""
        processed = []
        
        for example in self.data:
            # Format the conversation
            query = example['query']
            facts = example.get('facts', [])
            
            # Create conversation format
            conversation = f"<|system|>\n{self.system_prompt}\n\n<|user|>\nQuery: {query}\n\n<|assistant|>\n"
            conversation += "\n".join(facts)
            
            # Tokenize
            tokens = self.tokenizer(
                conversation,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            processed.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze().clone()
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]


class SLMTrainer:
    """
    Trainer class for fine-tuning SLMs on legal fact extraction.
    
    This class handles the complete training pipeline including:
    - Data preparation and validation
    - Model setup with LoRA fine-tuning
    - Training loop with monitoring
    - Model evaluation and saving
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Setup output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="hybex-law-slm",
                config=config.__dict__
            )
        
        logger.info(f"SLM Trainer initialized with config: {config}")
    
    def prepare_model(self):
        """Load and prepare the model for training."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization for efficient training
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA if requested
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model prepared for training")
    
    def prepare_datasets(
        self, 
        train_data: List[Dict[str, Any]], 
        eval_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and evaluation datasets.
        
        Args:
            train_data: Training examples
            eval_data: Evaluation examples (optional)
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info(f"Preparing datasets: {len(train_data)} train, {len(eval_data) if eval_data else 0} eval")
        
        # Create datasets
        train_dataset = LegalFactDataset(
            train_data, 
            self.tokenizer, 
            self.config.max_length
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = LegalFactDataset(
                eval_data,
                self.tokenizer,
                self.config.max_length
            )
        
        return train_dataset, eval_dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup the Hugging Face Trainer."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb else None
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("Trainer setup complete")
    
    def train(
        self, 
        train_data: List[Dict[str, Any]], 
        eval_data: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Execute the complete training pipeline.
        
        Args:
            train_data: Training examples
            eval_data: Evaluation examples (optional)
        """
        logger.info("Starting SLM training pipeline")
        
        # Prepare model
        if self.model is None:
            self.prepare_model()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(train_data, eval_data)
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        start_time = time.time()
        self.trainer.train()
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_model()
        
        # Log training summary
        if self.config.use_wandb:
            wandb.log({
                "training_time_seconds": training_time,
                "final_train_loss": self.trainer.state.log_history[-1].get("train_loss", 0)
            })
    
    def save_model(self, path: Optional[str] = None):
        """Save the fine-tuned model."""
        save_path = path or self.config.output_dir
        
        logger.info(f"Saving model to: {save_path}")
        
        # Save the model
        self.trainer.save_model(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save training config
        config_path = Path(save_path) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info("Model saved successfully")
    
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            eval_data: Evaluation examples
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating model on {len(eval_data)} examples")
        
        # Prepare evaluation dataset
        eval_dataset = LegalFactDataset(
            eval_data,
            self.tokenizer,
            self.config.max_length
        )
        
        # Run evaluation
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info(f"Evaluation results: {eval_results}")
        
        return eval_results
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        config: TrainingConfig
    ) -> 'SLMTrainer':
        """
        Load a trainer with a pre-trained model.
        
        Args:
            model_path: Path to the saved model
            config: Training configuration
            
        Returns:
            Initialized trainer with loaded model
        """
        trainer = cls(config)
        
        # Load model and tokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_path)
        trainer.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        logger.info(f"Trainer loaded from: {model_path}")
        
        return trainer


# Utility functions for data preparation
def prepare_training_data_from_samples(sample_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert sample data format to training format.
    
    Args:
        sample_data: Data in sample format (with 'query' and 'expected_facts')
        
    Returns:
        Data formatted for training
    """
    training_data = []
    
    for sample in sample_data:
        training_example = {
            'query': sample['query'],
            'facts': sample.get('expected_facts', [])
        }
        training_data.append(training_example)
    
    return training_data


def create_data_splits(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.8, 
    eval_ratio: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train/eval/test sets.
    
    Args:
        data: Complete dataset
        train_ratio: Ratio for training set
        eval_ratio: Ratio for evaluation set
        
    Returns:
        Tuple of (train_data, eval_data, test_data)
    """
    import random
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    
    # Split data
    train_data = shuffled_data[:train_size]
    eval_data = shuffled_data[train_size:train_size + eval_size]
    test_data = shuffled_data[train_size + eval_size:]
    
    logger.info(f"Data split: {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test")
    
    return train_data, eval_data, test_data


# Example usage
if __name__ == "__main__":
    # Example training configuration
    config = TrainingConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        output_dir="models/slm_legal_facts",
        batch_size=2,  # Small batch for demonstration
        num_epochs=1,
        learning_rate=5e-5,
        use_lora=True,
        use_wandb=False  # Set to True for experiment tracking
    )
    
    # Example training data
    sample_training_data = [
        {
            'query': "I am a woman facing domestic violence. I earn 15000 rupees monthly.",
            'facts': [
                'applicant(user).',
                'income_monthly(user, 15000).',
                "case_type(user, 'domestic_violence').",
                'is_woman(user, true).'
            ]
        },
        {
            'query': "My landlord is evicting me. I am unemployed and from scheduled caste.",
            'facts': [
                'applicant(user).',
                'income_monthly(user, 0).',
                "case_type(user, 'property_dispute').",
                'is_sc_st(user, true).'
            ]
        }
    ]
    
    print("🧪 SLM Training Example")
    print("=" * 50)
    print(f"Training on {len(sample_training_data)} examples")
    
    try:
        # Initialize trainer
        trainer = SLMTrainer(config)
        
        # Train model (commented out for safety)
        # trainer.train(sample_training_data)
        
        print("✅ Trainer initialized successfully")
        print("Note: Actual training requires GPU and proper setup")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: This requires proper SLM dependencies and GPU access")
