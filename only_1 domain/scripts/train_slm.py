"""
SLM Training Script - End-to-end training pipeline for legal fact extraction.

This script provides a complete training pipeline for fine-tuning Small Language
Models on legal fact extraction tasks.

Usage:
    python scripts/train_slm.py --config config/slm_training_config.json
    python scripts/train_slm.py --model microsoft/Phi-3-mini-4k-instruct --epochs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.slm_pipeline.slm_trainer import SLMTrainer, TrainingConfig, prepare_training_data_from_samples, create_data_splits
from src.data_generation.legal_data_generator import LegalDataGenerator
from data.sample_data import SAMPLE_QUERIES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return TrainingConfig(**config_dict)


def generate_training_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic training data."""
    logger.info(f"Generating {num_samples} training samples...")
    
    generator = LegalDataGenerator()
    
    # Generate diverse queries
    training_data = []
    for i in range(num_samples):
        query_data = generator.generate_query()
        
        training_example = {
            'query': query_data['query'],
            'facts': query_data['expected_facts']
        }
        training_data.append(training_example)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")
    
    logger.info(f"Generated {len(training_data)} training examples")
    return training_data


def prepare_data(use_generated: bool = True, num_generated: int = 1000) -> tuple:
    """Prepare training, validation, and test data."""
    logger.info("Preparing training data...")
    
    # Start with sample data
    sample_training_data = prepare_training_data_from_samples(SAMPLE_QUERIES)
    
    all_data = sample_training_data
    
    if use_generated:
        # Add generated data
        generated_data = generate_training_data(num_generated)
        all_data.extend(generated_data)
    
    # Create splits
    train_data, eval_data, test_data = create_data_splits(
        all_data, 
        train_ratio=0.8, 
        eval_ratio=0.1
    )
    
    logger.info(f"Data prepared: {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test")
    
    return train_data, eval_data, test_data


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train SLM for legal fact extraction")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to training configuration JSON"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/slm_legal_facts",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--use-wandb", 
        action="store_true",
        help="Use Weights & Biases for experiment tracking"
    )
    parser.add_argument(
        "--no-generated-data", 
        action="store_true",
        help="Only use sample data, no synthetic generation"
    )
    
    args = parser.parse_args()
    
    # Create training configuration
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = TrainingConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_wandb=args.use_wandb
        )
        logger.info("Using default configuration with command line overrides")
    
    print("🚀 HybEx-Law SLM Training Pipeline")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print()
    
    try:
        # Prepare data
        train_data, eval_data, test_data = prepare_data(
            use_generated=not args.no_generated_data,
            num_generated=args.num_samples
        )
        
        print(f"📊 Data Summary:")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Validation samples: {len(eval_data)}")
        print(f"  Test samples: {len(test_data)}")
        print()
        
        # Save test data for later evaluation
        test_data_path = Path(config.output_dir) / "test_data.json"
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        logger.info(f"Test data saved to: {test_data_path}")
        
        # Initialize trainer
        print("🔧 Initializing trainer...")
        trainer = SLMTrainer(config)
        
        # Start training
        print("🎯 Starting training...")
        trainer.train(train_data, eval_data)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {config.output_dir}")
        
        # Evaluate on test data
        print("\n📈 Evaluating on test data...")
        test_results = trainer.evaluate(test_data)
        print(f"Test Loss: {test_results.get('eval_loss', 'N/A')}")
        
        print("\n🎉 Training pipeline completed!")
        print("\nNext steps:")
        print("1. Test the trained model with SLM pipeline")
        print("2. Compare with baseline performance")
        print("3. Run comprehensive evaluation")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
