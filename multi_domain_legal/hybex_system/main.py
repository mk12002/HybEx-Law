# hybex_system/main.py

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch # Add this (even if commented out, it suggests future use)
import numpy as np # Add this

# Import HybEx-Law components (ensure these imports point to the correct files)
from .config import HybExConfig
from .trainer import TrainingOrchestrator
from .evaluator import ModelEvaluator
from .data_processor import DataPreprocessor
# from .neu            print(f"  Eligible: {'Yes' if final_dec.get('eligible') else 'No'}")al_models import ModelTrainer # Not needed here, ModelTrainer is part of TrainingOrchestrator
from .prolog_engine import PrologEngine # Corrected import for the updated engine
from .legal_scraper import LegalDataScraper # To integrate scraper into main CLI flow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybExLawSystem:
    """Main HybEx-Law System Interface"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the HybEx-Law system"""
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = HybExConfig(**config_dict)
        else:
            self.config = HybExConfig()
        
        # Create necessary directories
        self.config.create_directories()
        
        # Initialize components (lazy loading, but ensure scraper is potentially run early)
        self._trainer = None
        self._evaluator = None
        self._data_processor = None
        self._prolog_engine = None
        self._legal_scraper = None # Add scraper
        
        logger.info("HybEx-Law System Initialized")
        logger.info(f"Configuration: {self.config.get_summary()}")
    
    @property
    def trainer(self) -> TrainingOrchestrator:
        """Lazy-load training orchestrator"""
        if self._trainer is None:
            self._trainer = TrainingOrchestrator(self.config)
        return self._trainer
    
    @property
    def evaluator(self) -> ModelEvaluator:
        """Lazy-load model evaluator"""
        if self._evaluator is None:
            self._evaluator = ModelEvaluator(self.config)
        return self._evaluator
    
    @property
    def data_processor(self) -> DataPreprocessor:
        """Lazy-load data processor"""
        if self._data_processor is None:
            self._data_processor = DataPreprocessor(self.config)
        return self._data_processor
    
    @property
    def prolog_engine(self) -> PrologEngine:
        """Lazy-load Prolog engine"""
        if self._prolog_engine is None:
            self._prolog_engine = PrologEngine(self.config)
        return self._prolog_engine

    @property
    def legal_scraper(self) -> LegalDataScraper: # Add scraper property
        """Lazy-load legal data scraper"""
        if self._legal_scraper is None:
            self._legal_scraper = LegalDataScraper(self.config)
        return self._legal_scraper
    
    def update_legal_knowledge(self) -> Dict[str, Any]: # New method for scraping
        """Update legal knowledge from external sources using the scraper."""
        logger.info("Starting legal knowledge update...")
        try:
            results = self.legal_scraper.update_legal_knowledge()
            logger.info(f"Legal knowledge update completed: {results.get('status')}")
            # After scraping, re-initialize PrologEngine to ensure it picks up latest config (thresholds)
            self._prolog_engine = PrologEngine(self.config) # Re-initialize to load new rules/thresholds
            logger.info("PrologEngine re-initialized with potentially updated legal knowledge.")
            return results
        except Exception as e:
            logger.error(f"Legal knowledge update failed: {e}")
            raise

    def train_complete_system(self, data_directory: str, **kwargs) -> Dict[str, Any]:
        """Train the complete HybEx-Law system"""
        logger.info("Starting complete system training")
        logger.info(f"Data directory: {data_directory}")
        
        data_path = Path(data_directory)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON data files found in {data_directory}")
        
        logger.info(f"Found {len(json_files)} data files")
        
        try:
            # Pass the config to the trainer, which internally handles data processor and prolog engine
            results = self.trainer.run_complete_training_pipeline(data_directory)
            
            logger.info("Complete system training finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_system(self, test_data: Optional[str] = None, model_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Evaluate the trained HybEx-Law system"""
        logger.info("Starting system evaluation")
        
        # Load test data logic remains the same
        if test_data:
            test_data_path = Path(test_data)
        else:
            # Use processed_data from results_dir
            test_data_path = self.config.RESULTS_DIR / "processed_data" / "test_samples.json"
        
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}. Please run 'preprocess' or 'train' first.")
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        
        logger.info(f"Loaded {len(test_samples)} test samples for evaluation")
        
        # The ModelEvaluator is expected to handle loading and evaluation.
        # It needs the config, which contains model paths.
        try:
            # The evaluator will load models from config.MODELS_DIR
            # If model_paths are provided, the evaluator needs to support overriding default paths.
            # Assuming ModelEvaluator.evaluate_end_to_end_system can load default models or take explicit paths.
            
            evaluation_results = self.evaluator.evaluate_end_to_end_system(
                models=model_paths, # Pass explicit paths if provided, else evaluator uses defaults
                test_samples=test_samples
            )
            
            results_file = self.evaluator.save_evaluation_results(evaluation_results)
            report = self.evaluator.generate_evaluation_report(evaluation_results)
            self.evaluator.create_evaluation_visualizations(
                evaluation_results, 
                str(self.config.RESULTS_DIR / "evaluation_plots")
            )
            
            logger.info("System evaluation completed successfully")
            logger.info(f"Results saved to: {results_file}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def preprocess_data(self, data_directory: str) -> Dict[str, Any]:
        """Preprocess legal data for training"""
        logger.info("Starting data preprocessing")
        
        try:
            results = self.data_processor.run_preprocessing_pipeline(data_directory)
            
            logger.info("Data preprocessing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def predict_legal_eligibility(self, query: str, case_details: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict legal aid eligibility for a single query using the hybrid system."""
        logger.info("Predicting legal eligibility for a single query (hybrid approach)")
        logger.info(f"Query: {query[:150]}...") # Log a bit more of the query
        
        try:
            # Step 1: Extract entities using DataPreprocessor
            # Assuming DataPreprocessor has an extract_entities method
            # This method should be available in a trained/loaded DataProcessor
            extracted_entities = self.data_processor.extract_entities(query)
            logger.info(f"Extracted Entities: {extracted_entities}")

            # Step 2: Use PrologEngine for comprehensive legal analysis
            # The PrologEngine should be initialized and ready.
            # It will use the comprehensive rules loaded during its init.
            analysis_result = self.prolog_engine.comprehensive_legal_analysis(extracted_entities)

            # Step 3: Integrate Neural Models (Optional, depending on hybrid strategy)
            # For a true hybrid approach, you'd also run neural predictions here.
            # This would require loading the trained neural models.
            # For now, we rely heavily on the Prolog engine for the 'final' decision in this predict function.
            # If a neural prediction is *also* needed for fusion, it would be done by the evaluator's predict method.
            
            # Example (if you wanted to run a neural predictor here too):
            # domain_model = self.evaluator.load_trained_model(str(self.config.MODELS_DIR / "domain_classifier"), "domain_classifier")
            # eligibility_model = self.evaluator.load_trained_model(str(self.config.MODELS_DIR / "eligibility_predictor"), "eligibility_predictor")
            #
            # tokenizer = self.evaluator.tokenizer # Assuming evaluator has tokenizer
            # encoding = tokenizer(query, truncation=True, padding='max_length', max_length=self.config.MODEL_CONFIG['max_length'], return_tensors='pt')
            #
            # # Run neural predictions (simplified)
            # with torch.no_grad():
            #     domain_logits = domain_model(encoding['input_ids'].to(self.evaluator.device), encoding['attention_mask'].to(self.evaluator.device))['logits']
            #     neural_domains = [self.config.ENTITY_CONFIG['domains'][i] for i, val in enumerate((torch.sigmoid(domain_logits) > 0.5).squeeze().cpu().numpy()) if val]
            #     
            #     eligibility_logits = eligibility_model(encoding['input_ids'].to(self.evaluator.device), encoding['attention_mask'].to(self.evaluator.device))['logits']
            #     neural_eligibility_prob = torch.sigmoid(eligibility_logits).item()
            #
            # neural_prediction_info = {
            #     'domains': neural_domains,
            #     'eligibility_probability': neural_eligibility_prob,
            #     'confidence': neural_eligibility_prob if neural_eligibility_prob > 0.5 else (1 - neural_eligibility_prob)
            # }

            prediction_result = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'extracted_entities': extracted_entities,
                # 'neural_prediction': neural_prediction_info, # Uncomment if you implement the neural part
                'prolog_reasoning': {
                    'eligible': analysis_result.eligible,
                    'reasoning': analysis_result.primary_reason,
                    'confidence': analysis_result.confidence,
                    'applied_rules': analysis_result.applicable_rules,
                    'detailed_reasoning': analysis_result.detailed_reasoning,
                    'legal_citations': analysis_result.legal_citations,
                    'method': analysis_result.method
                },
                'final_decision': {
                    'eligible': analysis_result.eligible,
                    'confidence': analysis_result.confidence,
                    'explanation': analysis_result.primary_reason # For now, direct from Prolog
                },
            }
            
            logger.info("Single query prediction completed.")
            return prediction_result

        except Exception as e:
            logger.error(f"Single query prediction failed: {e}", exc_info=True)
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'error': f'Prediction failed: {e}'
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health check"""
        logger.info("Checking system status")
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.get_summary(),
            'directories': {
                'data_dir': str(self.config.DATA_DIR),
                'models_dir': str(self.config.MODELS_DIR),
                'results_dir': str(self.config.RESULTS_DIR),
                'logs_dir': str(self.config.LOGS_DIR)
            },
            'directory_exists': {
                'data_dir': self.config.DATA_DIR.exists(),
                'models_dir': self.config.MODELS_DIR.exists(),
                'results_dir': self.config.RESULTS_DIR.exists(),
                'logs_dir': self.config.LOGS_DIR.exists()
            },
            'trained_models': {},
            'prolog_engine': {
                'available': False,
                'version': 'Unknown',
                'rules_loaded_from_kb': False, # New status
                'rule_summary': {} # New status
            },
            'pytorch': {
                'available': False,
                'cuda_available': False,
                'version': 'Unknown'
            },
            'legal_scraper': { # New status for scraper
                'db_path': str(self.legal_scraper.db_path),
                'db_exists': self.legal_scraper.db_path.exists(),
                'last_update_status': self.config.get_legal_data_status().get('last_scraper_update', 'N/A')
            }
        }
        
        # Check for trained models
        if self.config.MODELS_DIR.exists():
            for model_name in ['domain_classifier', 'eligibility_predictor']: # Removed entity_extractor for now, as it's not directly saved as model.pt in neural_models.py
                model_path = self.config.MODELS_DIR / model_name / "model.pt"
                status['trained_models'][model_name] = model_path.exists()
        
        # Check Prolog availability
        try:
            prolog_engine = self.prolog_engine # Access property to trigger lazy loading
            status['prolog_engine']['available'] = prolog_engine.prolog_available
            status['prolog_engine']['rules_loaded_from_kb'] = prolog_engine.rules_loaded # Access the new flag
            status['prolog_engine']['rule_summary'] = prolog_engine.get_comprehensive_rule_summary()
        except Exception as e:
            status['prolog_engine']['error'] = str(e)
        
        # Check PyTorch
        try:
            import torch
            status['pytorch']['available'] = True
            status['pytorch']['cuda_available'] = torch.cuda.is_available()
            status['pytorch']['version'] = torch.__version__
        except ImportError:
            status['pytorch']['error'] = 'PyTorch not available'
        
        logger.info("System status check completed")
        return status
    
    def export_config(self, output_path: str) -> str:
        """Export current configuration to file"""
        config_path = Path(output_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Configuration exported to {config_path}")
        return str(config_path)
    
    def cleanup(self):
        """Clean up system resources"""
        logger.info("Cleaning up system resources")
        
        # Cleanup Prolog engine
        if self._prolog_engine:
            self._prolog_engine.cleanup()
        
        # Cleanup trainer (if it has a cleanup method)
        if self._trainer:
            # Check if trainer has a cleanup method
            if hasattr(self._trainer, 'cleanup') and callable(self._trainer.cleanup):
                self._trainer.cleanup()
            else:
                logger.warning("Trainer does not have a cleanup method.")
        
        logger.info("System cleanup completed")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="HybEx-Law: Hybrid Neural-Symbolic Legal AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update legal knowledge from government sources
  python -m hybex_system.main update_knowledge

  # Train complete system
  python -m hybex_system.main train --data-dir data/

  # Evaluate trained system
  python -m hybex_system.main evaluate

  # Check system status
  python -m hybex_system.main status

  # Preprocess data only
  python -m hybex_system.main preprocess --data-dir data/

  # Predict legal eligibility for a query
  python -m hybex_system.main predict --query "I am from SC category with 3 lakh annual income and need help for a family dispute."
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Update Knowledge command (New)
    update_parser = subparsers.add_parser('update_knowledge', help='Update legal knowledge from external sources')
    update_parser.add_argument('--config', type=str, help='Path to configuration file')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the complete HybEx-Law system')
    train_parser.add_argument('--data-dir', type=str, required=True,
                            help='Directory containing training data JSON files')
    train_parser.add_argument('--config', type=str,
                            help='Path to configuration file')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--test-data', type=str,
                           help='Path to test data file')
    eval_parser.add_argument('--model-dir', type=str,
                           help='Directory containing trained models (if not using defaults)')
    eval_parser.add_argument('--config', type=str,
                           help='Path to configuration file')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data only')
    preprocess_parser.add_argument('--data-dir', type=str, required=True,
                                 help='Directory containing raw data JSON files')
    preprocess_parser.add_argument('--config', type=str,
                                 help='Path to configuration file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    status_parser.add_argument('--config', type=str,
                             help='Path to configuration file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict legal eligibility for a single query')
    predict_parser.add_argument('--query', type=str, required=True,
                              help='Legal query text')
    predict_parser.add_argument('--config', type=str,
                              help='Path to configuration file')
    
    return parser

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    system = None # Initialize system outside try-except for cleanup
    try:
        # Initialize system
        system = HybExLawSystem(args.config if hasattr(args, 'config') else None)
        
        # Execute command
        if args.command == 'update_knowledge': # New command handler
            logger.info("Starting update_knowledge command")
            results = system.update_legal_knowledge()
            print("\nLegal knowledge updated successfully!")
            print(f"Status: {results.get('status')}")
            
        elif args.command == 'train':
            logger.info("Starting training command")
            results = system.train_complete_system(args.data_dir)
            print("\nTraining completed successfully!")
            print(f"Results saved to: {results.get('results_file', 'N/A')}")
            
        elif args.command == 'evaluate':
            logger.info("Starting evaluation command")
            # For model_paths, if --model-dir is provided, it needs to map to individual model paths
            # Currently, evaluator.evaluate_end_to_end_system expects a Dict[str, str] for model_paths
            # For simplicity, if model_dir is given, assume it implies default structured paths.
            # A more robust solution might require parsing subdirectories for each model type.
            explicit_model_paths = None
            if getattr(args, 'model_dir', None):
                model_base_path = Path(args.model_dir)
                if model_base_path.exists():
                    explicit_model_paths = {
                        'domain_classifier': str(model_base_path / "domain_classifier"),
                        'eligibility_predictor': str(model_base_path / "eligibility_predictor"),
                        # Add other models if their paths are structured similarly
                    }
                    logger.info(f"Using provided model directory: {args.model_dir}")
                else:
                    logger.warning(f"Provided model directory not found: {args.model_dir}. Using default model paths.")

            results = system.evaluate_system(
                test_data=getattr(args, 'test_data', None),
                model_paths=explicit_model_paths
            )
            print("\nEvaluation completed successfully!")
            
        elif args.command == 'preprocess':
            logger.info("Starting preprocessing command")
            results = system.preprocess_data(args.data_dir)
            print("\nData preprocessing completed successfully!")
            print(f"Processed {results.get('total_processed_samples', 0)} samples")
            
        elif args.command == 'status':
            logger.info("Starting status check")
            status = system.get_system_status()
            
            print("\nHybEx-Law System Status")
            print("="*50)
            print(f"Timestamp: {status['timestamp']}")
            print(f"PyTorch Available: {status['pytorch']['available']}")
            print(f"CUDA Available: {status['pytorch']['cuda_available']}")
            print(f"Prolog Available: {status['prolog_engine']['available']}")
            print(f"Prolog Rules from KB: {status['prolog_engine']['rules_loaded_from_kb']}") # New
            print("\nTrained Models:")
            for model_name, exists in status['trained_models'].items():
                print(f"  • {model_name}: {'OK' if exists else 'MISSING'}")
            print("\nDirectories:")
            for dir_name, exists in status['directory_exists'].items():
                print(f"  • {dir_name}: {'OK' if exists else 'MISSING'}")
            print("\nProlog Rule Summary:") # New
            for cat, count in status['prolog_engine']['rule_summary'].get('rule_counts', {}).items():
                print(f"  • {cat}: {count} rules")
            print(f"  • Total Prolog Rules: {status['prolog_engine']['rule_summary'].get('total_rules', 'N/A')}")
            print("\nLegal Scraper Status:") # New
            print(f"  • DB Exists: {status['legal_scraper']['db_exists']}")
            print(f"  • Last Update Status: {status['legal_scraper']['last_update_status']}")
            
        elif args.command == 'predict':
            logger.info("Starting prediction command")
            result = system.predict_legal_eligibility(args.query)
            
            print("\nLegal Eligibility Prediction")
            print("="*50)
            print(f"Query: {args.query}")
            print("\nExtracted Entities:")
            for k, v in result.get('extracted_entities', {}).items():
                print(f"  {k}: {v}")
            print("\nProlog Reasoning:")
            prolog_res = result.get('prolog_reasoning', {})
            print(f"  Eligible: {'Yes' if prolog_res.get('eligible') else 'No'}")
            print(f"  Confidence: {prolog_res.get('confidence', 0.0):.2f}")
            print(f"  Primary Reason: {prolog_res.get('reasoning', 'N/A')}")
            if prolog_res.get('detailed_reasoning'):
                print("  Detailed Reasoning:")
                for dr in prolog_res['detailed_reasoning']:
                    print(f"    - {dr.get('content', 'N/A')} (Confidence: {dr.get('confidence', 0.0):.2f})")
            print(f"  Applied Rules: {', '.join(prolog_res.get('applied_rules', ['N/A']))}")
            print(f"  Legal Citations: {', '.join(prolog_res.get('legal_citations', ['N/A']))}")
            print(f"  Method: {prolog_res.get('method', 'N/A')}")

            # If you add neural predictions to main.py's predict_legal_eligibility, display them here too
            # neural_res = result.get('neural_prediction', {})
            # if neural_res:
            #     print("\nNeural Prediction:")
            #     print(f"  Domains: {', '.join(neural_res.get('domains', ['N/A']))}")
            #     print(f"  Eligibility Probability: {neural_res.get('eligibility_probability', 0.0):.2f}")

            print("\nFinal Decision:")
            final_dec = result.get('final_decision', {})
            print(f"  Eligible: {'Yes' if final_dec.get('eligible') else 'No'}")
            print(f"  Confidence: {final_dec.get('confidence', 0.0):.2f}")
            print(f"  Explanation: {final_dec.get('explanation', 'N/A')}")
            
        # Cleanup
        if system:
            system.cleanup()
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        if system: system.cleanup()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        if system: system.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()