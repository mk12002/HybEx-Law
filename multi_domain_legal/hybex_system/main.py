# hybex_system/main.py

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
import numpy as np
import concurrent.futures
import time # Added for placeholders
import random # Added for placeholders

# Import all system modules
from .config import HybExConfig
from .trainer import TrainingOrchestrator
from .evaluator import ModelEvaluator
from .data_processor import DataPreprocessor
from .knowledge_graph_engine import KnowledgeGraphEngine # Assume this replaces PrologEngine
from .prolog_engine import PrologEngine # Keep PrologEngine for 'status' check detail compatibility
from .neural_models import DomainClassifier, EligibilityPredictor
from .master_scraper import MasterLegalScraper # Fixed: correct class name
from transformers import AutoTokenizer
from dataclasses import asdict

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybExLawSystem:
    """Main HybEx-Law System Interface"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the HybEx-Law system"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Assuming HybExConfig can handle dictionary input or uses defaults if keys are missing
            self.config = HybExConfig(**config_dict)
        else:
            self.config = HybExConfig()

        self.config.create_directories()
        
        # Initialize internal component references
        self._trainer: Optional[TrainingOrchestrator] = None
        self._evaluator: Optional[ModelEvaluator] = None
        self._data_processor: Optional[DataPreprocessor] = None
        self._knowledge_graph_engine: Optional[KnowledgeGraphEngine] = None 
        self._prolog_engine: Optional[PrologEngine] = None # Added for compatibility with status check
        self._master_scraper: Optional['MasterLegalScraper'] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._domain_classifier: Optional[DomainClassifier] = None
        self._eligibility_predictor: Optional[EligibilityPredictor] = None

        logger.info("HybEx-Law System Initialized with Knowledge Graph Engine")
        logger.info(f"Configuration: {self.config.get_summary()}")

    @property
    def trainer(self) -> TrainingOrchestrator:
        if self._trainer is None:
            self._trainer = TrainingOrchestrator(self.config)
        return self._trainer

    @property
    def evaluator(self) -> ModelEvaluator:
        if self._evaluator is None:
            self._evaluator = ModelEvaluator(self.config)
        return self._evaluator

    @property
    def data_processor(self) -> DataPreprocessor:
        if self._data_processor is None:
            self._data_processor = DataPreprocessor(self.config)
        return self._data_processor
    
    # Assuming the symbolic component is now primarily the Knowledge Graph Engine, 
    # but keeping Prolog for compatibility reasons in the overall hybrid system architecture.
    @property
    def knowledge_graph_engine(self) -> KnowledgeGraphEngine:
        if self._knowledge_graph_engine is None:
            self._knowledge_graph_engine = KnowledgeGraphEngine(self.config)
        return self._knowledge_graph_engine

    @property
    def prolog_engine(self) -> PrologEngine:
        # Assuming Prolog is used for status check/backward compatibility if KG is primary
        if self._prolog_engine is None:
             self._prolog_engine = PrologEngine(self.config)
        return self._prolog_engine

    @property
    def master_scraper(self) -> 'MasterLegalScraper':
        if self._master_scraper is None:
             self._master_scraper = MasterLegalScraper(
                 data_dir=self.config.DATA_DIR,
                 yaml_path=self.config.BASE_DIR / "knowledge_base" / "legal_rules.yaml"
             )
        return self._master_scraper

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy-load tokenizer for neural models"""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_CONFIG['base_model'])
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                raise RuntimeError("Failed to initialize tokenizer.")
        return self._tokenizer
        
    def _load_neural_model(self, model_class, model_name: str):
        """Helper to load a neural model's state dictionary."""
        model_path = self.config.MODELS_DIR / model_name / "model.pt"
        if not model_path.exists():
            logger.warning(f"Model file not found for {model_name} at {model_path}. Cannot load.")
            return None
        
        try:
            model = model_class(self.config)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            logger.info(f"Loaded {model_name} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            return None
            
    @property
    def domain_classifier(self) -> Optional[DomainClassifier]:
        if self._domain_classifier is None:
            self._domain_classifier = self._load_neural_model(DomainClassifier, 'domain_classifier')
        return self._domain_classifier

    @property
    def eligibility_predictor(self) -> Optional[EligibilityPredictor]:
        if self._eligibility_predictor is None:
            self._eligibility_predictor = self._load_neural_model(EligibilityPredictor, 'eligibility_predictor')
        return self._eligibility_predictor

    def _run_neural_predictions(self, query: str) -> Dict[str, Any]:
        """Run neural predictions for domain classification and eligibility."""
        logger.debug(f"Running neural predictions for query: {query[:50]}...")
        
        if not self.domain_classifier or not self.eligibility_predictor:
            logger.error("One or more neural models are missing/failed to load.")
            return {
                'domains': ['N/A'], 
                'eligibility_probability': 0.5, 
                'confidence': 0.3,
                'status': 'error_no_model'
            }
        
        # NOTE: This implementation is simplified. A production model would use 
        # a dedicated dataloader and device management.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            encoding = self.tokenizer(
                query,
                truncation=True,
                padding='max_length',
                max_length=self.config.MODEL_CONFIG['max_length'],
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Domain Classifier Prediction
            domain_logits = self.domain_classifier(input_ids, attention_mask)['logits'].cpu()
            domain_probs = torch.sigmoid(domain_logits).numpy().squeeze()
            predicted_domains = [
                self.config.ENTITY_CONFIG['domains'][i] 
                for i, prob in enumerate(domain_probs) if prob > 0.5
            ]
            
            # Eligibility Predictor Prediction
            eligibility_logits = self.eligibility_predictor(input_ids, attention_mask)['logits'].cpu()
            eligibility_prob = torch.sigmoid(eligibility_logits).item()
            
            return {
                'domains': predicted_domains if predicted_domains else ['Other/General'],
                'eligibility_probability': eligibility_prob,
                'confidence': np.mean(domain_probs) * eligibility_prob, # Simple confidence fusion
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Neural prediction failed: {e}", exc_info=True)
            return {
                'domains': ['N/A'], 
                'eligibility_probability': 0.5, 
                'confidence': 0.3,
                'status': f'error: {str(e)}'
            }

    def _run_graph_analysis(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Run Graph analysis for legal aid eligibility."""
        try:
            # We use the KnowledgeGraphEngine (which might wrap the PrologEngine)
            analysis_result = self.knowledge_graph_engine.predict_eligibility(entities)
            # analysis_result should be Dict with 'eligible', 'confidence', 'primary_reason', 'method'
            return analysis_result
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}", exc_info=True)
            # Fallback structure matching _fuse_predictions expectations
            return {
                'eligible': False,
                'confidence': 0.3,
                'primary_reason': f"Graph Analysis failed: {e}",
                'method': 'graph_error_fallback'
            }

    def predict_legal_eligibility(self, query: str, case_details: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Predict legal aid eligibility using a parallelized hybrid neural + graph-based approach.
        """
        logger.info("Predicting legal eligibility using PARALLEL hybrid neural-graph approach")
        logger.info(f"Query: {query[:150]}...")
        
        try:
            # 1. Entity Extraction
            if case_details:
                extracted_entities = case_details
                logger.info("Using provided case details for analysis.")
            else:
                extracted_entities = self.data_processor.extract_entities(query)
                logger.info(f"Extracted {len(extracted_entities)} entities from query.")

            # 2. Parallel Prediction
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_neural = executor.submit(self._run_neural_predictions, query)
                future_graph = executor.submit(self._run_graph_analysis, extracted_entities)
                
                # Fetch results
                neural_prediction = future_neural.result()
                graph_analysis = future_graph.result()

            # 3. Fusion
            final_decision = self._fuse_predictions(neural_prediction, graph_analysis)

            # 4. Final Result Compilation
            prediction_result = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'extracted_entities': extracted_entities,
                'neural_prediction': neural_prediction,
                # Renamed for consistency with the analysis logic
                'graph_reasoning': graph_analysis, 
                'final_decision': final_decision,
                'system_type': 'hybrid_neural_graph_parallel'
            }
            return prediction_result
        except Exception as e:
            logger.error(f"Hybrid prediction failed: {e}", exc_info=True)
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'error': f'Hybrid prediction failed: {e}',
                'system_type': 'hybrid_neural_graph_parallel'
            }

    def _fuse_predictions(self, neural_prediction: Dict[str, Any], graph_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse neural and graph-based predictions into a final decision."""
        # ... (Method logic is fine, kept as is) ...
        try:
            # IMPORTANT: The provided logic uses 'eligibility_probability' from neural and 'eligible' from graph.
            # We assume a fixed confidence threshold (0.5) for the neural model.
            neural_eligible = neural_prediction.get('eligibility_probability', 0.5) > 0.5
            neural_confidence = neural_prediction.get('confidence', 0.5)

            graph_eligible = graph_analysis.get('eligible', False)
            graph_confidence = graph_analysis.get('confidence', 0.5)

            # Prioritize the graph-based prediction if its confidence is high
            if graph_confidence > self.config.FUSION_CONFIG.get('graph_override_threshold', 0.8):
                final_eligible = graph_eligible
                final_confidence = graph_confidence
                explanation = f"Graph-based reasoning preferred due to high confidence: {graph_analysis.get('primary_reason')}"
                fusion_method = "graph_override"
            # If both models agree, combine their confidence
            elif neural_eligible == graph_eligible:
                final_eligible = neural_eligible
                final_confidence = (neural_confidence + graph_confidence) / 2
                explanation = "Both neural and graph-based systems agree."
                fusion_method = "agreement"
            # If they disagree, fallback to the more confident model
            else:
                if graph_confidence > neural_confidence:
                    final_eligible = graph_eligible
                    final_confidence = graph_confidence
                    explanation = "Disagreement between systems; defaulting to Graph-based prediction due to higher confidence."
                    fusion_method = "disagreement_graph_preferred"
                else:
                    final_eligible = neural_eligible
                    final_confidence = neural_confidence
                    explanation = "Disagreement between systems; defaulting to Neural prediction due to higher confidence."
                    fusion_method = "disagreement_neural_preferred"

            return {
                'eligible': final_eligible,
                'confidence': final_confidence,
                'explanation': explanation,
                'fusion_method': fusion_method,
            }

        except Exception as e:
            logger.error(f"Prediction fusion failed: {e}", exc_info=True)
            return {
                'eligible': False,
                'confidence': 0.2,
                'explanation': f"Fusion failed: {e}",
                'fusion_method': 'error_fallback'
            }

    # --- Command Handlers ---

    def update_legal_knowledge(self):
        """Handler for the update_knowledge command (alias for comprehensive scraping)."""
        logger.info("Executing comprehensive_scraping via update_knowledge alias...")
        return self.comprehensive_scraping()

    def comprehensive_scraping(self, priority_only: bool = False, report_only: bool = False) -> Dict[str, Any]:
        """Handler for the scrape command."""
        logger.info(f"Running comprehensive scraping (Priority Only: {priority_only}, Report Only: {report_only})")
        
        # Use the correct method name for scraping
        try:
            if priority_only:
                # For priority-only scraping, use the priority websites method
                results = self.master_scraper.scrape_priority_websites()
                total_items = sum(len(content) for content in results.values())
                return {
                    'status': 'success', 
                    'summary': {
                        'websites_scraped': len(results),
                        'total_content_extracted': total_items,
                        'status': 'completed',
                        'last_scraping_date': datetime.now().isoformat()
                    }
                }
            else:
                # For comprehensive scraping
                results = self.master_scraper.run_comprehensive_scraping()
                total_items = sum(len(content) for content in results.values())
                return {
                    'status': 'success', 
                    'summary': {
                        'websites_scraped': len(results),
                        'total_content_extracted': total_items,
                        'status': 'completed',
                        'last_scraping_date': datetime.now().isoformat()
                    }
                }
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def train_complete_system(self, data_dir: str) -> Dict[str, Any]:
        """Handler for the train command."""
        logger.info(f"Starting complete system training with data from: {data_dir}")
        
        try:
            results = self.trainer.run_complete_training_pipeline(data_dir)
            return results
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {'status': 'error', 'error': str(e), 'results_file': 'N/A'}

    def evaluate_system(self, test_data: Optional[str] = None, model_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Handler for the evaluate command."""
        logger.info(f"Starting system evaluation. Test Data: {test_data}")
        
        # ASSUMPTION: Evaluator loads models using model_paths if provided, or uses config defaults.
        try:
            # If test_data is a path, load it
            test_samples = []
            if test_data and Path(test_data).exists():
                with open(test_data, 'r') as f:
                    test_samples = json.load(f)
                logger.info(f"Loaded {len(test_samples)} samples for evaluation.")
            
            # Since trainer.py saves the final_models with paths, we need a way to pass them 
            # to the evaluator, but in CLI mode, we often rely on the default saved paths.
            # The evaluator is assumed to handle model loading based on the default MODELS_DIR
            # and the structure set by `train_complete_system`.
            
            # We call the main evaluation method
            results = self.evaluator.evaluate_end_to_end_system(
                 test_samples=test_samples, 
                 models_paths=model_paths # Pass explicit paths if provided
            )
            return results
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def preprocess_data(self, data_dir: str) -> Dict[str, Any]:
        """Handler for the preprocess command."""
        logger.info(f"Starting data preprocessing for data in: {data_dir}")
        try:
            results = self.data_processor.run_preprocessing_pipeline(data_dir)
            return results
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def run_interactive_session(self):
        """Handler for the chat command."""
        print("\n" + "="*50)
        print(" HybEx-Law Interactive Eligibility Predictor")
        print("="*50)
        print("Enter 'quit' or 'exit' to end the session.")
        
        while True:
            query = input("\nEnter your legal query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            if not query:
                continue
                
            try:
                # Run the prediction pipeline
                result = self.predict_legal_eligibility(query)
                print_prediction_result(result)
            except Exception as e:
                print(f"An error occurred during prediction: {e}")
                logger.error(f"Interactive prediction error: {e}")
        
        print("\nInteractive session ended. Goodbye!")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health check"""
        # ... (Status logic is fine, corrected for PrologEngine name consistency) ...
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
                'rules_loaded_from_kb': False,
                'rule_summary': {}
            },
            'pytorch': {
                'available': False,
                'cuda_available': False,
                'version': 'Unknown'
            },
            'master_scraper': {
                'available': True,
                'db_path': str(self.config.DATA_DIR / "legal_knowledge.db"),
                'db_exists': (self.config.DATA_DIR / "legal_knowledge.db").exists(),
                'priority_websites': 7, 
                'last_scraping': 'N/A',
                'integration_script': str(Path(__file__).parent.parent / "legal_integration.py")
            }
        }
        
        # Check for trained models
        if self.config.MODELS_DIR.exists():
            for model_name in ['domain_classifier', 'eligibility_predictor', 'gnn_model']: # Added gnn_model
                model_path = self.config.MODELS_DIR / model_name / "model.pt"
                status['trained_models'][model_name] = model_path.exists()
        
        # Check Prolog availability
        try:
            prolog_engine = self.prolog_engine # Access property to trigger lazy loading
            status['prolog_engine']['available'] = prolog_engine.prolog_available
            # ASSUMPTION: PrologEngine has these attributes
            status['prolog_engine']['rules_loaded_from_kb'] = prolog_engine.rules_loaded
            status['prolog_engine']['rule_summary'] = prolog_engine.get_comprehensive_rule_summary()
        except Exception as e:
            status['prolog_engine']['error'] = str(e)
        
        # Check PyTorch
        try:
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
        
        # Cleanup Prolog engine (corrected property name)
        if self._prolog_engine:
             if hasattr(self._prolog_engine, 'cleanup') and callable(self._prolog_engine.cleanup):
                 self._prolog_engine.cleanup()
        
        # Cleanup KG engine
        if self._knowledge_graph_engine:
            if hasattr(self._knowledge_graph_engine, 'cleanup') and callable(self._knowledge_graph_engine.cleanup):
                self._knowledge_graph_engine.cleanup()
        
        # Cleanup trainer
        if self._trainer:
            if hasattr(self._trainer, 'cleanup') and callable(self._trainer.cleanup):
                self._trainer.cleanup()
            
        logger.info("System cleanup completed")

def print_prediction_result(result: Dict[str, Any]):
    """Helper function to print the prediction result in a structured way."""
    if 'error' in result:
        print(f"\n[ERROR] An error occurred during prediction: {result['error']}")
        return
        
    print("\n" + "="*60)
    print("      HybEx-Law Final Analysis Report")
    print("="*60)
    print(f"\nInitial Query: {result.get('query')}")
    
    print("\n--- Gathered Case Facts ---")
    extracted_entities = result.get('extracted_entities', {})
    if extracted_entities:
        for k, v in extracted_entities.items():
            print(f"  • {k.replace('_', ' ').title()}: {v}")
    else:
        print("  • No specific facts were extracted.")
    
    neural_res = result.get('neural_prediction', {})
    print("\n--- Neural Prediction (Domain & Eligibility) ---")
    print(f"  • Predicted Domains: {', '.join(neural_res.get('domains', ['N/A']))}")
    print(f"  • Eligibility Probability: {neural_res.get('eligibility_probability', 0.0):.2f}")
    
    # Corrected key from 'prolog_reasoning' to 'graph_reasoning' for consistency
    graph_res = result.get('graph_reasoning', {})
    print("\n--- Graph-based Reasoning (KGNN/Prolog) ---")
    print(f"  • Eligible: {'Yes' if graph_res.get('eligible') else 'No'}")
    print(f"  • Confidence: {graph_res.get('confidence', 0.0):.2f}")
    print(f"  • Primary Reason: {graph_res.get('primary_reason', graph_res.get('reasoning', 'N/A'))}")

    final_dec = result.get('final_decision', {})
    print("\n--- Final Hybrid Decision ---")
    print(f"  • Final Eligibility: {'Yes' if final_dec.get('eligible') else 'No'} **")
    print(f"  • Final Confidence: {final_dec.get('confidence', 0.0):.2f}")
    print(f"  • Explanation: {final_dec.get('explanation', 'N/A')}")
    print("="*60)
    print(f"** Final decision determined by fusion method: {final_dec.get('fusion_method', 'N/A')}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="HybEx-Law: Hybrid Neural-Symbolic Legal AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 # Update legal knowledge from government sources
 python -m hybex_system.main scrape                    # Run comprehensive legal knowledge scraping
 python -m hybex_system.main scrape --priority-only   # Scrape only highest priority sites
 python -m hybex_system.main update_knowledge         # Same as 'scrape' command

 # Train complete system
 python -m hybex_system.main train --data-dir data/

 # Evaluate trained system
 python -m hybex_system.main evaluate

 # Check system status
 python -m hybex_system.main status

 # Predict legal eligibility for a query
 python -m hybex_system.main predict --query "I am from SC category with 3 lakh annual income and need help for a family dispute."

 # Start interactive session
 python -m hybex_system.main chat
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Update Knowledge command (New)
    update_parser = subparsers.add_parser('update_knowledge', help='Update legal knowledge from external sources')
    update_parser.add_argument('--config', type=str, help='Path to configuration file')

    # Comprehensive scraping command (master scraper)
    scrape_parser = subparsers.add_parser('scrape', help='Run comprehensive legal knowledge scraping')
    scrape_parser.add_argument('--config', type=str, help='Path to configuration file')
    scrape_parser.add_argument('--priority-only', action='store_true', 
                                 help='Only scrape highest priority websites (IndiaCode, NALSA)')
    scrape_parser.add_argument('--report-only', action='store_true',
                                 help='Generate report from existing scraped data without new scraping')

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
    
    # Interactive Chat command
    chat_parser = subparsers.add_parser('chat', help='Start an interactive session to determine eligibility')
    chat_parser.add_argument('--config', type=str, help='Path to configuration file')

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
        system = HybExLawSystem(args.config if hasattr(args, 'config') and args.config else None)
        
        # Execute command
        if args.command == 'update_knowledge': 
            logger.info("Starting update_knowledge command")
            results = system.update_legal_knowledge()
            print("\nLegal knowledge update sequence completed.")
            print(f"Status: {results.get('status')}")
            
        elif args.command == 'scrape': 
            logger.info("Starting comprehensive scraping command")
            priority_only = getattr(args, 'priority_only', False)
            report_only = getattr(args, 'report_only', False)

            if report_only:
                # Direct logic for report generation, corrected to use MasterScraper
                print("\nGenerating Report from Existing Scraped Data...")
                results = system.master_scraper.generate_report()
                
                print("="*40)
                print(f"Total items in database: {results.get('total_count', 0)}")
                print("By domain:")
                for domain, count in results.get('domain_counts', {}).items():
                    print(f"  {domain}: {count} items")
                print("="*40)

            else:
                # Run actual scraping
                results = system.comprehensive_scraping(priority_only=priority_only)
                if results.get('status') == 'success':
                    print("\nComprehensive legal knowledge scraping completed successfully!")
                    summary = results['summary']
                    print(f"Websites scraped: {summary.get('websites_scraped', 'N/A')}")
                    print(f"Total content extracted: {summary.get('total_content_extracted', 'N/A')}")
                    print(f"Status: {summary.get('status', 'completed')}")
                    print(f"Completion time: {summary.get('last_scraping_date', 'N/A')}")
                else:
                    print(f"\nScraping failed: {results.get('error', 'Unknown error')}")
            
        elif args.command == 'train':
            logger.info("Starting training command")
            results = system.train_complete_system(args.data_dir)
            
            # Check the result status before printing success
            if results.get('status') == 'success' or results.get('status') == 'completed':
                print("\nTraining completed successfully!")
                print(f"Results saved to: {results.get('results_file', 'N/A')}")
            else:
                print("\nTraining failed.")
                print(f"Error: {results.get('error', 'An unknown error occurred during training.')}")
                if results.get('results_file'):
                    print(f"Error details saved to: {results.get('results_file')}")
            
        elif args.command == 'evaluate':
            logger.info("Starting evaluation command")
            explicit_model_paths = None
            if getattr(args, 'model_dir', None):
                 # Logic to map model_dir to individual model paths is complex for CLI.
                 # Evaluation should primarily rely on models saved in the default MODELS_DIR.
                 pass # Let the `evaluate_system` method handle default loading

            results = system.evaluate_system(
                test_data=getattr(args, 'test_data', None),
                model_paths=explicit_model_paths
            )
            print("\nEvaluation completed successfully!")
            # Display results summary (e.g., F1, Accuracy) here
            
        elif args.command == 'preprocess':
            logger.info("Starting preprocessing command")
            results = system.preprocess_data(args.data_dir)
            print("\nData preprocessing completed successfully!")
            print(f"Processed {results.get('total_processed_samples', 0)} samples")
            
        elif args.command == 'status':
            logger.info("Starting status check")
            status = system.get_system_status()
            
            print("\n" + "="*50)
            print("      HybEx-Law System Status")
            print("="*50)
            print(f"Timestamp: {status['timestamp']}")
            print(f"PyTorch Available: {status['pytorch']['available']}")
            print(f"CUDA Available: {status['pytorch']['cuda_available']}")
            print(f"Prolog Available: {status['prolog_engine']['available']}")
            print(f"Prolog Rules from KB: {status['prolog_engine']['rules_loaded_from_kb']}")
            print("\nTrained Models:")
            for model_name, exists in status['trained_models'].items():
                print(f"  • {model_name}: {'OK' if exists else 'MISSING'}")
            print("\nDirectories:")
            for dir_name, exists in status['directory_exists'].items():
                print(f"  • {dir_name}: {'OK' if exists else 'MISSING'}")
            print("\nProlog Rule Summary:")
            for cat, count in status['prolog_engine']['rule_summary'].get('rule_counts', {}).items():
                print(f"  • {cat}: {count} rules")
            print(f"  • Total Prolog Rules: {status['prolog_engine']['rule_summary'].get('total_rules', 'N/A')}")
            print("\nMaster Scraper Status:")
            print(f"  • Available: {status['master_scraper']['available']}")
            print(f"  • DB Exists: {status['master_scraper']['db_exists']}")
            print(f"  • Priority Websites: {status['master_scraper']['priority_websites']}")
            print(f"  • Integration Script: Available")
            
        elif args.command == 'predict':
            logger.info("Starting prediction command")
            result = system.predict_legal_eligibility(args.query)
            # Use the single, corrected helper function
            print_prediction_result(result) 
            
        elif args.command == 'chat':
            logger.info("Starting interactive session")
            system.run_interactive_session()
            
        # Cleanup is called outside the block but inside the main try/except
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        if system: system.cleanup()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        if system: 
            try:
                system.cleanup()
            except Exception as cleanup_e:
                logger.error(f"Cleanup failed: {cleanup_e}")
        sys.exit(1)

    finally:
        # Final cleanup attempt
        if system:
            system.cleanup()


if __name__ == "__main__":
    main()