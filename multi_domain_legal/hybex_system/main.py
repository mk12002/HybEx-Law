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
from .neural_models import DomainClassifier, EligibilityPredictor  # Add neural models
from transformers import AutoTokenizer
from dataclasses import asdict

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
        
        # Initialize components (lazy loading)
        self._trainer = None
        self._evaluator = None
        self._data_processor = None
        self._prolog_engine = None
        self._master_scraper = None # Only master scraper now
        
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
    def domain_classifier(self):
        """Lazy-load domain classifier neural model"""
        if not hasattr(self, '_domain_classifier'):
            try:
                import torch
                model_path = self.config.MODELS_DIR / "domain_classifier" / "model.pt"
                if model_path.exists():
                    self._domain_classifier = DomainClassifier(self.config)
                    self._domain_classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
                    self._domain_classifier.eval()
                    logger.info("Domain classifier loaded successfully")
                else:
                    logger.warning(f"Domain classifier model not found at {model_path}")
                    self._domain_classifier = None
            except Exception as e:
                logger.error(f"Failed to load domain classifier: {e}")
                self._domain_classifier = None
        return self._domain_classifier

    @property
    def eligibility_predictor(self):
        """Lazy-load eligibility predictor neural model"""
        if not hasattr(self, '_eligibility_predictor'):
            try:
                import torch
                model_path = self.config.MODELS_DIR / "eligibility_predictor" / "model.pt"
                if model_path.exists():
                    self._eligibility_predictor = EligibilityPredictor(self.config)
                    self._eligibility_predictor.load_state_dict(torch.load(model_path, map_location='cpu'))
                    self._eligibility_predictor.eval()
                    logger.info("Eligibility predictor loaded successfully")
                else:
                    logger.warning(f"Eligibility predictor model not found at {model_path}")
                    self._eligibility_predictor = None
            except Exception as e:
                logger.error(f"Failed to load eligibility predictor: {e}")
                self._eligibility_predictor = None
        return self._eligibility_predictor

    @property
    def tokenizer(self):
        """Lazy-load tokenizer for neural models"""
        if not hasattr(self, '_tokenizer'):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_CONFIG['base_model'])
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                self._tokenizer = None
        return self._tokenizer

    def _run_neural_predictions(self, query: str) -> Dict[str, Any]:
        """Run neural predictions for domain classification and eligibility."""
        try:
            # Check if neural models are available
            if self.domain_classifier is None or self.eligibility_predictor is None or self.tokenizer is None:
                logger.warning("Neural models not available, using fallback")
                return {
                    'domains': ['general_legal_aid'],
                    'eligibility_probability': 0.5,
                    'confidence': 0.5,
                    'method': 'fallback',
                    'available': False
                }
            
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(
                query, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=self.config.MODEL_CONFIG.get('max_length', 512)
            )
            
            with torch.no_grad():
                # Domain classification
                domain_outputs = self.domain_classifier(**inputs)
                domain_logits = domain_outputs['logits']
                domain_probs = torch.sigmoid(domain_logits).cpu().numpy()[0]
                predicted_domains = [
                    self.config.ENTITY_CONFIG['domains'][i]
                    for i in range(len(domain_probs))
                    if domain_probs[i] > 0.5
                ]
                domain_confidence = float(domain_probs.max()) if predicted_domains else 0.5
                
                # Eligibility prediction
                eligibility_outputs = self.eligibility_predictor(**inputs)
                eligibility_logits = eligibility_outputs['logits']
                eligibility_prob = torch.sigmoid(eligibility_logits).cpu().numpy().item()  # FIXED LINE: Use .item()
                eligibility_confidence = float(eligibility_prob)
                
                # Combine results
                confidence = (domain_confidence + eligibility_confidence) / 2
                return {
                    'domains': predicted_domains if predicted_domains else ['general_legal_aid'],
                    'eligibility_probability': eligibility_prob,
                    'confidence': confidence,
                    'method': 'neural',
                    'available': True
                }
                
        except Exception as e:
            logger.error(f"Neural prediction failed: {e}")
            return {
                'domains': ['general_legal_aid'],
                'eligibility_probability': 0.5,
                'confidence': 0.3,
                'method': 'error_fallback',
                'available': False,
                'error': str(e)
            }
    
    def _run_prolog_analysis(self, entities: Dict[str, Any], neural_prediction: Dict[str, Any]) -> Any:
        """Run Prolog analysis informed by neural predictions."""
        try:
            # Use neural-predicted domains to inform Prolog analysis
            predicted_domains = neural_prediction.get('domains', ['general_legal_aid'])
            
            # Run comprehensive legal analysis with domain hints
            analysis_result = self.prolog_engine.comprehensive_legal_analysis(
                entities, 
                domains=predicted_domains
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Prolog analysis failed: {e}")
            # Return a fallback result
            from .prolog_engine import LegalReasoning
            return LegalReasoning(
                case_id=f"fallback_{datetime.now().strftime('%H%M%S')}",
                eligible=False,
                confidence=0.3,
                primary_reason=f"Analysis failed: {e}",
                detailed_reasoning=[],
                applicable_rules=[],
                legal_citations=[],
                method='prolog_fallback'
            )
    
    def _fuse_predictions(self, neural_prediction: Dict[str, Any], prolog_analysis: Any) -> Dict[str, Any]:
        """Fuse neural and Prolog predictions into final decision."""
        try:
            neural_eligible = neural_prediction.get('eligibility_probability', 0.5) > 0.5
            neural_confidence = neural_prediction.get('confidence', 0.5)
            
            prolog_eligible = prolog_analysis.eligible
            prolog_confidence = prolog_analysis.confidence
            
            # Fusion strategies
            if neural_prediction.get('available', False):
                # Both systems agree
                if neural_eligible == prolog_eligible:
                    final_eligible = neural_eligible
                    final_confidence = (neural_confidence + prolog_confidence) / 2
                    explanation = f"Both neural and symbolic reasoning agree: {prolog_analysis.primary_reason}"
                    fusion_method = "agreement"
                # Disagreement - trust Prolog for rule-based cases, neural for complex patterns
                else:
                    if prolog_confidence > neural_confidence:
                        final_eligible = prolog_eligible
                        final_confidence = prolog_confidence * 0.9  # Slight penalty for disagreement
                        explanation = f"Symbolic reasoning preferred: {prolog_analysis.primary_reason}"
                        fusion_method = "prolog_preferred"
                    else:
                        final_eligible = neural_eligible
                        final_confidence = neural_confidence * 0.9
                        explanation = f"Neural prediction preferred (confidence: {neural_confidence:.2f})"
                        fusion_method = "neural_preferred"
            else:
                # Neural not available, use Prolog only
                final_eligible = prolog_eligible
                final_confidence = prolog_confidence
                explanation = prolog_analysis.primary_reason
                fusion_method = "prolog_only"
            
            return {
                'eligible': final_eligible,
                'confidence': final_confidence,
                'explanation': explanation,
                'fusion_method': fusion_method,
                'neural_agreement': neural_eligible == prolog_eligible if neural_prediction.get('available') else None
            }
            
        except Exception as e:
            logger.error(f"Prediction fusion failed: {e}")
            return {
                'eligible': False,
                'confidence': 0.2,
                'explanation': f"Fusion failed: {e}",
                'fusion_method': 'error_fallback'
            }

    @property
    def master_scraper(self): # Master scraper property
        """Lazy-load master legal scraper"""
        if self._master_scraper is None:
            try:
                # Add the scripts directory to Python path
                scripts_dir = Path(__file__).parent.parent / "scripts"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                
                from master_legal_scraper import MasterLegalScraper
                # Initialize with data directory from config
                self._master_scraper = MasterLegalScraper(data_dir=self.config.DATA_DIR)
                logger.info("Master legal scraper initialized successfully")
            except ImportError as e:
                logger.error(f"Master legal scraper not available: {e}")
                logger.error("Please ensure master_legal_scraper.py is in the scripts directory.")
                raise ImportError("Master legal scraper not available")
        return self._master_scraper
    
    def update_legal_knowledge(self) -> Dict[str, Any]: # Updated method using master scraper
        """Update legal knowledge from external sources using the master scraper."""
        logger.info("Starting legal knowledge update with master scraper...")
        try:
            # Use master scraper for comprehensive knowledge update
            results = self.comprehensive_scraping()
            logger.info(f"Legal knowledge update completed: {results.get('status')}")
            # After scraping, re-initialize PrologEngine to ensure it picks up latest scraped data
            self._prolog_engine = None  # Force reinitialization
            logger.info("PrologEngine will be re-initialized with updated legal knowledge.")
            return results
        except Exception as e:
            logger.error(f"Legal knowledge update failed: {e}")
            raise

    def comprehensive_scraping(self) -> Dict[str, Any]:
        """Run comprehensive legal knowledge scraping using the master scraper"""
        logger.info("Starting comprehensive legal knowledge scraping with master scraper")
        try:
            # Use the master scraper for comprehensive data extraction
            results = self.master_scraper.run_comprehensive_scraping()
            
            # Update the config with latest scraping info
            scraping_summary = {
                'last_scraping_date': datetime.now().isoformat(),
                'websites_scraped': len(results),
                'total_content_extracted': sum(len(content) for content in results.values()),
                'status': 'completed'
            }
            
            # Re-initialize PrologEngine to ensure it picks up latest scraped data
            self._prolog_engine = None  # Force reinitialization
            logger.info("PrologEngine will be re-initialized with updated knowledge base.")
            
            return {
                'master_scraper_results': results,
                'summary': scraping_summary,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Comprehensive scraping failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

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
                model_paths, # Pass explicit paths if provided, else evaluator uses defaults
                test_samples
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
        """
        Predict legal aid eligibility using the hybrid neural + symbolic approach.
        Can accept pre-gathered case_details to bypass initial entity extraction.
        """
        logger.info("Predicting legal eligibility using hybrid neural-symbolic approach")
        logger.info(f"Query: {query[:150]}...")
        
        try:
            # Step 1: Use pre-existing entities if provided, otherwise extract them from the query.
            if case_details:
                extracted_entities = case_details
                logger.info("Using pre-gathered entities for analysis.")
            else:
                extracted_entities = self.data_processor.extract_entities(query)
            
            logger.info(f"Entities for Analysis: {extracted_entities}")

            # Step 2: Neural Predictions (Domain Classification + Eligibility)
            neural_prediction = self._run_neural_predictions(query)
            logger.info(f"Neural Predictions: {neural_prediction}")

            # Step 3: Enhanced Prolog Analysis with Neural-informed domains and full facts
            prolog_analysis = self._run_prolog_analysis(extracted_entities, neural_prediction)
            logger.info(f"Prolog Analysis: {prolog_analysis.eligible} (confidence: {prolog_analysis.confidence:.2f})")

            # Step 4: Hybrid Fusion
            final_decision = self._fuse_predictions(neural_prediction, prolog_analysis)
            logger.info(f"Final Hybrid Decision: {final_decision}")

            # Step 5: Construct comprehensive result
            prediction_result = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'extracted_entities': extracted_entities,
                'neural_prediction': neural_prediction,
                'prolog_reasoning': asdict(prolog_analysis),
                'final_decision': final_decision,
                'system_type': 'hybrid_neural_symbolic'
            }
            
            logger.info("Hybrid prediction completed successfully.")
            return prediction_result

        except Exception as e:
            logger.error(f"Hybrid prediction failed: {e}", exc_info=True)
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'error': f'Hybrid prediction failed: {e}',
                'system_type': 'hybrid_neural_symbolic'
            }
        
    def _get_required_facts_for_domain(self, domain: str) -> list:
        """Determines the list of required facts for a given domain."""
        required = set(self.config.REQUIRED_FACTS_CONFIG.get('default', []))
        required.update(self.config.REQUIRED_FACTS_CONFIG.get(domain, []))
        return list(required)

    def run_interactive_session(self):
        """Starts an interactive conversational session for legal aid analysis."""
        print("\nWelcome to the HybEx-Law Interactive Assistant.")
        print("You can type 'quit' at any time to exit.")
        initial_query = input("System: Please describe your legal issue in a sentence or two.\nUser: ")

        if initial_query.lower().strip() == 'quit':
            return

        gathered_facts = {}
        
        # 1. Initial analysis to extract facts and determine domain
        initial_entities = self.data_processor.extract_entities(initial_query)
        gathered_facts.update(initial_entities)
        
        neural_pred = self._run_neural_predictions(initial_query)
        domain = neural_pred['domains'][0] if neural_pred.get('domains') else 'default'
        print(f"System: Based on your query, this seems to be a '{domain.replace('_', ' ')}' issue. To give you the most accurate analysis, I need to ask a few more questions.")

        # 2. Determine what information is missing
        required_facts = self._get_required_facts_for_domain(domain)
        missing_facts = [fact for fact in required_facts if fact not in gathered_facts]

        # 3. Conversational loop to gather missing facts
        while missing_facts:
            fact_to_find = missing_facts.pop(0)
            question = self.config.QUESTION_MAPPING.get(fact_to_find, f"Could you please provide information about your {fact_to_find.replace('_', ' ')}?")
            
            user_response = input(f"System: {question}\nUser: ")
            
            if user_response.lower().strip() == 'quit':
                print("System: Session ended by user.")
                return
            
            # Extract entities from the user's latest response and update our knowledge
            new_entities = self.data_processor.extract_entities(user_response)
            if new_entities:
                gathered_facts.update(new_entities)
                print(f"System: Understood. I've noted the following: {new_entities}")
            
            # Re-check what's still missing
            missing_facts = [fact for fact in required_facts if fact not in gathered_facts]

        # 4. Final Analysis
        print("\nSystem: Thank you. I have all the necessary information. Analyzing your case now...")
        final_result = self.predict_legal_eligibility(initial_query, case_details=gathered_facts)

        # 5. Display Final Result
        print_prediction_result(final_result)

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
            'master_scraper': { # Status for master scraper
                'available': True,
                'db_path': str(self.config.DATA_DIR / "legal_knowledge.db"),
                'db_exists': (self.config.DATA_DIR / "legal_knowledge.db").exists(),
                'priority_websites': 7,  # Number of priority websites
                'last_scraping': 'N/A',
                'integration_script': str(Path(__file__).parent.parent / "legal_integration.py")
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

def print_prediction_result(result: Dict[str, Any]):
    """Helper function to print the prediction result in a structured way."""
    if 'error' in result:
        print(f"\nAn error occurred: {result['error']}")
        return
        
    print("\n" + "="*60)
    print("      HybEx-Law Final Analysis Report")
    print("="*60)
    print(f"\nInitial Query: {result.get('query')}")
    
    print("\n--- Gathered Case Facts ---")
    if result.get('extracted_entities'):
        for k, v in result['extracted_entities'].items():
            print(f"  • {k.replace('_', ' ').title()}: {v}")
    else:
        print("  • No specific facts were extracted.")
    
    prolog_res = result.get('prolog_reasoning', {})
    print("\n--- Symbolic Reasoning (Prolog) ---")
    print(f"  • Eligible: {'Yes' if prolog_res.get('eligible') else 'No'}")
    print(f"  • Confidence: {prolog_res.get('confidence', 0.0):.2f}")
    print(f"  • Primary Reason: {prolog_res.get('primary_reason', 'N/A')}")

    final_dec = result.get('final_decision', {})
    print("\n--- Final Hybrid Decision ---")
    print(f"  • Final Eligibility: {'Yes' if final_dec.get('eligible') else 'No'}")
    print(f"  • Final Confidence: {final_dec.get('confidence', 0.0):.2f}")
    print(f"  • Explanation: {final_dec.get('explanation', 'N/A')}")
    print("="*60)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="HybEx-Law: Hybrid Neural-Symbolic Legal AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update legal knowledge from government sources
  python -m hybex_system.main scrape                    # Run comprehensive legal knowledge scraping
  python -m hybex_system.main scrape --priority-only   # Scrape only highest priority sites
  python -m hybex_system.main scrape --report-only     # Generate report from existing data
  python -m hybex_system.main update_knowledge         # Same as 'scrape' command (uses master scraper)

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
        system = HybExLawSystem(args.config if hasattr(args, 'config') else None)
        
        # Execute command
        if args.command == 'update_knowledge': # New command handler
            logger.info("Starting update_knowledge command")
            results = system.update_legal_knowledge()
            print("\nLegal knowledge updated successfully!")
            print(f"Status: {results.get('status')}")
            
        elif args.command == 'scrape': # Comprehensive scraping command
            logger.info("Starting comprehensive scraping command")
            if getattr(args, 'report_only', False):
                # Generate report from existing data
                try:
                    scraper = system.master_scraper
                    # Read existing database and generate report
                    import sqlite3
                    db_path = system.config.DATA_DIR / "legal_knowledge.db"
                    if db_path.exists():
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM scraped_content")
                        total_count = cursor.fetchone()[0]
                        cursor.execute("SELECT legal_domain, COUNT(*) FROM scraped_content GROUP BY legal_domain")
                        domain_counts = dict(cursor.fetchall())
                        conn.close()
                        
                        print("\nExisting Scraped Data Report")
                        print("="*40)
                        print(f"Total items in database: {total_count}")
                        print("By domain:")
                        for domain, count in domain_counts.items():
                            print(f"  {domain}: {count} items")
                    else:
                        print("\nNo existing scraped data found.")
                except Exception as e:
                    print(f"Error generating report: {e}")
            else:
                # Run actual scraping
                results = system.comprehensive_scraping()
                if results['status'] == 'success':
                    print("\nComprehensive legal knowledge scraping completed successfully!")
                    summary = results['summary']
                    print(f"Websites scraped: {summary['websites_scraped']}")
                    print(f"Total content extracted: {summary['total_content_extracted']}")
                    print(f"Status: {summary['status']}")
                    print(f"Completion time: {summary['last_scraping_date']}")
                else:
                    print(f"\nScraping failed: {results.get('error', 'Unknown error')}")
            
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
            print("\nMaster Scraper Status:") # Updated to master scraper
            print(f"  • Available: {status['master_scraper']['available']}")
            print(f"  • DB Exists: {status['master_scraper']['db_exists']}")
            print(f"  • Priority Websites: {status['master_scraper']['priority_websites']}")
            print(f"  • Integration Script: Available")
            
        elif args.command == 'predict':
            logger.info("Starting prediction command")
            result = system.predict_legal_eligibility(args.query)
            print_prediction_result(result)
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
            
        # elif args.command == 'status':
        #     # ... (status command logic remains the same)
            
        # elif args.command == 'predict':
        #     logger.info("Starting prediction command")
        #     result = system.predict_legal_eligibility(args.query)
        #     print_prediction_result(result) # Use the helper function
            
        elif args.command == 'chat':
            logger.info("Starting interactive session")
            system.run_interactive_session()
            
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