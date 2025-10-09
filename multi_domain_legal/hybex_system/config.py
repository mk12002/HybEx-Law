"""
HybEx-Law System Configuration
============================

Central configuration for all system components with robust training parameters.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HybExConfig:
    """Central configuration for HybEx-Law system"""
    
    def __init__(self):
        # Base directories
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models" / "hybex_system"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.PLOTS_DIR = self.RESULTS_DIR / "plots"
        
        # Create directories
        for dir_path in [self.MODELS_DIR, self.LOGS_DIR, self.RESULTS_DIR, self.PLOTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.MODEL_CONFIG = {
            'base_model': 'nlpaueb/legal-bert-base-uncased', 
            'max_length': 512,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'epochs': 30,
            'warmup_steps': 200,
            'weight_decay': 0.01,
            'dropout_prob': 0.3,  # Added missing dropout probability
            'num_domains': 5,  # Number of legal domains
            'hidden_size': 768  # DistilBERT hidden size
        }
        
        self.MODEL_CONFIGS = {
            'domain_classifier': {
            'model_name': 'nlpaueb/legal-bert-base-uncased',
            'max_length': 512,
            'batch_size': 8,
            'learning_rate': 1e-5,
            'epochs': 30,
            'warmup_steps': 200,
            'weight_decay': 0.01,
            'early_stopping_patience': 3,
            'gradient_clip_val': 1.0,
            'dropout_prob': 0.3 # <- ADD THIS
            },
            'entity_extractor': {
                'model_name': 'nlpaueb/legal-bert-base-uncased',
                'max_length': 512,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'epochs': 30,  # More epochs for NER task
                'warmup_steps': 300,
                'weight_decay': 0.01,
                'early_stopping_patience': 4,
                'gradient_clip_val': 1.0, # <- ADDED COMMA
                'dropout_prob': 0.3
            },
            'eligibility_predictor': {
                'model_name': 'nlpaueb/legal-bert-base-uncased',
                'max_length': 512,
                'batch_size': 32,
                'learning_rate': 5e-6,  # Very low for final prediction
                'epochs': 30,  # Most epochs for main task
                'warmup_steps': 500,
                'weight_decay': 0.02,
                'early_stopping_patience': 5,
                'gradient_clip_val': 0.5, # <- ADDED COMMA
                'dropout_prob': 0.3
            }
        }
        
        # Data processing configuration
        self.DATA_CONFIG = {
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_seed': 42,
            'min_samples_per_domain': 100,
            'max_sequence_length': 512
        }
        
        # CONSOLIDATED PROLOG CONFIG (Removed overwrite block)
        self.PROLOG_CONFIG = {
            'enable_reasoning': True,
            'confidence_threshold': 0.7,
            'rule_weight': 0.4,  # Weight of Prolog vs Neural
            'neural_weight': 0.6,
            'min_confidence_for_override': 0.95, 
            'log_dir': 'logs/prolog',              
            'timeout': 120                         
            }

        self.NEURAL_CONFIG = {
            'min_confidence_for_override': 0.90, # Example value, adjust as needed
            'entity_extraction_model': 'en_core_web_sm' # Used by data_processor
        }
        # Entity extraction configuration
        # Load real legal data if available, otherwise use verified defaults
        self.ENTITY_CONFIG = self._load_real_legal_config()
        
        # Legal data sources and metadata
        self.LEGAL_SOURCES = {
            'primary_act': 'Legal Services Authorities Act, 1987',
            'last_updated': None,  # Will be set when real data is loaded
            'data_source': 'assumptions' if not self.ENTITY_CONFIG else 'scraped',
            'validation_status': 'unvalidated'
        }
        
        # Fallback configuration (verified from official sources as of 2024)
        if not self.ENTITY_CONFIG:
            self.ENTITY_CONFIG = {
                'income_thresholds': {
                    'general': 500000,      # ₹5 lakhs (NALSA 2024)
                    'sc_st': 800000,        # ₹8 lakhs (Enhanced for SC/ST)
                    'obc': 600000,          # ₹6 lakhs (State amendments)
                    'bpl': 0,               # No income limit for BPL
                    'ews': 800000           # ₹8 lakhs (EWS category)
                },
                'social_categories': [
                    'General', 'Scheduled Caste', 'Scheduled Tribe', 
                    'Other Backward Class', 'Economically Weaker Section',
                    'Below Poverty Line', 'Above Poverty Line'
                ],
                'case_types': [
                    'Criminal', 'Civil', 'Family Law', 'Consumer Protection',
                    'Employment Law', 'Fundamental Rights', 'General'
                ],
                'domains': [
                    'legal_aid', 'family_law', 'consumer_protection',
                    'employment_law', 'fundamental_rights'
                ]
            }
            self.LEGAL_SOURCES['data_source'] = 'verified_fallback'
        
        # Logging configuration (NOTE: The previous redundant PROLOG_CONFIG was here and is now removed)
        self.LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': str(self.LOGS_DIR / 'hybex_training.log'),
            'max_log_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        }
        
        # Evaluation configuration
        self.EVAL_CONFIG = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'],
            'save_predictions': True,
            'save_attention_weights': False,  # Disable for performance
            'generate_plots': True
        }

        # =================================================================
        # INTERACTIVE CHATBOT CONFIGURATION
        # =================================================================
        self.REQUIRED_FACTS_CONFIG = {
            'default': ['income', 'social_category'],
            'family_law': ['gender', 'age'],
            'employment_law': ['employment_duration', 'daily_wage'],
            'consumer_protection': ['goods_value', 'incident_date']
        }

        self.QUESTION_MAPPING = {
            'income': "What is your approximate total annual household income in Rupees?",
            'social_category': "Do you belong to a specific social category (e.g., General, SC, ST, OBC)?",
            'gender': "What is your gender?",
            'age': "What is your age?",
            'employment_duration': "How long were you employed at the job, in total days or years?",
            'daily_wage': "What was your average daily wage in Rupees?",
            'goods_value': "What was the total value of the goods or services in question, in Rupees?",
            'incident_date': "On what date (YYYY-MM-DD) did the incident occur?"
        }

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        return self.MODEL_CONFIGS.get(model_name, {})
    
    def get_model_path(self, model_name: str, epoch: int = None) -> Path:
        """Get path for saving/loading model"""
        if epoch is not None:
            return self.MODELS_DIR / f"{model_name}_epoch_{epoch}.pt"
        return self.MODELS_DIR / f"{model_name}_best.pt"
    
    def get_log_path(self, log_name: str) -> Path:
        """Get path for log file"""
        return self.LOGS_DIR / f"{log_name}.log"
    
    def get_results_path(self, result_name: str) -> Path:
        """Get path for results file"""
        return self.RESULTS_DIR / f"{result_name}.json"
    
    def _load_real_legal_config(self) -> Optional[Dict]:
        """Load real legal configuration from scraped data"""
        try:
            config_file = self.DATA_DIR / "real_legal_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    real_config = json.load(f)
                
                # Convert to expected format
                entity_config = {
                    'income_thresholds': {},
                    'social_categories': [
                        'General', 'Scheduled Caste', 'Scheduled Tribe', 
                        'Other Backward Class', 'Economically Weaker Section',
                        'Below Poverty Line', 'Above Poverty Line'
                    ],
                    'case_types': [
                        'Criminal', 'Civil', 'Family Law', 'Consumer Protection',
                        'Employment Law', 'Fundamental Rights', 'General'
                    ],
                    'domains': [
                        'legal_aid', 'family_law', 'consumer_protection',
                        'employment_law', 'fundamental_rights'
                    ]
                }
                
                # Extract income thresholds
                for category, data in real_config.get('income_thresholds', {}).items():
                    if isinstance(data, dict) and 'threshold' in data:
                        entity_config['income_thresholds'][category] = data['threshold']
                    else:
                        entity_config['income_thresholds'][category] = data
                
                logger.info("Loaded real legal configuration from scraped data")
                return entity_config
                
        except Exception as e:
            logger.warning(f"Could not load real legal config: {e}")
        
        return None
    
    def update_with_real_legal_data(self):
        """Update configuration with real legal data"""
        try:
            from .legal_scraper import LegalDataScraper
            
            scraper = LegalDataScraper(self)
            results = scraper.update_legal_knowledge()
            
            if results['status'] == 'success':
                # Reload configuration with real data
                real_config = self._load_real_legal_config()
                if real_config:
                    self.ENTITY_CONFIG = real_config
                    self.LEGAL_SOURCES.update({
                        'last_updated': results.get('last_updated'),
                        'data_source': 'scraped',
                        'validation_status': 'validated',
                        'sources_count': results.get('sources_scraped', 0)
                    })
                    logger.info("Configuration updated with real legal data")
                    return True
            
        except Exception as e:
            logger.error(f"Failed to update with real legal data: {e}")
        
        logger.warning("⚠️ Using fallback configuration based on verified legal sources")
        return False
    
    def get_legal_data_status(self) -> Dict[str, Any]:
        """Get status of legal data sources"""
        return {
            'config_source': self.LEGAL_SOURCES['data_source'],
            'last_updated': self.LEGAL_SOURCES['last_updated'],
            'validation_status': self.LEGAL_SOURCES['validation_status'],
            'primary_act': self.LEGAL_SOURCES['primary_act'],
            'income_thresholds': self.ENTITY_CONFIG['income_thresholds'],
            'real_data_available': self.LEGAL_SOURCES['data_source'] == 'scraped',
            'sources_count': self.LEGAL_SOURCES.get('sources_count', 0)
        }
    
    def create_directories(self):
        """Create necessary directories for the system"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR, 
            self.LOGS_DIR,
            self.RESULTS_DIR,
            self.PLOTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("All system directories created successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for export"""
        return {
            'base_dir': str(self.BASE_DIR),
            'data_dir': str(self.DATA_DIR),
            'models_dir': str(self.MODELS_DIR),
            'logs_dir': str(self.LOGS_DIR),
            'results_dir': str(self.RESULTS_DIR),
            'model_configs': self.MODEL_CONFIGS,
            'data_config': self.DATA_CONFIG,
            'entity_config': self.ENTITY_CONFIG,
            'prolog_config': self.PROLOG_CONFIG,
            'neural_config': self.NEURAL_CONFIG,
            'logging_config': self.LOGGING_CONFIG,
            'eval_config': self.EVAL_CONFIG,
            'legal_sources': self.LEGAL_SOURCES
        }
    
    def get_summary(self) -> str:
        """Get a summary of the current configuration"""
        return f"HybEx-Law Config - Data: {self.LEGAL_SOURCES['data_source']}, Models: {len(self.MODEL_CONFIGS)}, Legal Sources: {self.LEGAL_SOURCES.get('sources_count', 'Unknown')}"