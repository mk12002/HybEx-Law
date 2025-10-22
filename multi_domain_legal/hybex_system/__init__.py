# hybex_system/__init__.py

from .main import HybExLawSystem
from .config import HybExConfig
from .master_scraper import MasterLegalScraper
from .prolog_engine import PrologEngine, LegalReasoning, PrologQuery # Expose PrologEngine components
from .data_processor import DataPreprocessor
from .neural_models import (
    DomainClassifier, EligibilityPredictor, 
    EnhancedLegalBERT, EnhancedLegalBERTTrainer,
    ModelTrainer, ModelMetrics, LegalDataset
) # Expose neural components
from .trainer import TrainingOrchestrator, AdvancedTrainingStrategy, DataAugmenter
from .evaluator import ModelEvaluator, EvaluationResults # Expose evaluator components

# Define what happens when 'from hybex_system import *' is used
__all__ = [
    "HybExLawSystem",
    "HybExConfig",
    "MasterLegalScraper",
    "PrologEngine",
    "LegalReasoning",
    "PrologQuery",
    "DataPreprocessor",
    "DomainClassifier",
    "EligibilityPredictor",
    "EnhancedLegalBERT",
    "EnhancedLegalBERTTrainer",
    "ModelTrainer",
    "ModelMetrics",
    "LegalDataset",
    "TrainingOrchestrator",
    "AdvancedTrainingStrategy",
    "DataAugmenter",
    "ModelEvaluator",
    "EvaluationResults",
    "create_system" # Keep the factory function
]

def create_system(config_path: str = None) -> HybExLawSystem:
    """
    Factory function to create and return an initialized HybExLawSystem instance.
    This is useful for programmatic access to the system.
    """
    return HybExLawSystem(config_path=config_path)