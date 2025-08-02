# hybex_system/__init__.py

from .main import HybExLawSystem
from .config import HybExConfig
from .legal_scraper import LegalDataScraper
from .prolog_engine import PrologEngine, LegalReasoning, PrologQuery # Expose PrologEngine components
from .data_processor import DataPreprocessor
from .neural_models import DomainClassifier, EligibilityPredictor, ModelTrainer, ModelMetrics, LegalDataset # Expose neural components
from .trainer import TrainingOrchestrator
from .evaluator import ModelEvaluator, EvaluationResults # Expose evaluator components

# Define what happens when 'from hybex_system import *' is used
__all__ = [
    "HybExLawSystem",
    "HybExConfig",
    "LegalDataScraper",
    "PrologEngine",
    "LegalReasoning",
    "PrologQuery",
    "DataPreprocessor",
    "DomainClassifier",
    "EligibilityPredictor",
    "ModelTrainer",
    "ModelMetrics",
    "LegalDataset",
    "TrainingOrchestrator",
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