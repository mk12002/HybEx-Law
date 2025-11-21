# hybex_system/__init__.py

import warnings
warnings.filterwarnings('ignore', category=ImportWarning)

# Core config is always available
from .config import HybExConfig

# Try to import training/evaluation components (optional for streamlit app)
try:
    from .main import HybExLawSystem
    from .master_scraper import MasterLegalScraper
    from .prolog_engine import PrologEngine, LegalReasoning, PrologQuery
    from .data_processor import DataPreprocessor
    from .neural_models import (
        DomainClassifier, EligibilityPredictor, 
        EnhancedLegalBERT, EnhancedLegalBERTTrainer,
        ModelTrainer, ModelMetrics, LegalDataset
    )
    from .trainer import TrainingOrchestrator, AdvancedTrainingStrategy, DataAugmenter
    from .evaluator import ModelEvaluator, EvaluationResults
    
    _TRAINING_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    # Training components not available (missing utils or other dependencies)
    # This is fine for streamlit app which only needs predictor/translator
    import sys
    if '--verbose' in sys.argv:
        print(f"⚠️  Training components not available: {e}")
    _TRAINING_AVAILABLE = False
    HybExLawSystem = None
    MasterLegalScraper = None
    PrologEngine = None
    LegalReasoning = None
    PrologQuery = None
    DataPreprocessor = None
    DomainClassifier = None
    EligibilityPredictor = None
    EnhancedLegalBERT = None
    EnhancedLegalBERTTrainer = None
    ModelTrainer = None
    ModelMetrics = None
    LegalDataset = None
    TrainingOrchestrator = None
    AdvancedTrainingStrategy = None
    DataAugmenter = None
    ModelEvaluator = None
    EvaluationResults = None

# Define what happens when 'from hybex_system import *' is used
__all__ = [
    "HybExConfig",
    "HybExLawSystem",
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
    "create_system"
]

def create_system(config_path: str = None):
    """
    Factory function to create and return an initialized HybExLawSystem instance.
    This is useful for programmatic access to the system.
    """
    if not _TRAINING_AVAILABLE:
        raise ImportError(
            "Training components are not available. "
            "This is expected for streamlit app deployment. "
            "For training, ensure all dependencies including 'utils' module are present."
        )
    return HybExLawSystem(config_path=config_path)