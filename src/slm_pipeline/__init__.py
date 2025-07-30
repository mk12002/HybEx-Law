"""
Small Language Model Pipeline for HybEx-Law
==========================================

This module implements the SLM-based fact extraction pipeline as an advanced
alternative to the TF-IDF + Logistic Regression baseline.

Key Features:
- Fine-tuned SLM for direct fact extraction
- Structured Prolog output generation
- Integration with existing reasoning engine
- Comprehensive evaluation framework
"""

from .slm_fact_extractor import SLMFactExtractor
from .slm_pipeline import SLMPipeline
from .slm_trainer import SLMTrainer
from .slm_evaluator import SLMEvaluator

__version__ = "1.0.0"
__author__ = "HybEx-Law Team"

__all__ = [
    "SLMFactExtractor",
    "SLMPipeline", 
    "SLMTrainer",
    "SLMEvaluator"
]
