"""
Hybrid Legal NLP Pipeline - Main orchestrator for the two-stage processing system.

This module implements the core architecture:
1. Stage 1: Coarse-grained entity classification
2. Stage 2: High-precision fact extraction
3. Integration with Prolog reasoning engine
"""

from typing import List, Dict, Any
import logging
from pathlib import Path

from .stage1_classifier import EntityPresenceClassifier
from ..extractors.income_extractor import IncomeExtractor
from ..extractors.case_type_classifier import CaseTypeClassifier
from ..extractors.social_category_extractor import SocialCategoryExtractor
from ..utils.text_preprocessor import TextPreprocessor


class HybridLegalNLPPipeline:
    """
    Main pipeline that orchestrates the two-stage fact extraction process.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the hybrid pipeline with all components.
        
        Args:
            model_path: Optional path to pre-trained models
        """
        self.logger = logging.getLogger(__name__)
        
        # Text preprocessing
        self.preprocessor = TextPreprocessor()
        
        # Stage 1: Entity presence classifier
        self.entity_classifier = EntityPresenceClassifier(model_path)
        
        # Stage 2: Specialized extractors
        self.income_extractor = IncomeExtractor()
        self.case_type_classifier = CaseTypeClassifier(model_path)
        self.social_category_extractor = SocialCategoryExtractor()
        
        # Mapping of entities to extractors
        self.extractor_map = {
            'income': self.income_extractor,
            'case_type': self.case_type_classifier,
            'social_category': self.social_category_extractor
        }
        
    def process_query(self, query: str, verbose: bool = False) -> List[str]:
        """
        Process a natural language query and extract legal facts.
        
        Args:
            query: Natural language description of legal situation
            verbose: Whether to show detailed processing steps
            
        Returns:
            List of Prolog facts as strings
        """
        if verbose:
            print(f"📝 Original query: {query}")
        
        # Preprocess the text
        processed_text = self.preprocessor.preprocess(query)
        if verbose:
            print(f"🔧 Preprocessed: {processed_text}")
        
        # Stage 1: Determine which entities are present
        present_entities = self.entity_classifier.predict(processed_text)
        if verbose:
            print(f"🔍 Stage 1 - Detected entities: {present_entities}")
        
        # Stage 2: Extract facts using appropriate extractors
        extracted_facts = []
        
        for entity in present_entities:
            if entity in self.extractor_map:
                extractor = self.extractor_map[entity]
                facts = extractor.extract(processed_text, query)
                extracted_facts.extend(facts)
                
                if verbose:
                    print(f"⚙️  Stage 2 - {entity} extractor found: {facts}")
        
        # Always add applicant fact
        if not any('applicant(' in fact for fact in extracted_facts):
            extracted_facts.insert(0, 'applicant(user).')
        
        # Clean and validate facts
        validated_facts = self._validate_facts(extracted_facts)
        
        if verbose:
            print(f"✅ Final validated facts: {validated_facts}")
        
        return validated_facts
    
    def _validate_facts(self, facts: List[str]) -> List[str]:
        """
        Validate and clean extracted facts.
        
        Args:
            facts: Raw extracted facts
            
        Returns:
            Cleaned and validated facts
        """
        validated = []
        seen_facts = set()
        
        for fact in facts:
            # Clean the fact
            cleaned_fact = fact.strip()
            if not cleaned_fact.endswith('.'):
                cleaned_fact += '.'
            
            # Remove duplicates
            if cleaned_fact not in seen_facts:
                validated.append(cleaned_fact)
                seen_facts.add(cleaned_fact)
        
        return validated
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline component information
        """
        return {
            'stage1_classifier': type(self.entity_classifier).__name__,
            'extractors': {
                entity: type(extractor).__name__ 
                for entity, extractor in self.extractor_map.items()
            },
            'preprocessor': type(self.preprocessor).__name__
        }
