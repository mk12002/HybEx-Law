"""
FIXED Entity Extraction Script for HybEx-Law
FORCES text-based extraction, OVERWRITING synthetic entities
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEntityExtractor:
    """Extracts ALL entities from actual query text"""
    
    def __init__(self):
        """Initialize spaCy model"""
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            import spacy
            import spacy.cli
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_entities(self, query: str) -> dict:
        """Extract entities from query text"""
        entities = {}
        query_lower = query.lower()
        
        # ===== INCOME EXTRACTION =====
        income_patterns = [
            (r'rs\.?\s*(\d+)\s*(?:monthly|per\s+month)', 'monthly'),
            (r'(\d+)\s*rupees?\s*(?:monthly|per\s+month)', 'monthly'),
            (r'earning\s+(?:rs\.?\s*)?(\d+)', 'monthly'),
            (r'earning\s+monthly\s+(\d+)', 'monthly'),
            (r'salary\s+(?:is|was)?\s*(?:rs\.?\s*)?(\d+)', 'monthly'),
            (r'income\s+(?:is|of)?\s*(?:rs\.?\s*)?(\d+)', 'monthly'),
        ]
        
        for pattern, _ in income_patterns:
            match = re.search(pattern, query_lower)
            if match:
                income_value = int(match.group(1))
                if 500 <= income_value <= 500000:
                    entities['income'] = income_value
                    if income_value <= 12000:
                        entities['income_category'] = 'very_low'
                    elif income_value <= 25000:
                        entities['income_category'] = 'low'
                    elif income_value <= 50000:
                        entities['income_category'] = 'medium'
                    else:
                        entities['income_category'] = 'high'
                    break
        
        # ===== AGE EXTRACTION =====
        age_match = re.search(r'(\d{2})\s+year(?:s)?\s+old', query_lower)
        if not age_match:
            age_match = re.search(r'(?:of|aged)\s+(\d{2})\s+years?', query_lower)
        if not age_match:
            age_match = re.search(r'(?:laborer|worker|person)\s+of\s+(\d{2})\s+years?', query_lower)
        
        if age_match:
            age = int(age_match.group(1))
            if 18 <= age <= 100:
                entities['age'] = age
        
        # ===== GENDER EXTRACTION (Priority-based) =====
        # Priority 1: Direct self-identification
        if re.search(r'\bI am (?:a )?woman\b', query_lower):
            entities['gender'] = 'female'
        elif re.search(r'\bI am (?:a )?man\b', query_lower):
            entities['gender'] = 'male'
        # Priority 2: Relationship words indicating speaker's gender
        elif re.search(r'\b(widow|mother|wife|daughter)\b', query_lower):
            entities['gender'] = 'female'
        elif re.search(r'\bmy husband\b|\bhusband\s+(?:is|was|drinks)', query_lower):
            # If talking ABOUT husband → speaker is likely female
            entities['gender'] = 'female'
        
        # ===== SOCIAL CATEGORY (PERFECT FIX) =====
        # Check ST FIRST to avoid conflict with "Scheduled" matching SC pattern
        if re.search(r'\bst\b|scheduled\s+tribe|\btribal\b|\badivasi\b', query_lower):
            entities['social_category'] = 'st'
        elif re.search(r'\bsc\b|scheduled\s+caste|\bdalit\b', query_lower):
            # \bsc\b matches "SC" in "SC community" ✅
            # Does NOT match "sc" in "Scheduled" because:
            # - "scheduled tribe" is caught by ST check above ✅
            # - "scheduled caste" is caught by "scheduled\s+caste" pattern ✅
            entities['social_category'] = 'sc'
        elif re.search(r'\bobc\b|backward\s+class', query_lower):
            entities['social_category'] = 'obc'
        elif re.search(r'\bbpl\b|below\s+poverty', query_lower):
            entities['social_category'] = 'bpl'
        else:
            entities['social_category'] = 'general'
        
        # ===== VULNERABLE CATEGORIES (only add if True) =====
        if re.search(r'disabled|handicapped|differently\s+abled|acid\s+attack', query_lower):
            entities['is_disabled'] = True
        
        if 'age' in entities and entities['age'] >= 65:
            entities['is_senior_citizen'] = True
        elif re.search(r'senior\s+citizen', query_lower):
            entities['is_senior_citizen'] = True
        
        if re.search(r'transgender', query_lower):
            entities['is_transgender'] = True
        
        if re.search(r'\bwidow\b', query_lower):
            entities['is_widow'] = True
        
        if re.search(r'single\s+(?:mother|father|parent)', query_lower):
            entities['is_single_parent'] = True
        
        return entities

def add_entities_to_split_files(data_dir: str = "data"):
    """FORCE re-extraction of entities, overwriting existing ones"""
    
    extractor = ComprehensiveEntityExtractor()
    data_path = Path(data_dir)
    
    split_files = ["train_split.json", "val_split.json", "test_split.json"]
    
    total_updated = 0
    
    for split_file in split_files:
        file_path = data_path / split_file
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {split_file}")
        logger.info(f"{'='*80}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"Loaded {len(samples)} samples")
        
        # ALWAYS extract, overwriting existing entities
        for sample in tqdm(samples, desc=f"Extracting entities for {split_file}"):
            sample['extracted_entities'] = extractor.extract_entities(sample['query'])
            total_updated += 1
        
        logger.info(f"✅ Updated ALL {len(samples)} samples with text-extracted entities")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved to {file_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ COMPLETE: Extracted entities for {total_updated} samples")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    add_entities_to_split_files()
