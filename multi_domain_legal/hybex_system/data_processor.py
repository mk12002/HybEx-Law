# hybex_system/data_processor.py

import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
import re
import spacy
import spacy.cli # Add this
from datetime import datetime
from tqdm import tqdm

from .config import HybExConfig

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses raw legal data into structured formats for training and evaluation."""
    
    def __init__(self, config: HybExConfig):
        self.config = config
        self.nlp = self._load_spacy_model() # Load spaCy model for NER
        self.setup_logging()
        logger.info("DataPreprocessor initialized.")
    
    def setup_logging(self):
        log_file = self.config.get_log_path('data_preprocessing')
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('data_preprocessing.log') for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(self.config.LOGGING_CONFIG['format'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to DataPreprocessor logger.")
        logger.info("="*50)
        logger.info("Starting HybEx-Law Data Preprocessing")
        logger.info("="*50)

    def _load_spacy_model(self):
        """Load a spaCy model for entity extraction."""
        try:
            # Ensure 'en_core_web_sm' is downloaded: python -m spacy download en_core_web_sm
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Downloading it...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}. Entity extraction might be limited.")
            return None

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts legal entities (income, social category, case type, location) from text.
        This is a rule-based/regex-based entity extraction, can be enhanced with NER models.
        """
        entities = {}
        text_lower = text.lower()
        
        # Using spaCy for initial NER to get general entities, then refine with regex
        doc = self.nlp(text) if self.nlp else None

        # 1. Income Extraction
        # Improved regex to handle various formats (e.g., "3 lakhs", "Rs. 5,00,000", "500000")
        income_patterns = [
            r'(\d+(?:[\\.,]\\d+)*)\\s*(?:lakhs?|lac(?:s)?|million)?\\s*(?:rupees|rs\\.?|k)?(?:\\s*per\\s*year|annual|monthly)?',
            r'rs\\.?\\s*(\d+(?:[\\.,]\\d+)*)',
            r'(\d+(?:[\\.,]\\d+)*)\\s*(?:k|thousand)?(?:\\s*per\\s*year|annual|monthly)?'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount_str = match.group(1).replace(',', '').replace('.', '') # Remove commas/dots initially
                unit = match.group(0).lower() # Get the full matched string to check for units
                
                try:
                    amount = float(amount_str)
                    if 'lakh' in unit or 'lac' in unit:
                        amount *= 100000
                    elif 'million' in unit:
                        amount *= 1000000
                    elif 'k' in unit or 'thousand' in unit: # Basic handling for 'k'
                        amount *= 1000
                    
                    entities['income'] = int(amount) # Store as integer
                    break # Stop after first successful match
                except ValueError:
                    logger.warning(f"Could not convert extracted income '{amount_str}' to number.")

        # 2. Social Category Extraction (prioritize specific terms)
        social_category_mapping = {
            'scheduled caste': 'Scheduled Caste', 'sc': 'Scheduled Caste', 'dalit': 'Scheduled Caste',
            'scheduled tribe': 'Scheduled Tribe', 'st': 'Scheduled Tribe', 'tribal': 'Scheduled Tribe',
            'below poverty line': 'Below Poverty Line', 'bpl': 'Below Poverty Line', 'poor': 'Below Poverty Line',
            'economically weaker section': 'Economically Weaker Section', 'ews': 'Economically Weaker Section',
            'other backward class': 'Other Backward Class', 'obc': 'Other Backward Class'
        }
        for term, category in social_category_mapping.items():
            if term in text_lower:
                entities['social_category'] = category
                break
        
        # 3. Case Type Extraction
        case_type_mapping = {
            'criminal': 'Criminal', 'crime': 'Criminal', 'murder': 'Criminal', 'theft': 'Criminal', 'assault': 'Criminal',
            'family': 'Family Law', 'divorce': 'Family Law', 'custody': 'Family Law', 'matrimonial': 'Family Law', 'marriage': 'Family Law',
            'consumer': 'Consumer Protection', 'product defect': 'Consumer Protection', 'warranty': 'Consumer Protection',
            'employment': 'Employment Law', 'job': 'Employment Law', 'workplace': 'Employment Law', 'labor': 'Employment Law',
            'fundamental rights': 'Fundamental Rights', 'human rights': 'Fundamental Rights', 'constitutional': 'Fundamental Rights'
        }
        for term, case_type in case_type_mapping.items():
            if term in text_lower:
                entities['case_type'] = case_type
                break
        
        # 4. Location Extraction (basic, can be improved with GeoPy/more robust NER)
        # Prioritize spaCy GPE entities
        if doc:
            for ent in doc.ents:
                if ent.label_ == 'GPE': # Geo-political entity (cities, states, countries)
                    entities['location'] = ent.text
                    break # Take the first one for simplicity
        
        # Fallback regex for common Indian cities/states if spaCy misses
        if 'location' not in entities:
            indian_locations = ['delhi', 'mumbai', 'bengaluru', 'chennai', 'kolkata', 'hyderabad', 'pune', 'gujarat', 'maharashtra', 'karnataka', 'tamil nadu']
            for loc in indian_locations:
                if loc in text_lower:
                    entities['location'] = loc.title()
                    break

        logger.debug(f"Extracted entities: {entities}")
        return entities

    def validate_data_quality(self, data_path: Path) -> Dict[str, Any]:
        """Performs data quality checks on raw JSON files."""
        logger.info(f"Validating data quality in {data_path}...")
        
        total_files = 0
        total_samples = 0
        samples_with_query = 0
        samples_with_domains = 0
        samples_with_eligibility = 0
        
        errors_found = []
        
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {data_path}")
            return {'status': 'no_files_found'}

        for file_path in json_files:
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    errors_found.append(f"File {file_path.name}: Expected a list of samples, got {type(data)}")
                    continue
                
                for i, sample in enumerate(data):
                    total_samples += 1
                    if 'query' in sample and sample['query'].strip():
                        samples_with_query += 1
                    else:
                        errors_found.append(f"File {file_path.name}, Sample {i+1}: Missing or empty 'query' field.")
                    
                    if 'domains' in sample and isinstance(sample['domains'], list) and len(sample['domains']) > 0:
                        samples_with_domains += 1
                    else:
                        errors_found.append(f"File {file_path.name}, Sample {i+1}: Missing or empty 'domains' list.")
                    
                    if 'expected_eligibility' in sample and isinstance(sample['expected_eligibility'], bool):
                        samples_with_eligibility += 1
                    else:
                        errors_found.append(f"File {file_path.name}, Sample {i+1}: Missing or invalid 'expected_eligibility' (must be boolean).")
            
            except json.JSONDecodeError as e:
                errors_found.append(f"File {file_path.name}: Invalid JSON format: {e}")
            except Exception as e:
                errors_found.append(f"File {file_path.name}: Error processing file: {e}")
        
        validation_status = "success" if not errors_found else "warnings_or_errors"
        if errors_found:
            logger.error(f"Data validation completed with {len(errors_found)} errors/warnings.")
            for error in errors_found:
                logger.error(error)
        else:
            logger.info("Data validation completed with no errors.")

        return {
            'status': validation_status,
            'total_files_checked': total_files,
            'total_samples_processed': total_samples,
            'samples_with_query': samples_with_query,
            'samples_with_domains': samples_with_domains,
            'samples_with_eligibility': samples_with_eligibility,
            'errors': errors_found,
            'timestamp': datetime.now().isoformat()
        }

    def run_preprocessing_pipeline(self, data_directory: str) -> Dict[str, str]:
        """
        Runs the complete data preprocessing pipeline:
        1. Loads raw data.
        2. Validates data quality.
        3. Extracts entities and enriches samples.
        4. Splits data into train, validation, and test sets.
        5. Saves processed data to files.
        """
        logger.info(f"Running preprocessing pipeline for data in: {data_directory}")
        data_path = Path(data_directory)
        
        # 1. Load raw data
        all_samples = []
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON data files found in {data_directory}. Please ensure your raw data is in JSON format in the specified directory.")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        # Direct list of samples
                        all_samples.extend(file_data)
                    elif isinstance(file_data, dict) and 'samples' in file_data:
                        # Structured format with metadata and samples
                        samples = file_data['samples']
                        if isinstance(samples, list):
                            all_samples.extend(samples)
                            logger.info(f"Loaded {len(samples)} samples from {file_path.name} (structured format)")
                        else:
                            logger.warning(f"Skipping {file_path.name}: 'samples' key exists but contains {type(samples)}, not a list")
                    else:
                        logger.warning(f"Skipping {file_path.name}: Expected a list of samples or dict with 'samples' key, got {type(file_data)}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path.name}: {e}")
            except Exception as e:
                logger.error(f"Error reading file {file_path.name}: {e}")
        
        if not all_samples:
            raise ValueError("No valid samples loaded after attempting to read all JSON files.")
        
        logger.info(f"Loaded {len(all_samples)} raw samples from {len(json_files)} files.")

        # 2. Validate data quality
        validation_stats = self.validate_data_quality(data_path)
        if validation_stats['status'] != 'success':
            logger.warning("Data quality validation found issues. Proceeding but review errors.")

        # 3. Extract entities and enrich samples
        logger.info("Extracting entities and enriching samples...")
        for i, sample in enumerate(tqdm(all_samples, desc="Extracting legal entities", unit="samples")):
            # Ensure each sample has a unique ID, useful for tracking
            if 'sample_id' not in sample:
                sample['sample_id'] = f"sample_{i+1}"
            
            if 'query' in sample:
                sample['extracted_entities'] = self.extract_entities(sample['query'])
            else:
                sample['extracted_entities'] = {} # No query, no entities
                logger.warning(f"Sample {sample.get('sample_id', i)} has no 'query' field for entity extraction.")
        
        logger.info("Samples enriched with extracted_entities.")

        # 4. Split data into train, validation, and test sets
        logger.info("Splitting data into train, validation, and test sets...")
        
        # Stratified split on 'expected_eligibility' if possible, or 'domains'
        # For multi-label 'domains', direct stratification is complex.
        # We'll stratify by 'expected_eligibility' as it's binary.
        
        labels_for_stratification = [s.get('expected_eligibility', False) for s in all_samples]
        
        train_val_samples, test_samples = train_test_split(
            all_samples, test_size=self.config.DATA_CONFIG['test_split'], 
            random_state=self.config.DATA_CONFIG['random_seed'],
            stratify=labels_for_stratification # Stratify by eligibility
        )
        
        train_samples, val_samples = train_test_split(
            train_val_samples, test_size=self.config.DATA_CONFIG['val_split'] / (1 - self.config.DATA_CONFIG['test_split']),
            random_state=self.config.DATA_CONFIG['random_seed'],
            stratify=[s.get('expected_eligibility', False) for s in train_val_samples] # Stratify by eligibility
        )
        
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")
        logger.info(f"Test samples: {len(test_samples)}")
        
        # 5. Save processed data to files
        processed_data_dir = self.config.RESULTS_DIR / "processed_data"
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for name, data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            file_path = processed_data_dir / f"{name}_samples.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            saved_files[f'{name}_data_file'] = str(file_path)
            logger.info(f"Saved {name} samples to: {file_path}")
            
        logger.info("Data preprocessing pipeline completed.")
        return {
            'total_processed_samples': len(all_samples),
            'train_samples_count': len(train_samples),
            'val_samples_count': len(val_samples),
            'test_samples_count': len(test_samples),
            'saved_files': saved_files,
            'validation_stats': validation_stats
        }