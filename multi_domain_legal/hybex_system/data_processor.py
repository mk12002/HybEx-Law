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
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
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

    # In hybex_system/data_processor.py
# REPLACE the entire extract_entities method with this final version.

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts a wide range of legal entities using domain-specific regex and spaCy.
        This is the core of providing facts to the Prolog engine.
        
        **Normalization Strategy:**
        - `annual_income`: ALWAYS in rupees per year (monthly * 12)
        - `employment_duration_days`: ALWAYS in days (months * 30, years * 365)
        - `daily_wage`: ALWAYS rupees per day
        - `age`: Years
        - All monetary values in integer rupees
        - Boolean values: True/False
        - Categorical values: Lowercase strings ('sc', 'st', 'bpl', etc.)
        
        This ensures the Prolog engine and neural models receive consistent,
        comparable values regardless of how they're expressed in the query.
        """
        entities = {}
        text_lower = text.lower()
        doc = self.nlp(text) if self.nlp else None

        # =================================================================
        # General & Demographic Entities
        # =================================================================
        
        # 1. Income Extraction (Robust: handles monthly vs. annual)
        # ALWAYS store as annual income (normalized)
        income_patterns = [
            # Monthly patterns - will be multiplied by 12
            (r'(?:monthly|per month)\s+(?:salary|income|wage|earning).*?rs\.?\s*(\d[\d,]+(?:\.\d{2})?)', 'monthly'),
            (r'(?:salary|income|wage|earning).*?rs\.?\s*(\d[\d,]+(?:\.\d{2})?)\s*(?:per month|monthly)', 'monthly'),
            (r'rs\.?\s*(\d[\d,]+(?:\.\d{2})?)\s*(?:per month|monthly)', 'monthly'),
            # Annual patterns - use as-is
            (r'(?:annual|yearly)\s+(?:salary|income|wage|earning).*?rs\.?\s*(\d[\d,]+(?:\.\d{2})?)', 'annual'),
            (r'(?:salary|income|wage|earning).*?rs\.?\s*(\d[\d,]+(?:\.\d{2})?)\s*(?:per year|annually|yearly|annual)', 'annual'),
            (r'rs\.?\s*(\d[\d,]+(?:\.\d{2})?)\s*(?:per year|annually|yearly|annual)', 'annual'),
            (r'(\d+)\s*lakhs?\s+(?:annual|per year|yearly)', 'annual'),
            # Ambiguous patterns - assume annual if no context
            (r'annual\s+income.*?rs\.?\s*(\d[\d,]+)', 'annual'),
            (r'income.*?rs\.?\s*(\d[\d,]+)', 'annual'),
        ]
        for pattern, period_type in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    # Apply multiplier based on period type to normalize to annual
                    if period_type == 'monthly':
                        annual_income = amount * 12
                    else:  # annual
                        annual_income = amount
                    # Store normalized annual income
                    entities['annual_income'] = int(annual_income)
                    break
                except ValueError:
                    continue

        # 2. Social Category Extraction (Normalized for Prolog)
        # CRITICAL: Check ST before SC to avoid "sc" matching in "scheduled tribe"
        social_category_mapping = {
            'scheduled tribe': 'st', 'st': 'st', 'tribal': 'st', 'adivasi': 'st',  # Check first
            'scheduled caste': 'sc', 'dalit': 'sc',  # Removed standalone 'sc' to avoid ambiguity
            'below poverty line': 'bpl', 'bpl': 'bpl', 'poor': 'bpl',
            'economically weaker': 'ews', 'ews': 'ews',
            'other backward class': 'obc', 'obc': 'obc'
        }
        for term, category in social_category_mapping.items():
            if term in text_lower:
                entities['social_category'] = category
                break
        
        # 3. Age & Gender Extraction
        age_match = re.search(r'(\d+)\s*years?\s*old', text_lower)
        if age_match:
            entities['age'] = int(age_match.group(1))
            
        if 'woman' in text_lower or 'female' in text_lower or 'wife' in text_lower or 'mother' in text_lower:
            entities['gender'] = 'female'
        elif 'man' in text_lower or 'male' in text_lower or 'husband' in text_lower or 'father' in text_lower:
            entities['gender'] = 'male'

        # =================================================================
        # Employment Law Entities
        # =================================================================
        # Employment duration - only match in employment context (working, employed, job, service)
        # ALWAYS store as days (normalized)
        employment_duration_patterns = [
            (r'(?:working|employed|job|service|employment).*?(\d+)\s*(day|month|year)s?', None),
            (r'(\d+)\s*(day|month|year)s?\s*(?:of employment|of work|of service|working)', None),
            (r'(\d+)\s*(day|month|year)s?\s+(?:contract|tenure|period)', None),
        ]
        for pattern, _ in employment_duration_patterns:
            duration_match = re.search(pattern, text_lower)
            if duration_match:
                value, unit = int(duration_match.group(1)), duration_match.group(2)
                # Normalize to days
                if 'month' in unit:
                    days = value * 30
                elif 'year' in unit:
                    days = value * 365
                else:  # already in days
                    days = value
                entities['employment_duration_days'] = days
                break

        # Daily wage extraction - handles multiple patterns
        # Store as-is (already normalized to per-day rate)
        wage_patterns = [
            r'daily\s+wage.*?rs\.?\s*(\d+)',
            r'rs\.?\s*(\d+)\s*(?:per day|daily)',
            r'(\d+)\s*(?:per day|daily\s+wage)'
        ]
        for pattern in wage_patterns:
            wage_match = re.search(pattern, text_lower)
            if wage_match:
                entities['daily_wage'] = int(wage_match.group(1))
                break

        notice_match = re.search(r'(\d+)\s*days?\s*notice', text_lower)
        if notice_match:
            entities['notice_period_given'] = int(notice_match.group(1))
            
        if "no hearing" in text_lower or "without a hearing" in text_lower:
            entities['disciplinary_hearing_conducted'] = False

        # =================================================================
        # Family Law Entities
        # =================================================================
        if 'married' in text_lower or 'husband' in text_lower or 'wife' in text_lower:
            entities['is_married'] = True
            
        children_match = re.search(r'(\d+)\s*(child|children|kids)', text_lower)
        if children_match:
            entities['has_children'] = int(children_match.group(1))
        elif 'my son' in text_lower or 'my daughter' in text_lower:
            entities['has_children'] = 1

        # =================================================================
        # Consumer Protection Entities
        # =================================================================
        value_match = re.search(r'(?:worth|value|paid|for|cost)\s*(?:rs\.?)?\s*(\d[\d,]+)', text_lower)
        if value_match:
            entities['goods_value'] = int(value_match.group(1).replace(',', ''))

        date_match = re.search(r'on\s*(\d{4}-\d{2}-\d{2})', text_lower)
        if date_match:
            entities['incident_date'] = date_match.group(1)

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
        
        NOTE: If pre-split files (*_split.json) are detected, skips re-splitting.
        """
        logger.info(f"Running preprocessing pipeline for data in: {data_directory}")
        data_path = Path(data_directory)
        
        # ========================================
        # CHECK FOR PRE-SPLIT DATA FILES FIRST
        # ========================================
        train_file = data_path / "train_split.json"
        val_file = data_path / "val_split.json"
        test_file = data_path / "test_split.json"
        
        if train_file.exists() and val_file.exists() and test_file.exists():
            logger.info("✅ Detected pre-split data files (*_split.json). Loading without re-splitting...")
            
            # Load pre-split files directly
            with open(train_file, 'r', encoding='utf-8') as f:
                train_samples = json.load(f)
            with open(val_file, 'r', encoding='utf-8') as f:
                val_samples = json.load(f)
            with open(test_file, 'r', encoding='utf-8') as f:
                test_samples = json.load(f)
            
            logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
            
            # CRITICAL FIX: Always extract entities if missing
            all_pre_split_samples = train_samples + val_samples + test_samples
            missing_entities_count = sum(1 for s in all_pre_split_samples if 'extracted_entities' not in s)
            
            if missing_entities_count > 0:
                logger.info(f"Extracting entities for {missing_entities_count} samples without 'extracted_entities'...")
                
                extraction_errors = 0
                for sample in tqdm(all_pre_split_samples, desc="Extracting entities for pre-split data", unit="samples"):
                    if 'extracted_entities' not in sample:
                        if 'query' in sample:
                            try:
                                sample['extracted_entities'] = self.extract_entities(sample['query'])
                            except Exception as e:
                                logger.error(f"Failed to extract entities for {sample.get('sample_id')}: {e}")
                                sample['extracted_entities'] = {}
                                extraction_errors += 1
                        else:
                            sample['extracted_entities'] = {}
                
                if extraction_errors > 0:
                    logger.warning(f"⚠️  Entity extraction failed for {extraction_errors} samples")
                
                logger.info("✅ Entity extraction completed for pre-split data")
                
                # ✅ CRITICAL: Save updated data BACK to original *_split.json files
                logger.info("💾 Saving extracted entities back to original split files...")
                with open(train_file, 'w', encoding='utf-8') as f:
                    json.dump(train_samples, f, indent=2, ensure_ascii=False)
                with open(val_file, 'w', encoding='utf-8') as f:
                    json.dump(val_samples, f, indent=2, ensure_ascii=False)
                with open(test_file, 'w', encoding='utf-8') as f:
                    json.dump(test_samples, f, indent=2, ensure_ascii=False)
                logger.info("✅ Entities saved back to split files - future runs will skip extraction")
            else:
                logger.info("✅ All samples already have extracted_entities")

            # Save to processed_data directory as well
            processed_dir = self.config.RESULTS_DIR / "processed_data"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_file_out = processed_dir / "train_samples.json"
            val_file_out = processed_dir / "val_samples.json"
            test_file_out = processed_dir / "test_samples.json"
            
            with open(train_file_out, 'w', encoding='utf-8') as f:
                json.dump(train_samples, f, indent=2, ensure_ascii=False)
            with open(val_file_out, 'w', encoding='utf-8') as f:
                json.dump(val_samples, f, indent=2, ensure_ascii=False)
            with open(test_file_out, 'w', encoding='utf-8') as f:
                json.dump(test_samples, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Pre-split data with entities saved to both locations")
            
            return {
                'total_processed_samples': len(all_pre_split_samples),
                'train_samples_count': len(train_samples),
                'val_samples_count': len(val_samples),
                'test_samples_count': len(test_samples),
                'saved_files': {
                    'train_data_file': str(train_file_out),
                    'val_data_file': str(val_file_out),
                    'test_data_file': str(test_file_out)
                },
                'validation_stats': {
                    'status': 'presplit_with_entities',
                    'message': 'Pre-split data loaded and entities extracted',
                    'missing_entities_filled': missing_entities_count
                }
            }
        
        # ========================================
        # OTHERWISE: PROCEED WITH ORIGINAL PREPROCESSING
        # ========================================
        logger.info("No pre-split files detected. Proceeding with full preprocessing pipeline...")
        
        # 1. Load raw data
        all_samples = []
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON data files found in {data_directory}. Please ensure your raw data is in JSON format in the specified directory.")
        
        for file_path in json_files:
            # Skip scraping report files - they contain metadata, not training samples
            if 'scraping_report' in file_path.name:
                logger.info(f"Skipping non-sample JSON report file: {file_path.name}")
                continue
            
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
        extraction_errors = 0
        
        for i, sample in enumerate(tqdm(all_samples, desc="Extracting legal entities", unit="samples")):
            # Ensure each sample has a unique ID, useful for tracking
            if 'sample_id' not in sample:
                sample['sample_id'] = f"sample_{i+1}"
            
            if 'query' in sample:
                try:
                    sample['extracted_entities'] = self.extract_entities(sample['query'])
                except Exception as e:
                    logger.error(f"Failed to extract entities for sample {sample.get('sample_id')}: {e}")
                    sample['extracted_entities'] = {}  # Empty dict as fallback
                    extraction_errors += 1
            else:
                sample['extracted_entities'] = {}  # No query, no entities
                logger.warning(f"Sample {sample.get('sample_id', i)} has no 'query' field for entity extraction.")
        
        if extraction_errors > 0:
            logger.warning(f"⚠️  Entity extraction failed for {extraction_errors}/{len(all_samples)} samples")
        else:
            logger.info(f"✅ Successfully extracted entities for all {len(all_samples)} samples")

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