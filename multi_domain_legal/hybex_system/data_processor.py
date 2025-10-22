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


class EnhancedFeatureExtractor:
    """
    Extract rich semantic features from legal queries beyond basic entity extraction.
    
    Features extracted:
    1. Financial indicators: monetary amounts, financial statistics
    2. Urgency signals: high/medium/low urgency levels
    3. Implicit vulnerabilities: health, social, family, age-related, disaster indicators
    4. Domain-specific keywords: consumer, employment, family, rights keywords
    5. Sentiment analysis: negative, positive, neutral tone
    6. Named entities: persons, organizations, locations, dates, monetary values
    
    These features enhance model performance by providing richer semantic signals.
    """
    
    def __init__(self):
        """Initialize the feature extractor with spaCy model."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("EnhancedFeatureExtractor initialized with spaCy model")
        except OSError:
            logger.warning("SpaCy model not found. Downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def extract_enhanced_features(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive semantic features from a legal query.
        
        Args:
            query: The legal query text
            entities: Previously extracted entities (from extract_entities method)
        
        Returns:
            Dictionary containing enhanced features:
            - financial_indicators: Dict with amounts, max/min/avg, has_debt, has_savings
            - urgency_level: str ('high', 'medium', 'low')
            - implicit_vulnerabilities: Dict with health, social, family, age, disaster flags
            - domain_keywords: Dict with counts per domain
            - sentiment: Dict with label and score
            - named_entities: Dict with persons, organizations, locations, dates, money
        """
        enhanced_features = {}
        
        # Extract each feature category
        enhanced_features['financial_indicators'] = self._extract_financial_indicators(query, entities)
        enhanced_features['urgency_level'] = self._extract_urgency_signals(query)
        enhanced_features['implicit_vulnerabilities'] = self._extract_implicit_vulnerabilities(query)
        enhanced_features['domain_keywords'] = self._extract_domain_keywords(query)
        enhanced_features['sentiment'] = self._extract_sentiment(query)
        enhanced_features['named_entities'] = self._extract_named_entities(query)
        
        logger.debug(f"Extracted enhanced features: {enhanced_features}")
        return enhanced_features
    
    def _extract_financial_indicators(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract financial indicators including amounts, debt/savings signals.
        
        Returns:
            Dict with:
            - amounts: List of all monetary amounts found
            - max_amount: Maximum amount
            - min_amount: Minimum amount
            - avg_amount: Average amount
            - has_debt: Boolean indicating debt mentions
            - has_savings: Boolean indicating savings mentions
            - currency_type: Default 'INR'
        """
        indicators = {
            'amounts': [],
            'max_amount': None,
            'min_amount': None,
            'avg_amount': None,
            'has_debt': False,
            'has_savings': False,
            'currency_type': 'INR'
        }
        
        # Extract all monetary amounts using multiple patterns
        amount_patterns = [
            r'rs\.?\s*(\d[\d,]+(?:\.\d{2})?)',  # Rs. 50000 or Rs. 50,000
            r'₹\s*(\d[\d,]+(?:\.\d{2})?)',      # ₹50000
            r'(\d+)\s*lakhs?',                   # 5 lakhs
            r'(\d+)\s*crores?',                  # 2 crores
            r'rupees\s*(\d[\d,]+)',              # rupees 50000
        ]
        
        text_lower = query.lower()
        for pattern in amount_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    # Convert lakhs/crores to rupees
                    if 'lakh' in pattern:
                        amount *= 100000
                    elif 'crore' in pattern:
                        amount *= 10000000
                    indicators['amounts'].append(int(amount))
                except ValueError:
                    continue
        
        # Calculate statistics if amounts found
        if indicators['amounts']:
            indicators['max_amount'] = max(indicators['amounts'])
            indicators['min_amount'] = min(indicators['amounts'])
            indicators['avg_amount'] = sum(indicators['amounts']) // len(indicators['amounts'])
        
        # Check for debt indicators
        debt_keywords = ['loan', 'debt', 'owe', 'borrow', 'credit', 'mortgage', 'liability', 'dues']
        indicators['has_debt'] = any(keyword in text_lower for keyword in debt_keywords)
        
        # Check for savings indicators
        savings_keywords = ['savings', 'deposit', 'investment', 'fixed deposit', 'fd', 'assets']
        indicators['has_savings'] = any(keyword in text_lower for keyword in savings_keywords)
        
        # Add amounts from entities if available
        if 'annual_income' in entities and entities['annual_income'] not in indicators['amounts']:
            indicators['amounts'].append(entities['annual_income'])
        if 'goods_value' in entities and entities['goods_value'] not in indicators['amounts']:
            indicators['amounts'].append(entities['goods_value'])
        
        return indicators
    
    def _extract_urgency_signals(self, query: str) -> str:
        """
        Determine urgency level based on keywords and context.
        
        Returns:
            'high', 'medium', or 'low'
        """
        text_lower = query.lower()
        
        # High urgency indicators
        high_urgency_keywords = [
            'urgent', 'emergency', 'immediate', 'asap', 'critical', 
            'dying', 'life-threatening', 'eviction', 'terminate', 
            'fired', 'dismissed', 'arrest', 'threat', 'violence',
            'abuse', 'harassment', 'danger', 'crisis'
        ]
        
        # Medium urgency indicators
        medium_urgency_keywords = [
            'soon', 'quickly', 'within', 'days', 'deadline',
            'notice', 'warning', 'concern', 'worried', 'anxious',
            'problem', 'issue', 'trouble', 'dispute'
        ]
        
        # Check for high urgency
        if any(keyword in text_lower for keyword in high_urgency_keywords):
            return 'high'
        
        # Check for medium urgency
        if any(keyword in text_lower for keyword in medium_urgency_keywords):
            return 'medium'
        
        # Default to low urgency
        return 'low'
    
    def _extract_implicit_vulnerabilities(self, query: str) -> Dict[str, bool]:
        """
        Identify implicit vulnerability indicators that may not be explicitly stated.
        
        Returns:
            Dict with flags for:
            - health_vulnerability: Medical/health issues
            - social_vulnerability: Caste/social exclusion
            - family_vulnerability: Family issues, dependents
            - age_vulnerability: Elderly or minor
            - disaster_vulnerability: Natural disasters, accidents
        """
        vulnerabilities = {
            'health_vulnerability': False,
            'social_vulnerability': False,
            'family_vulnerability': False,
            'age_vulnerability': False,
            'disaster_vulnerability': False
        }
        
        text_lower = query.lower()
        
        # Health vulnerability indicators
        health_keywords = [
            'sick', 'ill', 'disease', 'medical', 'hospital', 'doctor',
            'treatment', 'medicine', 'disabled', 'handicap', 'injury',
            'cancer', 'diabetes', 'heart', 'mental health', 'depression',
            'accident', 'surgery'
        ]
        vulnerabilities['health_vulnerability'] = any(kw in text_lower for kw in health_keywords)
        
        # Social vulnerability indicators
        social_keywords = [
            'discrimination', 'caste', 'scheduled', 'dalit', 'tribal',
            'minority', 'backward', 'bpl', 'poor', 'poverty', 'slum',
            'homeless', 'refugee', 'migrant', 'exclusion', 'marginalized'
        ]
        vulnerabilities['social_vulnerability'] = any(kw in text_lower for kw in social_keywords)
        
        # Family vulnerability indicators
        family_keywords = [
            'children', 'child', 'dependent', 'elderly parent', 'pregnant',
            'single parent', 'widow', 'orphan', 'divorce', 'custody',
            'domestic', 'family violence', 'dowry', 'maintenance'
        ]
        vulnerabilities['family_vulnerability'] = any(kw in text_lower for kw in family_keywords)
        
        # Age vulnerability indicators
        age_keywords = [
            'senior citizen', 'elderly', 'old age', 'retirement',
            'minor', 'juvenile', 'young', 'child', 'student'
        ]
        vulnerabilities['age_vulnerability'] = any(kw in text_lower for kw in age_keywords)
        
        # Disaster vulnerability indicators
        disaster_keywords = [
            'flood', 'earthquake', 'cyclone', 'fire', 'disaster',
            'calamity', 'pandemic', 'covid', 'lockdown', 'emergency'
        ]
        vulnerabilities['disaster_vulnerability'] = any(kw in text_lower for kw in disaster_keywords)
        
        return vulnerabilities
    
    def _extract_domain_keywords(self, query: str) -> Dict[str, int]:
        """
        Count domain-specific keywords to identify relevant legal domains.
        
        Returns:
            Dict with keyword counts for each domain:
            - consumer_protection: Keywords related to consumer issues
            - employment_law: Keywords related to employment
            - family_law: Keywords related to family matters
            - civil_rights: Keywords related to rights and discrimination
        """
        domain_keywords = {
            'consumer_protection': 0,
            'employment_law': 0,
            'family_law': 0,
            'civil_rights': 0
        }
        
        text_lower = query.lower()
        
        # Consumer protection keywords
        consumer_keywords = [
            'product', 'goods', 'service', 'purchase', 'buy', 'sold',
            'defective', 'warranty', 'guarantee', 'refund', 'return',
            'seller', 'merchant', 'shop', 'store', 'consumer', 'complaint',
            'fraud', 'cheating', 'misleading', 'advertisement'
        ]
        domain_keywords['consumer_protection'] = sum(1 for kw in consumer_keywords if kw in text_lower)
        
        # Employment law keywords
        employment_keywords = [
            'job', 'work', 'employee', 'employer', 'company', 'office',
            'salary', 'wage', 'terminate', 'fire', 'dismiss', 'resign',
            'contract', 'notice', 'leave', 'working hours', 'overtime',
            'promotion', 'transfer', 'harassment', 'discrimination', 'labor'
        ]
        domain_keywords['employment_law'] = sum(1 for kw in employment_keywords if kw in text_lower)
        
        # Family law keywords
        family_keywords = [
            'marriage', 'divorce', 'spouse', 'husband', 'wife', 'children',
            'custody', 'maintenance', 'alimony', 'dowry', 'domestic',
            'family', 'parent', 'guardian', 'adoption', 'inheritance',
            'property', 'will', 'succession'
        ]
        domain_keywords['family_law'] = sum(1 for kw in family_keywords if kw in text_lower)
        
        # Civil rights keywords
        rights_keywords = [
            'right', 'rights', 'discrimination', 'equality', 'freedom',
            'liberty', 'justice', 'legal aid', 'court', 'lawyer',
            'police', 'arrest', 'bail', 'detention', 'constitution',
            'fundamental', 'violation', 'abuse'
        ]
        domain_keywords['civil_rights'] = sum(1 for kw in rights_keywords if kw in text_lower)
        
        return domain_keywords
    
    def _extract_sentiment(self, query: str) -> Dict[str, Any]:
        """
        Analyze sentiment/tone of the query.
        
        Returns:
            Dict with:
            - label: 'negative', 'positive', or 'neutral'
            - score: Float between -1.0 (very negative) and 1.0 (very positive)
        """
        text_lower = query.lower()
        
        # Simple rule-based sentiment analysis
        # Negative sentiment indicators
        negative_words = [
            'not', 'no', 'never', 'refuse', 'deny', 'reject', 'unfair',
            'wrong', 'bad', 'terrible', 'awful', 'worst', 'poor', 'fail',
            'problem', 'issue', 'trouble', 'difficult', 'hard', 'struggle',
            'suffer', 'pain', 'hurt', 'abuse', 'harassment', 'threat',
            'angry', 'upset', 'worried', 'concerned', 'afraid', 'scared'
        ]
        
        # Positive sentiment indicators
        positive_words = [
            'yes', 'good', 'great', 'excellent', 'best', 'happy', 'satisfied',
            'pleased', 'grateful', 'thank', 'appreciate', 'help', 'support',
            'resolve', 'solution', 'improve', 'better', 'fair', 'right', 'just'
        ]
        
        # Count occurrences
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in positive_words)
        
        # Calculate score (normalized to -1 to 1 range)
        total_words = len(query.split())
        if total_words == 0:
            score = 0.0
        else:
            score = (positive_count - negative_count) / max(total_words, 1)
            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
        
        # Determine label
        if score < -0.1:
            label = 'negative'
        elif score > 0.1:
            label = 'positive'
        else:
            label = 'neutral'
        
        return {
            'label': label,
            'score': round(score, 3)
        }
    
    def _extract_named_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract named entities using spaCy NER.
        
        Returns:
            Dict with lists of:
            - persons: Person names
            - organizations: Organization names
            - locations: Locations (GPE, LOC)
            - dates: Dates and times
            - money: Monetary values
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': []
        }
        
        if not self.nlp:
            logger.warning("spaCy model not available for NER")
            return entities
        
        try:
            doc = self.nlp(query)
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ('GPE', 'LOC'):
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'MONEY':
                    entities['money'].append(ent.text)
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
        
        except Exception as e:
            logger.error(f"Error during NER extraction: {e}")
        
        return entities


def preprocess_with_enhanced_features(
    query: str,
    basic_entities: Dict[str, Any],
    feature_extractor: EnhancedFeatureExtractor
) -> Dict[str, Any]:
    """
    Convenience function to combine basic entity extraction with enhanced features.
    
    Args:
        query: Legal query text
        basic_entities: Entities from DataPreprocessor.extract_entities()
        feature_extractor: Instance of EnhancedFeatureExtractor
    
    Returns:
        Dict containing both basic entities and enhanced features
    """
    enhanced_features = feature_extractor.extract_enhanced_features(query, basic_entities)
    
    return {
        'query': query,
        'basic_entities': basic_entities,
        'enhanced_features': enhanced_features
    }


class DataPreprocessor:
    """Preprocesses raw legal data into structured formats for training and evaluation."""
    
    def __init__(self, config: HybExConfig, use_enhanced_features: bool = True):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: HybExConfig instance
            use_enhanced_features: If True, extract enhanced semantic features in addition to basic entities
        """
        self.config = config
        self.nlp = self._load_spacy_model() # Load spaCy model for NER
        self.use_enhanced_features = use_enhanced_features
        
        # Initialize enhanced feature extractor if enabled
        if self.use_enhanced_features:
            self.enhanced_feature_extractor = EnhancedFeatureExtractor()
            logger.info("DataPreprocessor initialized with enhanced feature extraction enabled")
        else:
            self.enhanced_feature_extractor = None
            logger.info("DataPreprocessor initialized with basic feature extraction only")
        
        self.setup_logging()
        logger.info("DataPreprocessor initialization complete.")
    
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
    
    def extract_all_features(self, query: str) -> Dict[str, Any]:
        """
        Extract both basic entities and enhanced features from a query.
        
        This is the main feature extraction method that should be used during preprocessing.
        It extracts basic entities (income, age, etc.) and optionally enhanced semantic features
        (financial indicators, urgency, vulnerabilities, etc.).
        
        Args:
            query: The legal query text
        
        Returns:
            Dict containing:
            - If use_enhanced_features=True: {'basic_entities': {...}, 'enhanced_features': {...}}
            - If use_enhanced_features=False: Just the basic entities dict (backward compatible)
        """
        # Always extract basic entities
        basic_entities = self.extract_entities(query)
        
        # If enhanced features are disabled, return basic entities only (backward compatible)
        if not self.use_enhanced_features or not self.enhanced_feature_extractor:
            return basic_entities
        
        # Extract enhanced features
        try:
            enhanced_features = self.enhanced_feature_extractor.extract_enhanced_features(query, basic_entities)
            return {
                'basic_entities': basic_entities,
                'enhanced_features': enhanced_features
            }
        except Exception as e:
            logger.error(f"Failed to extract enhanced features: {e}. Falling back to basic entities only.")
            return basic_entities

    def normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a training sample to ensure consistent field names.
        
        This ensures compatibility between data generation, preprocessing, and training.
        Handles legacy field names and adds required aliases.
        
        Args:
            sample: Raw sample dictionary
        
        Returns:
            Normalized sample with consistent keys:
            - 'query': User's legal query text
            - 'text': Alias for 'query' (for compatibility)
            - 'domains': List of legal domains
            - 'expected_eligibility': Boolean eligibility label
            - 'label': Integer eligibility label (for PyTorch)
            - 'extracted_entities': Entity dictionary (if available)
            - 'prolog_facts': Prolog facts list (if available)
        """
        normalized = {}
        
        # Core text field (required)
        query_text = sample.get('query') or sample.get('text') or sample.get('input_text', '')
        normalized['query'] = query_text
        normalized['text'] = query_text  # Add alias for compatibility
        
        # Domains (required)
        normalized['domains'] = sample.get('domains', [])
        
        # Eligibility label (required)
        eligibility = sample.get('expected_eligibility', sample.get('eligibility', False))
        normalized['expected_eligibility'] = eligibility
        normalized['label'] = int(eligibility)  # Integer label for PyTorch
        
        # Entities (optional but recommended)
        normalized['extracted_entities'] = sample.get('extracted_entities', {})
        
        # Prolog facts (optional)
        normalized['prolog_facts'] = sample.get('prolog_facts', [])
        
        # Preserve other metadata fields
        for key in ['legal_reasoning', 'user_demographics', 'case_complexity', 
                    'priority_level', 'sample_id', 'confidence']:
            if key in sample:
                normalized[key] = sample[key]
        
        return normalized

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
                feature_type = "entities and enhanced features" if self.use_enhanced_features else "entities"
                logger.info(f"Extracting {feature_type} for {missing_entities_count} samples without 'extracted_entities'...")
                
                extraction_errors = 0
                for sample in tqdm(all_pre_split_samples, desc=f"Extracting {feature_type} for pre-split data", unit="samples"):
                    if 'extracted_entities' not in sample:
                        if 'query' in sample:
                            try:
                                # Use extract_all_features which handles both basic and enhanced features
                                sample['extracted_entities'] = self.extract_all_features(sample['query'])
                            except Exception as e:
                                logger.error(f"Failed to extract features for {sample.get('sample_id')}: {e}")
                                sample['extracted_entities'] = {}
                                extraction_errors += 1
                        else:
                            sample['extracted_entities'] = {}
                
                if extraction_errors > 0:
                    logger.warning(f"⚠️  Feature extraction failed for {extraction_errors} samples")
                
                logger.info(f"✅ Feature extraction completed for pre-split data ({feature_type})")
                
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
        feature_type = "entities and enhanced features" if self.use_enhanced_features else "entities"
        logger.info(f"Extracting {feature_type} and enriching samples...")
        extraction_errors = 0
        
        desc_text = "Extracting legal features" if self.use_enhanced_features else "Extracting legal entities"
        for i, sample in enumerate(tqdm(all_samples, desc=desc_text, unit="samples")):
            # Ensure each sample has a unique ID, useful for tracking
            if 'sample_id' not in sample:
                sample['sample_id'] = f"sample_{i+1}"
            
            if 'query' in sample:
                try:
                    # Use extract_all_features which handles both basic and enhanced features
                    sample['extracted_entities'] = self.extract_all_features(sample['query'])
                except Exception as e:
                    logger.error(f"Failed to extract features for sample {sample.get('sample_id')}: {e}")
                    sample['extracted_entities'] = {}  # Empty dict as fallback
                    extraction_errors += 1
            else:
                sample['extracted_entities'] = {}  # No query, no entities
                logger.warning(f"Sample {sample.get('sample_id', i)} has no 'query' field for feature extraction.")
        
        if extraction_errors > 0:
            logger.warning(f"⚠️  Feature extraction failed for {extraction_errors}/{len(all_samples)} samples")
        else:
            logger.info(f"✅ Successfully extracted {feature_type} for all {len(all_samples)} samples")

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