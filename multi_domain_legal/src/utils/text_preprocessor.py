"""
Text preprocessing utilities for multi-domain legal NLP system.

This module provides comprehensive text preprocessing capabilities
including cleaning, normalization, tokenization, and legal-specific
text processing functions.
"""

import re
import string
from typing import List, Dict, Any, Optional
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class LegalTextPreprocessor:
    """
    Comprehensive text preprocessor for legal domain queries.
    
    Handles cleaning, normalization, tokenization, and legal-specific
    preprocessing tasks while preserving important legal terms and entities.
    """
    
    def __init__(self):
        self.setup_nlp_tools()
        self.legal_stopwords = self._create_legal_stopwords()
        self.legal_abbreviations = self._create_legal_abbreviations()
        self.currency_patterns = self._create_currency_patterns()
        
    def setup_nlp_tools(self):
        """Initialize NLP tools and models"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"⚠️  NLTK setup error: {e}")
            self.stemmer = None
            self.lemmatizer = None
            self.stop_words = set()
    
    def _create_legal_stopwords(self) -> set:
        """Create legal domain specific stopwords"""
        return {
            'law', 'legal', 'act', 'section', 'rule', 'provision', 'clause',
            'article', 'under', 'according', 'pursuant', 'whereas', 'thereof',
            'herein', 'hereby', 'hereafter', 'aforementioned', 'aforesaid'
        }
    
    def _create_legal_abbreviations(self) -> Dict[str, str]:
        """Create mapping of legal abbreviations to full forms"""
        return {
            # Common legal abbreviations
            'sc': 'scheduled caste',
            'st': 'scheduled tribe',
            'rti': 'right to information',
            'fir': 'first information report',
            'crl': 'criminal',
            'posh': 'prevention of sexual harassment',
            'rera': 'real estate regulatory authority',
            'fssai': 'food safety and standards authority of india',
            'nhrc': 'national human rights commission',
            'icc': 'internal complaints committee',
            'pf': 'provident fund',
            'esi': 'employees state insurance',
            'bpl': 'below poverty line',
            'apl': 'above poverty line',
            
            # Currency and units
            'rs': 'rupees',
            'inr': 'indian rupees',
            'k': '000',
            'lakh': '100000',
            'lakhs': '100000',
            'crore': '10000000',
            'crores': '10000000'
        }
    
    def _create_currency_patterns(self) -> List[Dict[str, Any]]:
        """Create patterns for currency and amount detection"""
        return [
            {
                'pattern': r'(?:rs\.?|rupees?|inr)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                'type': 'currency',
                'unit': 'rupees'
            },
            {
                'pattern': r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs\.?|rupees?|inr)',
                'type': 'currency', 
                'unit': 'rupees'
            },
            {
                'pattern': r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|lakhs)',
                'type': 'currency',
                'multiplier': 100000
            },
            {
                'pattern': r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|crores)',
                'type': 'currency',
                'multiplier': 10000000
            }
        ]
    
    def basic_clean(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve legal punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\'\"\/]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Strip and lowercase
        text = text.strip().lower()
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand legal abbreviations to full forms.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        expanded_text = text
        
        for abbr, full_form in self.legal_abbreviations.items():
            # Word boundary matching for abbreviations
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded_text = re.sub(pattern, full_form, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text
    
    def normalize_currency(self, text: str) -> str:
        """
        Normalize currency mentions to standard format.
        
        Args:
            text: Input text with currency mentions
            
        Returns:
            Text with normalized currency
        """
        normalized_text = text
        
        for pattern_info in self.currency_patterns:
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            
            for match in reversed(list(matches)):  # Reverse to maintain positions
                amount_str = match.group(1)
                amount_num = float(amount_str.replace(',', ''))
                
                if 'multiplier' in pattern_info:
                    amount_num *= pattern_info['multiplier']
                
                # Replace with standardized format
                replacement = f"income_monthly {int(amount_num)}"
                normalized_text = normalized_text[:match.start()] + replacement + normalized_text[match.end():]
        
        return normalized_text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'laws': []
        }
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON']:
                entities['persons'].append(ent.text)
            elif ent.label_ in ['ORG']:
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(ent.text)
            elif ent.label_ in ['MONEY']:
                entities['money'].append(ent.text)
        
        # Extract legal act mentions
        legal_patterns = [
            r'\b\w+\s+(?:act|code|law|rules?)\b',
            r'\barticle\s+\d+\b',
            r'\bsection\s+\d+\b'
        ]
        
        for pattern in legal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['laws'].append(match.group())
        
        return entities
    
    def tokenize(self, text: str, method: str = "spacy") -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            method: Tokenization method ("spacy", "nltk", or "simple")
            
        Returns:
            List of tokens
        """
        if method == "spacy" and self.nlp:
            doc = self.nlp(text)
            return [token.text.lower() for token in doc if not token.is_space]
        elif method == "nltk":
            return [word.lower() for word in word_tokenize(text)]
        else:
            # Simple tokenization
            return re.findall(r'\b\w+\b', text.lower())
    
    def remove_stopwords(self, tokens: List[str], preserve_legal: bool = True) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            preserve_legal: Whether to preserve legal terms
            
        Returns:
            Filtered tokens
        """
        if preserve_legal:
            # Don't remove legal domain specific terms
            legal_terms = {
                'legal', 'law', 'act', 'right', 'court', 'case', 'claim',
                'defendant', 'plaintiff', 'judge', 'lawyer', 'attorney',
                'contract', 'agreement', 'violation', 'breach', 'damages'
            }
            
            return [token for token in tokens 
                   if token not in self.stop_words or token in legal_terms]
        else:
            return [token for token in tokens if token not in self.stop_words]
    
    def preprocess_query(self, 
                        query: str, 
                        clean: bool = True,
                        expand_abbr: bool = True,
                        normalize_curr: bool = True,
                        extract_ents: bool = False,
                        tokenize: bool = False) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for legal queries.
        
        Args:
            query: Raw legal query
            clean: Apply basic cleaning
            expand_abbr: Expand abbreviations
            normalize_curr: Normalize currency mentions
            extract_ents: Extract named entities
            tokenize: Return tokenized version
            
        Returns:
            Dictionary with processed text and metadata
        """
        result = {
            'original': query,
            'processed': query,
            'entities': {},
            'tokens': []
        }
        
        if not query:
            return result
        
        processed_text = query
        
        # Apply processing steps
        if clean:
            processed_text = self.basic_clean(processed_text)
        
        if expand_abbr:
            processed_text = self.expand_abbreviations(processed_text)
        
        if normalize_curr:
            processed_text = self.normalize_currency(processed_text)
        
        result['processed'] = processed_text
        
        # Extract entities if requested
        if extract_ents:
            result['entities'] = self.extract_entities(processed_text)
        
        # Tokenize if requested
        if tokenize:
            tokens = self.tokenize(processed_text)
            result['tokens'] = self.remove_stopwords(tokens)
        
        return result
    
    def preprocess_for_classification(self, query: str) -> str:
        """
        Preprocess text specifically for domain classification.
        
        Args:
            query: Raw legal query
            
        Returns:
            Preprocessed text optimized for classification
        """
        # Apply comprehensive preprocessing
        result = self.preprocess_query(
            query,
            clean=True,
            expand_abbr=True,
            normalize_curr=True,
            extract_ents=False,
            tokenize=False
        )
        
        return result['processed']
    
    def preprocess_for_extraction(self, query: str) -> Dict[str, Any]:
        """
        Preprocess text for fact extraction with entity information.
        
        Args:
            query: Raw legal query
            
        Returns:
            Comprehensive preprocessing results with entities
        """
        return self.preprocess_query(
            query,
            clean=True,
            expand_abbr=True,
            normalize_curr=True,
            extract_ents=True,
            tokenize=True
        )
