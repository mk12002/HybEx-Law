"""
Text Preprocessor: Utilities for cleaning and normalizing text

This module provides text preprocessing functionality for the legal NLP pipeline,
including normalization, cleaning, and basic linguistic processing.
"""

import re
import string
from typing import List, Dict


class TextPreprocessor:
    """
    Text preprocessing utilities for legal document processing.
    
    Provides methods for:
    - Text normalization
    - Cleaning and tokenization
    - Legal-specific text processing
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        
        # Common contractions and their expansions
        self.contractions = {
            "won't": "will not",
            "can't": "cannot", 
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        # Legal-specific normalizations
        self.legal_normalizations = {
            "rs.": "rupees",
            "rs ": "rupees ",
            "₹": "rupees",
            "sc/st": "scheduled caste scheduled tribe",
            "s.c.": "scheduled caste",
            "s.t.": "scheduled tribe",
            "obc": "other backward class",
            "fir": "first information report"
        }
        
        # Words to preserve (don't remove during cleaning)
        self.preserve_words = {
            'sc', 'st', 'obc', 'fir', 'rs', 'not', 'no'
        }
    
    def preprocess(self, text: str) -> str:
        """
        Main preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = self._basic_clean(text)
        
        # Expand contractions
        text = self._expand_contractions(text)
        
        # Legal-specific normalizations
        text = self._normalize_legal_terms(text)
        
        # Final cleaning
        text = self._final_clean(text)
        
        return text
    
    def _basic_clean(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """
        Expand common contractions.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _normalize_legal_terms(self, text: str) -> str:
        """
        Normalize legal-specific terms.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized legal terms
        """
        for term, normalized in self.legal_normalizations.items():
            text = text.replace(term, normalized)
        
        return text
    
    def _final_clean(self, text: str) -> str:
        """
        Final cleaning pass.
        
        Args:
            text: Input text
            
        Returns:
            Final cleaned text
        """
        # Remove excessive punctuation (keep single instances)
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[!]{2,}', '!', text)
        
        # Normalize spaces around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Preprocess first
        text = self.preprocess(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def extract_numbers(self, text: str) -> List[str]:
        """
        Extract numeric values from text.
        
        Args:
            text: Input text
            
        Returns:
            List of numeric strings found
        """
        # Pattern for various number formats
        number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        
        return numbers
    
    def extract_currency_amounts(self, text: str) -> List[Dict[str, str]]:
        """
        Extract currency amounts with context.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with amount and context
        """
        # Pattern for currency amounts
        currency_pattern = r'(?:rs\.?|rupees|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        
        amounts = []
        for match in re.finditer(currency_pattern, text, re.IGNORECASE):
            amounts.append({
                'amount': match.group(1),
                'full_match': match.group(0),
                'position': match.span()
            })
        
        return amounts
    
    def clean_for_classification(self, text: str) -> str:
        """
        Special preprocessing for classification tasks.
        
        Args:
            text: Input text
            
        Returns:
            Text optimized for classification
        """
        # Standard preprocessing
        text = self.preprocess(text)
        
        # Remove very common words that don't help classification
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        # Keep important words for legal context
        words = text.split()
        filtered_words = [
            word for word in words 
            if word not in stopwords or word in self.preserve_words
        ]
        
        return ' '.join(filtered_words)
