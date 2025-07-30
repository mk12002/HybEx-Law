"""
Case Type Classifier: Categorizes legal cases for eligibility determination

This classifier identifies the type of legal case from a user's description,
which is crucial for determining eligibility as some case types are excluded
from legal aid services.
"""

import re
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path


class CaseTypeClassifier:
    """
    Classifies legal queries into predefined case types.
    
    Case types include:
    - Property disputes
    - Family matters
    - Labor disputes  
    - Criminal cases
    - Excluded cases (defamation, business disputes, etc.)
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the case type classifier.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        
        # Define case type categories
        self.case_types = {
            'property_dispute': {
                'keywords': [
                    'landlord', 'tenant', 'evict', 'eviction', 'rent', 'house', 'property',
                    'land', 'possession', 'lease', 'apartment', 'flat', 'premises'
                ],
                'excluded': False
            },
            'family_matter': {
                'keywords': [
                    'divorce', 'marriage', 'custody', 'child custody', 'domestic violence',
                    'dowry', 'maintenance', 'alimony', 'wife', 'husband', 'family'
                ],
                'excluded': False
            },
            'labor_dispute': {
                'keywords': [
                    'job', 'employment', 'workplace', 'factory', 'company', 'employer',
                    'salary', 'wages', 'fired', 'terminated', 'industrial', 'worker'
                ],
                'excluded': False
            },
            'criminal_matter': {
                'keywords': [
                    'police', 'fir', 'arrest', 'jail', 'custody', 'crime', 'theft',
                    'assault', 'harassment', 'cheating', 'fraud'
                ],
                'excluded': False
            },
            'consumer_dispute': {
                'keywords': [
                    'consumer', 'product', 'service', 'defective', 'warranty',
                    'refund', 'compensation', 'goods', 'purchase'
                ],
                'excluded': False
            },
            'accident_compensation': {
                'keywords': [
                    'accident', 'injury', 'compensation', 'motor', 'vehicle',
                    'insurance', 'medical', 'hospital', 'disability'
                ],
                'excluded': False
            },
            'defamation': {
                'keywords': [
                    'defamation', 'reputation', 'character', 'slander', 'libel',
                    'false statement', 'honor', 'dignity'
                ],
                'excluded': True
            },
            'business_dispute': {
                'keywords': [
                    'business', 'commercial', 'contract', 'partnership', 'company',
                    'corporate', 'profit', 'loss', 'investment'
                ],
                'excluded': True
            },
            'election_offense': {
                'keywords': [
                    'election', 'vote', 'voting', 'candidate', 'electoral',
                    'poll', 'ballot', 'campaign'
                ],
                'excluded': True
            }
        }
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def extract(self, processed_text: str, original_text: str) -> List[str]:
        """
        Extract case type facts from text.
        
        Args:
            processed_text: Preprocessed text
            original_text: Original query text
            
        Returns:
            List of Prolog facts about case type
        """
        case_type = self.classify_case_type(original_text.lower())
        
        facts = []
        if case_type:
            facts.append(f'case_type(user, "{case_type}")')
        
        return facts
    
    def classify_case_type(self, text: str) -> Optional[str]:
        """
        Classify the case type from text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Predicted case type or None
        """
        if self.is_trained:
            return self._classify_ml(text)
        else:
            return self._classify_rule_based(text)
    
    def _classify_rule_based(self, text: str) -> Optional[str]:
        """
        Rule-based classification using keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            Best matching case type
        """
        scores = {}
        
        for case_type, info in self.case_types.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in text:
                    # Give higher weight to longer, more specific keywords
                    score += len(keyword.split())
            
            if score > 0:
                scores[case_type] = score
        
        if scores:
            # Return case type with highest score
            best_case_type = max(scores.items(), key=lambda x: x[1])[0]
            return best_case_type
        
        return None
    
    def _classify_ml(self, text: str) -> Optional[str]:
        """
        Machine learning-based classification.
        
        Args:
            text: Input text
            
        Returns:
            Predicted case type
        """
        # Vectorize text
        text_vector = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.classifier.predict(text_vector)[0]
        confidence = max(self.classifier.predict_proba(text_vector)[0])
        
        # Return prediction if confidence is high enough
        if confidence > 0.5:
            return prediction
        else:
            # Fallback to rule-based
            return self._classify_rule_based(text)
    
    def is_excluded_case_type(self, case_type: str) -> bool:
        """
        Check if a case type is excluded from legal aid.
        
        Args:
            case_type: Case type to check
            
        Returns:
            True if case type is excluded
        """
        if case_type in self.case_types:
            return self.case_types[case_type]['excluded']
        return False
    
    def train(self, texts: List[str], labels: List[str]):
        """
        Train the classifier on annotated data.
        
        Args:
            texts: List of text samples
            labels: List of corresponding case type labels
        """
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def save_model(self, model_path: str):
        """Save trained model to disk."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'case_types': self.case_types,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model(self, model_path: str):
        """Load pre-trained model from disk."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier'] 
        self.case_types = model_data['case_types']
        self.is_trained = model_data['is_trained']
