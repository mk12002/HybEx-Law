"""
Stage 1 Classifier: Entity Presence Detection

This classifier determines which types of legal entities (income, case_type, social_category)
are present in a user's query. It guides which Stage 2 extractors should be activated.
"""

from typing import List, Set
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiLabelBinarizer
import pickle
from pathlib import Path


class EntityPresenceClassifier:
    """
    Multi-label classifier to detect presence of legal entities in text.
    
    Uses TF-IDF features with logistic regression for reliable entity detection.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the entity presence classifier.
        
        Args:
            model_path: Path to pre-trained model (if available)
        """
        self.entities = ['income', 'case_type', 'social_category']
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(random_state=42)
        self.label_binarizer = MultiLabelBinarizer()
        
        # Entity-specific keywords for rule-based fallback
        self.entity_keywords = {
            'income': {
                'salary', 'income', 'earn', 'money', 'pay', 'wage', 'job', 'work', 
                'unemployed', 'employment', 'business', 'rupees', 'rs', 'lakh', 
                'thousand', 'per month', 'annually', 'lost job', 'no income'
            },
            'case_type': {
                'case', 'dispute', 'court', 'legal', 'landlord', 'tenant', 'evict',
                'divorce', 'custody', 'property', 'labor', 'domestic violence',
                'harassment', 'cheating', 'fraud', 'accident', 'compensation'
            },
            'social_category': {
                'woman', 'child', 'sc', 'st', 'scheduled caste', 'scheduled tribe',
                'general category', 'obc', 'disabled', 'handicap', 'custody',
                'prison', 'jail', 'industrial worker', 'factory'
            }
        }
        
        self.is_trained = False
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def predict(self, text: str) -> List[str]:
        """
        Predict which entities are present in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of entity names that are present
        """
        if self.is_trained:
            return self._predict_ml(text)
        else:
            return self._predict_rule_based(text)
    
    def _predict_rule_based(self, text: str) -> List[str]:
        """
        Rule-based prediction using keyword matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities
        """
        text_lower = text.lower()
        detected_entities = []
        
        for entity, keywords in self.entity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_entities.append(entity)
        
        # Ensure at least one entity is detected
        if not detected_entities:
            # Default assumption: if someone is asking about legal aid,
            # they likely have a case and may have income concerns
            detected_entities = ['case_type', 'income']
        
        return detected_entities
    
    def _predict_ml(self, text: str) -> List[str]:
        """
        Machine learning-based prediction.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities
        """
        # Vectorize the text
        text_vector = self.vectorizer.transform([text])
        
        # Predict probabilities
        probabilities = self.classifier.predict_proba(text_vector)[0]
        
        # Apply threshold (0.3 for multi-label classification)
        threshold = 0.3
        predicted_labels = [
            self.entities[i] for i, prob in enumerate(probabilities) 
            if prob > threshold
        ]
        
        # Fallback to rule-based if no entities detected
        if not predicted_labels:
            predicted_labels = self._predict_rule_based(text)
        
        return predicted_labels
    
    def train(self, texts: List[str], labels: List[List[str]]):
        """
        Train the classifier on annotated data.
        
        Args:
            texts: List of input texts
            labels: List of entity labels for each text
        """
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Prepare labels
        y = self.label_binarizer.fit_transform(labels)
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
    
    def save_model(self, model_path: str):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_binarizer': self.label_binarizer,
            'entities': self.entities,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model(self, model_path: str):
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_binarizer = model_data['label_binarizer']
        self.entities = model_data['entities']
        self.is_trained = model_data['is_trained']
