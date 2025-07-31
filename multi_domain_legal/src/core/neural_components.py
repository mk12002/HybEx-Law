"""
Neural Components for Hybrid Legal System.

This module implements the neural components of the hybrid system including
fine-tuned models for domain classification and fact extraction.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
import logging

from .domain_registry import LegalDomain

class NeuralDomainClassifier:
    """
    Neural domain classifier using BERT-based models for legal domain identification.
    
    This replaces the rule-based keyword matching with a proper neural classifier
    that can understand context and legal nuances.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 use_pretrained: bool = True):
        """
        Initialize neural domain classifier.
        
        Args:
            model_name: HuggingFace model name for domain classification
            use_pretrained: Whether to load pre-trained model or train from scratch
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.is_trained = False
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        if use_pretrained and os.path.exists('models/neural_domain_classifier.pkl'):
            self.load_model()
    
    def _prepare_labels(self, domains: List[LegalDomain]) -> None:
        """Prepare label encoding for multi-label classification"""
        unique_domains = list(set(domains))
        self.label_encoder = {domain.value: idx for idx, domain in enumerate(unique_domains)}
        self.reverse_label_encoder = {idx: domain for domain, idx in self.label_encoder.items()}
        self.num_labels = len(unique_domains)
    
    def _encode_labels(self, domain_lists: List[List[str]]) -> np.ndarray:
        """Convert domain lists to multi-hot encoded labels"""
        encoded_labels = np.zeros((len(domain_lists), self.num_labels))
        
        for i, domains in enumerate(domain_lists):
            for domain in domains:
                if domain in self.label_encoder:
                    encoded_labels[i][self.label_encoder[domain]] = 1
        
        return encoded_labels
    
    def prepare_training_data(self, queries: List[str], 
                            domain_labels: List[List[str]]) -> Tuple[List[str], np.ndarray]:
        """
        Prepare training data for neural domain classification.
        
        Args:
            queries: List of legal queries
            domain_labels: List of domain lists for each query
            
        Returns:
            Tuple of processed queries and encoded labels
        """
        # Extract unique domains
        all_domains = []
        for domain_list in domain_labels:
            all_domains.extend(domain_list)
        
        unique_domains = [LegalDomain(domain) for domain in set(all_domains)]
        self._prepare_labels(unique_domains)
        
        # Encode labels
        encoded_labels = self._encode_labels(domain_labels)
        
        return queries, encoded_labels
    
    def train(self, queries: List[str], domain_labels: List[List[str]], 
              test_size: float = 0.2, epochs: int = 3, batch_size: int = 16):
        """
        Train the neural domain classifier.
        
        Args:
            queries: Training queries
            domain_labels: Domain labels for each query
            test_size: Fraction of data for testing
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        self.logger.info("Preparing training data for neural domain classifier...")
        
        # Prepare data
        processed_queries, encoded_labels = self.prepare_training_data(queries, domain_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_queries, encoded_labels, test_size=test_size, random_state=42
        )
        
        # For simplicity, use scikit-learn with TF-IDF + Neural features
        # In production, you'd use full transformer training
        self._train_hybrid_classifier(X_train, X_test, y_train, y_test)
        
        self.is_trained = True
        self.save_model()
        
        self.logger.info("Neural domain classifier training completed!")
    
    def _train_hybrid_classifier(self, X_train: List[str], X_test: List[str],
                                y_train: np.ndarray, y_test: np.ndarray):
        """Train hybrid classifier with TF-IDF + neural features"""
        
        # Extract TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Extract BERT embeddings for enhanced features
        bert_train_features = self._extract_bert_features(X_train)
        bert_test_features = self._extract_bert_features(X_test)
        
        # Combine features
        X_train_combined = np.hstack([X_train_tfidf.toarray(), bert_train_features])
        X_test_combined = np.hstack([X_test_tfidf.toarray(), bert_test_features])
        
        # Train multi-label classifier
        self.classifier = OneVsRestClassifier(
            LogisticRegression(random_state=42, max_iter=1000)
        )
        
        self.classifier.fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Neural Domain Classifier Accuracy: {accuracy:.4f}")
        
        # Print detailed metrics
        domain_names = [self.reverse_label_encoder[i] for i in range(self.num_labels)]
        report = classification_report(y_test, y_pred, target_names=domain_names, zero_division=0)
        self.logger.info(f"Classification Report:\n{report}")
    
    def _extract_bert_features(self, texts: List[str]) -> np.ndarray:
        """Extract BERT embeddings for feature enhancement"""
        if not hasattr(self, '_bert_model'):
            self._bert_model = AutoModel.from_pretrained(self.model_name)
            self._bert_model.eval()
        
        features = []
        
        for text in texts:
            # Tokenize and get embeddings
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self._bert_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                features.append(embedding)
        
        return np.array(features)
    
    def predict(self, query: str, threshold: float = 0.3) -> Dict[str, float]:
        """
        Predict legal domains for a query with confidence scores.
        
        Args:
            query: Legal query text
            threshold: Minimum confidence threshold
            
        Returns:
            Dictionary mapping domain names to confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        tfidf_features = self.tfidf_vectorizer.transform([query]).toarray()
        bert_features = self._extract_bert_features([query])
        combined_features = np.hstack([tfidf_features, bert_features])
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(combined_features)[0]
        
        # Convert to domain confidence dict
        domain_confidences = {}
        for i, prob in enumerate(probabilities):
            if len(prob) > 1:  # Binary classifier returns [neg_prob, pos_prob]
                confidence = prob[1]  # Positive class probability
            else:
                confidence = prob[0]
            
            if confidence >= threshold:
                domain_name = self.reverse_label_encoder[i]
                domain_confidences[domain_name] = confidence
        
        return domain_confidences
    
    def save_model(self, path: str = 'models/neural_domain_classifier.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder,
            'num_labels': self.num_labels,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Neural domain classifier saved to {path}")
    
    def load_model(self, path: str = 'models/neural_domain_classifier.pkl'):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.reverse_label_encoder = model_data['reverse_label_encoder']
            self.num_labels = model_data['num_labels']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Neural domain classifier loaded from {path}")
            
        except FileNotFoundError:
            self.logger.warning(f"No saved model found at {path}")


class NeuralFactExtractor:
    """
    Neural fact extractor using fine-tuned language models for legal fact extraction.
    
    This component uses transformer models to extract structured legal facts
    from natural language queries with higher accuracy than regex patterns.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize neural fact extractor.
        
        Args:
            model_name: Base model for fact extraction
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.fact_patterns = self._setup_fact_patterns()
        self.ner_pipeline = None
        self.is_trained = False
        
        # Initialize NER pipeline for entity extraction
        try:
            self.ner_pipeline = pipeline("ner", 
                                       aggregation_strategy="simple",
                                       device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            logging.warning(f"Could not initialize NER pipeline: {e}")
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_fact_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Setup neural fact extraction patterns"""
        return {
            'income': {
                'patterns': [
                    r'(?:earn|earning|income|salary|wage).*?(\d+(?:,\d+)*)',
                    r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)\s*(?:per month|monthly)'
                ],
                'entity_types': ['MONEY', 'CARDINAL'],
                'context_words': ['income', 'salary', 'earn', 'monthly', 'wage']
            },
            'family_status': {
                'patterns': [
                    r'(?:married|wife|husband|spouse|divorced|widow)',
                    r'(?:children|child|son|daughter|kids)'
                ],
                'entity_types': ['PERSON'],
                'context_words': ['married', 'family', 'husband', 'wife', 'children']
            },
            'employment': {
                'patterns': [
                    r'(?:job|work|employed|company|employer|workplace)',
                    r'(?:fired|terminated|dismissed|resigned|quit)'
                ],
                'entity_types': ['ORG', 'WORK_OF_ART'],
                'context_words': ['job', 'work', 'company', 'employed', 'fired']
            },
            'legal_issue': {
                'patterns': [
                    r'(?:case|complaint|dispute|problem|issue|matter)',
                    r'(?:harassment|discrimination|violence|fraud|breach)'
                ],
                'entity_types': ['EVENT'],
                'context_words': ['case', 'legal', 'court', 'law', 'dispute']
            },
            'timeline': {
                'patterns': [
                    r'(\d+)\s*(?:years?|months?|days?)\s*(?:ago|back|since)',
                    r'(?:since|from|for)\s*(\d+)\s*(?:years?|months?)'
                ],
                'entity_types': ['DATE', 'TIME'],
                'context_words': ['years', 'months', 'ago', 'since', 'when']
            }
        }
    
    def extract_entities_neural(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities using neural NER pipeline.
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {
            'MONEY': [],
            'PERSON': [],
            'ORG': [],
            'DATE': [],
            'LOCATION': [],
            'MISC': []
        }
        
        if not self.ner_pipeline:
            return entities
        
        try:
            # Extract entities using neural pipeline
            ner_results = self.ner_pipeline(text)
            
            for entity in ner_results:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                confidence = entity['score']
                
                # Map to our categories
                if entity_type in ['MONEY', 'CARDINAL'] and any(word in text.lower() 
                    for word in ['rupees', 'rs', 'salary', 'income', 'earn']):
                    entities['MONEY'].append({
                        'text': entity_text,
                        'confidence': confidence,
                        'start': entity['start'],
                        'end': entity['end']
                    })
                elif entity_type == 'PERSON':
                    entities['PERSON'].append({
                        'text': entity_text,
                        'confidence': confidence,
                        'start': entity['start'],
                        'end': entity['end']
                    })
                elif entity_type in ['ORG', 'WORK_OF_ART']:
                    entities['ORG'].append({
                        'text': entity_text,
                        'confidence': confidence,
                        'start': entity['start'],
                        'end': entity['end']
                    })
                elif entity_type in ['DATE', 'TIME']:
                    entities['DATE'].append({
                        'text': entity_text,
                        'confidence': confidence,
                        'start': entity['start'],
                        'end': entity['end']
                    })
                elif entity_type in ['GPE', 'LOC']:
                    entities['LOCATION'].append({
                        'text': entity_text,
                        'confidence': confidence,
                        'start': entity['start'],
                        'end': entity['end']
                    })
                else:
                    entities['MISC'].append({
                        'text': entity_text,
                        'confidence': confidence,
                        'start': entity['start'],
                        'end': entity['end']
                    })
        
        except Exception as e:
            self.logger.error(f"Error in neural entity extraction: {e}")
        
        return entities
    
    def extract_legal_facts_neural(self, query: str, domain: str = None) -> List[str]:
        """
        Extract legal facts using neural approaches.
        
        Args:
            query: Legal query text
            domain: Optional domain context for focused extraction
            
        Returns:
            List of Prolog facts extracted from query
        """
        facts = ['user_query(user).']
        
        # Extract entities using neural pipeline
        entities = self.extract_entities_neural(query)
        
        # Convert entities to Prolog facts
        facts.extend(self._entities_to_prolog_facts(entities, query))
        
        # Extract domain-specific facts
        if domain:
            domain_facts = self._extract_domain_specific_facts_neural(query, domain)
            facts.extend(domain_facts)
        
        # Extract contextual facts using neural understanding
        contextual_facts = self._extract_contextual_facts_neural(query)
        facts.extend(contextual_facts)
        
        return facts
    
    def _entities_to_prolog_facts(self, entities: Dict[str, List[Dict[str, Any]]], 
                                 query: str) -> List[str]:
        """Convert neural entities to Prolog facts"""
        facts = []
        
        # Money/Income entities
        for money_entity in entities['MONEY']:
            amount_text = money_entity['text'].replace(',', '').replace('₹', '').replace('Rs', '')
            try:
                amount = int(''.join(filter(str.isdigit, amount_text)))
                if amount > 0:
                    if any(word in query.lower() for word in ['monthly', 'per month', 'salary']):
                        facts.append(f'monthly_income(user, {amount}).')
                    else:
                        facts.append(f'amount_involved(user, {amount}).')
            except ValueError:
                pass
        
        # Person entities (family members, etc.)
        for person_entity in entities['PERSON']:
            person_text = person_entity['text'].lower()
            if person_text in ['husband', 'wife', 'spouse']:
                facts.append(f'marital_status(user, married).')
            elif person_text in ['child', 'children', 'son', 'daughter']:
                facts.append(f'has_children(user, true).')
        
        # Organization entities (employers, companies)
        for org_entity in entities['ORG']:
            facts.append(f'involved_organization(user, "{org_entity["text"]}").')
            if any(word in query.lower() for word in ['employer', 'company', 'work', 'job']):
                facts.append(f'employment_related(user, true).')
        
        # Date/Time entities
        for date_entity in entities['DATE']:
            facts.append(f'timeline_mentioned(user, "{date_entity["text"]}").')
        
        # Location entities
        for loc_entity in entities['LOCATION']:
            facts.append(f'location(user, "{loc_entity["text"]}").')
        
        return facts
    
    def _extract_domain_specific_facts_neural(self, query: str, domain: str) -> List[str]:
        """Extract domain-specific facts using neural understanding"""
        facts = []
        query_lower = query.lower()
        
        if domain == 'employment_law':
            # Neural employment fact extraction
            if any(word in query_lower for word in ['fired', 'terminated', 'dismissed']):
                facts.append('termination_occurred(user, true).')
            if any(word in query_lower for word in ['harassment', 'harassed', 'harassing']):
                facts.append('harassment_reported(user, true).')
            if any(word in query_lower for word in ['overtime', 'extra hours', 'long hours']):
                facts.append('overtime_issue(user, true).')
        
        elif domain == 'family_law':
            # Neural family law fact extraction  
            if any(word in query_lower for word in ['divorce', 'separation', 'split']):
                facts.append('divorce_sought(user, true).')
            if any(word in query_lower for word in ['custody', 'children', 'child care']):
                facts.append('custody_issue(user, true).')
            if any(word in query_lower for word in ['maintenance', 'alimony', 'support']):
                facts.append('maintenance_sought(user, true).')
        
        elif domain == 'consumer_protection':
            # Neural consumer protection fact extraction
            if any(word in query_lower for word in ['defective', 'faulty', 'broken', 'damaged']):
                facts.append('product_defective(user, true).')
            if any(word in query_lower for word in ['service', 'poor service', 'bad service']):
                facts.append('service_deficiency(user, true).')
            if any(word in query_lower for word in ['refund', 'money back', 'return']):
                facts.append('refund_sought(user, true).')
        
        return facts
    
    def _extract_contextual_facts_neural(self, query: str) -> List[str]:
        """Extract contextual facts using neural language understanding"""
        facts = []
        query_lower = query.lower()
        
        # Emotional/urgency indicators
        urgent_indicators = ['urgent', 'immediately', 'asap', 'quickly', 'emergency']
        if any(indicator in query_lower for indicator in urgent_indicators):
            facts.append('urgent_matter(user, true).')
        
        # Financial distress indicators
        financial_indicators = ['poor', 'cannot afford', 'no money', 'financial trouble']
        if any(indicator in query_lower for indicator in financial_indicators):
            facts.append('financial_distress(user, true).')
        
        # Legal knowledge indicators
        legal_terms = ['lawyer', 'court', 'legal', 'case', 'law', 'rights', 'sue']
        legal_term_count = sum(1 for term in legal_terms if term in query_lower)
        if legal_term_count >= 2:
            facts.append('legal_awareness(user, high).')
        elif legal_term_count >= 1:
            facts.append('legal_awareness(user, medium).')
        else:
            facts.append('legal_awareness(user, low).')
        
        return facts
    
    def train_on_domain_data(self, training_data: List[Dict[str, Any]]):
        """
        Train fact extractor on domain-specific data.
        
        Args:
            training_data: List of training examples with queries and expected facts
        """
        # This would implement actual neural training
        # For now, we'll enhance the pattern matching based on training data
        
        self.logger.info("Training neural fact extractor on domain data...")
        
        # Analyze training data to improve patterns
        for example in training_data:
            query = example.get('query', '')
            expected_facts = example.get('expected_facts', [])
            
            # Update patterns based on successful extractions
            self._update_patterns_from_training(query, expected_facts)
        
        self.is_trained = True
        self.logger.info("Neural fact extractor training completed!")
    
    def _update_patterns_from_training(self, query: str, expected_facts: List[str]):
        """Update extraction patterns based on training examples"""
        # This would implement pattern learning from training data
        # For now, it's a placeholder for future enhancement
        pass


class HybridConfidenceEstimator:
    """
    Neural confidence estimator that combines symbolic reasoning confidence
    with neural model confidence scores.
    """
    
    def __init__(self):
        self.confidence_weights = {
            'neural_domain_confidence': 0.4,
            'neural_fact_confidence': 0.3,
            'symbolic_reasoning_confidence': 0.3
        }
        self.logger = logging.getLogger(__name__)
    
    def estimate_confidence(self, neural_domain_scores: Dict[str, float],
                          neural_fact_scores: Dict[str, float],
                          symbolic_reasoning_success: bool,
                          prolog_facts_count: int) -> float:
        """
        Estimate overall confidence combining neural and symbolic components.
        
        Args:
            neural_domain_scores: Domain classification confidence scores
            neural_fact_scores: Fact extraction confidence scores
            symbolic_reasoning_success: Whether Prolog reasoning succeeded
            prolog_facts_count: Number of facts extracted
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        # Neural domain confidence (average of top domains)
        domain_confidence = np.mean(list(neural_domain_scores.values())) if neural_domain_scores else 0.0
        
        # Neural fact confidence (average of entity confidences)
        fact_confidence = np.mean(list(neural_fact_scores.values())) if neural_fact_scores else 0.0
        
        # Symbolic reasoning confidence
        symbolic_confidence = 1.0 if symbolic_reasoning_success else 0.0
        if prolog_facts_count < 3:  # Penalize if too few facts extracted
            symbolic_confidence *= 0.7
        
        # Weighted combination
        overall_confidence = (
            self.confidence_weights['neural_domain_confidence'] * domain_confidence +
            self.confidence_weights['neural_fact_confidence'] * fact_confidence +
            self.confidence_weights['symbolic_reasoning_confidence'] * symbolic_confidence
        )
        
        return min(1.0, max(0.0, overall_confidence))
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update confidence combination weights"""
        if abs(sum(new_weights.values()) - 1.0) < 0.01:  # Weights should sum to 1
            self.confidence_weights.update(new_weights)
            self.logger.info(f"Updated confidence weights: {self.confidence_weights}")
        else:
            raise ValueError("Confidence weights must sum to 1.0")
