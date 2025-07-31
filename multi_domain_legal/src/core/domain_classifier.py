"""
Domain Classification System for Multi-Domain Legal AI.

This module implements a sophisticated domain classifier that can identify
which legal domain(s) a query belongs to, supporting both single-domain
and multi-domain queries.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
from pathlib import Path

from .domain_registry import LegalDomain, LegalDomainRegistry

class DomainClassifier:
    """
    Multi-label classifier for identifying legal domains in queries.
    
    Uses TF-IDF features with logistic regression to classify queries
    into one or more legal domains. Supports both rule-based and ML-based
    classification approaches.
    """
    
    def __init__(self):
        self.registry = LegalDomainRegistry()
        self.pipeline = None
        self.label_encoder = {domain: idx for idx, domain in enumerate(LegalDomain)}
        self.label_decoder = {idx: domain for domain, idx in self.label_encoder.items()}
        self.is_trained = False
        
        # Rule-based patterns for high-confidence classification
        self.domain_patterns = self._create_domain_patterns()
    
    def _create_domain_patterns(self) -> Dict[LegalDomain, List[str]]:
        """Create regex patterns for rule-based classification"""
        return {
            LegalDomain.LEGAL_AID: [
                r'\b(legal aid|free lawyer|government lawyer|cannot afford lawyer)\b',
                r'\b(poor|poverty|below poverty line|bpl)\b',
                r'\b(sc|st|scheduled caste|scheduled tribe)\b',
                r'\b(disability|disabled|handicapped)\b',
                r'\b(juvenile|minor|child protection)\b'
            ],
            LegalDomain.FAMILY_LAW: [
                r'\b(divorce|separation|matrimonial dispute)\b',
                r'\b(husband|wife|spouse|marriage)\b', 
                r'\b(dowry|domestic violence|cruelty)\b',
                r'\b(maintenance|alimony|child custody)\b',
                r'\b(inheritance|succession|property dispute)\b',
                r'\b(nikah|talaq|hindu marriage|christian marriage)\b'
            ],
            LegalDomain.CONSUMER_PROTECTION: [
                r'\b(consumer court|consumer complaint|defective product)\b',
                r'\b(warranty|guarantee|refund|replacement)\b',
                r'\b(food poisoning|restaurant|hotel)\b',
                r'\b(builder|real estate|apartment|possession delay)\b',
                r'\b(online shopping|e-commerce|cyber fraud)\b'
            ],
            LegalDomain.FUNDAMENTAL_RIGHTS: [
                r'\b(fundamental rights|constitutional rights|discrimination)\b',
                r'\b(rti|right to information|government information)\b',
                r'\b(police harassment|illegal detention|arrest)\b',
                r'\b(freedom of speech|freedom of religion)\b',
                r'\b(human rights|rights violation)\b'
            ],
            LegalDomain.EMPLOYMENT_LAW: [
                r'\b(termination|firing|dismissal|wrongful termination)\b',
                r'\b(sexual harassment|workplace harassment|posh)\b',
                r'\b(minimum wage|salary|overtime|bonus)\b',
                r'\b(maternity leave|medical leave|pf|provident fund)\b',
                r'\b(labor dispute|strike|union|industrial dispute)\b'
            ]
        }
    
    def classify_rule_based(self, query: str) -> List[Tuple[LegalDomain, float]]:
        """
        Rule-based classification using regex patterns.
        
        Args:
            query: Legal query text
            
        Returns:
            List of (domain, confidence) tuples
        """
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
                    score += 1.0
            
            if matches > 0:
                # Normalize by number of patterns
                domain_scores[domain] = score / len(patterns)
        
        # Also check keyword-based scoring
        for domain in LegalDomain:
            keywords = self.registry.get_domain_keywords(domain)
            keyword_matches = sum(1 for kw in keywords if kw.lower() in query_lower)
            
            if keyword_matches > 0:
                keyword_score = keyword_matches / len(keywords)
                if domain in domain_scores:
                    domain_scores[domain] = max(domain_scores[domain], keyword_score)
                else:
                    domain_scores[domain] = keyword_score
        
        # Sort by confidence and return top domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter domains with confidence > 0.1
        return [(domain, conf) for domain, conf in sorted_domains if conf > 0.1]
    
    def prepare_training_data(self) -> Tuple[List[str], np.ndarray]:
        """
        Prepare synthetic training data for ML classifier.
        
        Returns:
            Tuple of (queries, labels) where labels is a multi-hot encoded array
        """
        training_queries = []
        training_labels = []
        
        # Generate training examples for each domain
        domain_examples = {
            LegalDomain.LEGAL_AID: [
                "I need legal aid as I cannot afford a lawyer",
                "I am poor and need free legal assistance", 
                "Being SC/ST, can I get government lawyer",
                "I am disabled and need legal help",
                "My child needs legal protection"
            ],
            LegalDomain.FAMILY_LAW: [
                "I want to divorce my husband for cruelty",
                "My wife is demanding dowry from my family",
                "I need maintenance from my ex-husband",
                "Child custody dispute after divorce",
                "Property inheritance rights of daughters",
                "Muslim marriage and nikah procedures"
            ],
            LegalDomain.CONSUMER_PROTECTION: [
                "My mobile phone is defective, shop refusing refund",
                "Builder delayed apartment possession by 2 years",
                "Food poisoning from restaurant, want compensation",
                "Online shopping fraud, product not delivered",
                "Warranty claim rejected by manufacturer"
            ],
            LegalDomain.FUNDAMENTAL_RIGHTS: [
                "Police arrested me without warrant",
                "Government office discriminating based on caste",
                "RTI application rejected illegally",
                "Freedom of speech violation by authorities",
                "Human rights violation by police"
            ],
            LegalDomain.EMPLOYMENT_LAW: [
                "Wrongful termination from job without notice",
                "Sexual harassment at workplace by boss",
                "Not getting minimum wage as per law",
                "Denied maternity leave by company",
                "Gender discrimination in salary"
            ]
        }
        
        # Create training data
        for domain, examples in domain_examples.items():
            for example in examples:
                training_queries.append(example)
                
                # Create multi-hot label
                label = np.zeros(len(LegalDomain))
                label[self.label_encoder[domain]] = 1
                training_labels.append(label)
        
        # Add some multi-domain examples
        multi_domain_examples = [
            ("I am a poor woman facing domestic violence and need legal aid", 
             [LegalDomain.LEGAL_AID, LegalDomain.FAMILY_LAW]),
            ("My employer fired me for being SC/ST and I need legal help",
             [LegalDomain.EMPLOYMENT_LAW, LegalDomain.FUNDAMENTAL_RIGHTS, LegalDomain.LEGAL_AID]),
            ("Builder cheated me and I cannot afford lawyer",
             [LegalDomain.CONSUMER_PROTECTION, LegalDomain.LEGAL_AID]),
            ("Police harassment at workplace and need legal aid",
             [LegalDomain.FUNDAMENTAL_RIGHTS, LegalDomain.EMPLOYMENT_LAW, LegalDomain.LEGAL_AID])
        ]
        
        for query, domains in multi_domain_examples:
            training_queries.append(query)
            label = np.zeros(len(LegalDomain))
            for domain in domains:
                label[self.label_encoder[domain]] = 1
            training_labels.append(label)
        
        return training_queries, np.array(training_labels)
    
    def train(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the ML-based domain classifier.
        
        Args:
            save_path: Optional path to save the trained model
            
        Returns:
            Training metrics and information
        """
        print("🔧 Preparing training data...")
        X_train, y_train = self.prepare_training_data()
        
        print(f"📊 Training on {len(X_train)} examples across {len(LegalDomain)} domains...")
        
        # Create pipeline with TF-IDF and multi-label classifier
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )),
            ('classifier', OneVsRestClassifier(
                LogisticRegression(random_state=42, max_iter=1000)
            ))
        ])
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.pipeline.predict(X_train)
        accuracy = np.mean(np.all(y_pred == y_train, axis=1))
        
        print(f"✅ Training completed! Accuracy: {accuracy:.1%}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
            print(f"💾 Model saved to: {save_path}")
        
        return {
            'accuracy': accuracy,
            'num_examples': len(X_train),
            'num_domains': len(LegalDomain)
        }
    
    def classify_ml(self, query: str) -> List[Tuple[LegalDomain, float]]:
        """
        ML-based classification using trained model.
        
        Args:
            query: Legal query text
            
        Returns:
            List of (domain, confidence) tuples
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get prediction probabilities
        proba = self.pipeline.predict_proba([query])[0]
        
        # Convert to domain-confidence pairs
        domain_scores = []
        for idx, prob in enumerate(proba):
            if prob > 0.3:  # Threshold for considering a domain
                domain = self.label_decoder[idx]
                domain_scores.append((domain, prob))
        
        # Sort by confidence
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        
        return domain_scores
    
    def classify(self, query: str, method: str = "hybrid") -> List[Tuple[LegalDomain, float]]:
        """
        Classify query into legal domains.
        
        Args:
            query: Legal query text
            method: Classification method ("rule", "ml", or "hybrid")
            
        Returns:
            List of (domain, confidence) tuples sorted by confidence
        """
        if method == "rule":
            return self.classify_rule_based(query)
        elif method == "ml":
            return self.classify_ml(query)
        elif method == "hybrid":
            # Combine rule-based and ML-based approaches
            rule_results = self.classify_rule_based(query)
            
            if self.is_trained:
                try:
                    ml_results = self.classify_ml(query)
                    
                    # Combine results
                    combined_scores = {}
                    
                    # Add rule-based scores
                    for domain, score in rule_results:
                        combined_scores[domain] = score * 0.4  # 40% weight
                    
                    # Add ML scores
                    for domain, score in ml_results:
                        if domain in combined_scores:
                            combined_scores[domain] += score * 0.6  # 60% weight
                        else:
                            combined_scores[domain] = score * 0.6
                    
                    # Sort and return
                    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                    return [(domain, score) for domain, score in sorted_results if score > 0.2]
                    
                except Exception:
                    # Fallback to rule-based if ML fails
                    return rule_results
            else:
                return rule_results
        else:
            raise ValueError(f"Unknown classification method: {method}")
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        save_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'is_trained': self.is_trained
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.pipeline = save_data['pipeline']
        self.label_encoder = save_data['label_encoder']
        self.label_decoder = save_data['label_decoder']
        self.is_trained = save_data['is_trained']
