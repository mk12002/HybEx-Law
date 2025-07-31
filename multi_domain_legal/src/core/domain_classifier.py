"""
Domain Classification System for Multi-Domain Legal AI.

This module implements a sophisticated hybrid domain classifier that combines
neural and symbolic approaches to identify which legal domain(s) a query belongs to.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pickle
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .domain_registry import LegalDomain, LegalDomainRegistry

# Import neural components if available
try:
    from .neural_components import NeuralDomainClassifier, HybridConfidenceEstimator
    NEURAL_COMPONENTS_AVAILABLE = True
except ImportError:
    NEURAL_COMPONENTS_AVAILABLE = False

class HybridDomainClassifier:
    """
    Hybrid multi-label classifier for identifying legal domains in queries.
    
    Combines neural domain classification with rule-based approaches and
    symbolic reasoning for improved accuracy and explainability.
    """
    
    def __init__(self, use_neural: bool = True):
        """
        Initialize hybrid domain classifier.
        
        Args:
            use_neural: Whether to use neural components if available
        """
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
    
    def predict_hybrid(self, query: str, confidence_threshold: float = 0.3) -> Dict[str, float]:
        """
        Hybrid prediction combining neural and rule-based approaches.
        
        Args:
            query: Legal query text
            confidence_threshold: Minimum confidence for domain inclusion
            
        Returns:
            Dictionary mapping domain names to confidence scores
        """
        domain_confidences = {}
        
        # 1. Rule-based classification (always available)
        rule_based_results = self.classify_rule_based(query)
        for domain, confidence in rule_based_results:
            domain_confidences[domain.value] = confidence
        
        # 2. Neural classification (if available)
        neural_confidences = {}
        if self.use_neural and hasattr(self, 'neural_classifier') and self.neural_classifier:
            try:
                neural_confidences = self.neural_classifier.predict(query, confidence_threshold)
            except Exception as e:
                print(f"Warning: Neural classification failed: {e}")
        
        # 3. ML-based classification (if trained)
        ml_confidences = {}
        if self.is_trained:
            try:
                ml_results = self.predict(query)
                for domain, confidence in ml_results:
                    ml_confidences[domain.value] = confidence
            except Exception as e:
                print(f"Warning: ML classification failed: {e}")
        
        # 4. Combine all approaches using weighted average
        all_domains = set()
        all_domains.update(domain_confidences.keys())
        all_domains.update(neural_confidences.keys())
        all_domains.update(ml_confidences.keys())
        
        final_confidences = {}
        for domain in all_domains:
            scores = []
            weights = []
            
            # Rule-based score
            if domain in domain_confidences:
                scores.append(domain_confidences[domain])
                weights.append(0.4)  # Rule-based gets 40% weight
            
            # Neural score
            if domain in neural_confidences:
                scores.append(neural_confidences[domain])
                weights.append(0.4)  # Neural gets 40% weight
            
            # ML score
            if domain in ml_confidences:
                scores.append(ml_confidences[domain])
                weights.append(0.2)  # ML gets 20% weight
            
            if scores:
                # Weighted average
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                
                # Boost confidence if multiple approaches agree
                agreement_bonus = (len(scores) - 1) * 0.1
                final_score = min(1.0, weighted_score + agreement_bonus)
                
                if final_score >= confidence_threshold:
                    final_confidences[domain] = final_score
        
        return final_confidences
    
    def set_classification_mode(self, mode: str):
        """
        Set the classification mode.
        
        Args:
            mode: One of 'neural', 'rule_based', 'hybrid'
        """
        if mode not in self.classification_modes:
            raise ValueError(f"Mode must be one of {self.classification_modes}")
        
        if mode == 'neural' and not self.use_neural:
            raise ValueError("Neural mode not available - neural components not initialized")
        
        self.current_mode = mode
    
    def get_classification_explanation(self, query: str) -> Dict[str, Any]:
        """
        Get detailed explanation of classification decision.
        
        Args:
            query: Legal query text
            
        Returns:
            Dictionary with classification explanations
        """
        explanation = {
            'query': query,
            'rule_based_results': [],
            'neural_results': {},
            'ml_results': [],
            'final_decision': {},
            'reasoning': []
        }
        
        # Rule-based explanation
        rule_results = self.classify_rule_based(query)
        explanation['rule_based_results'] = [(domain.value, conf) for domain, conf in rule_results]
        
        if rule_results:
            explanation['reasoning'].append("Rule-based classification found pattern matches")
        
        # Neural explanation (if available)
        if self.use_neural and hasattr(self, 'neural_classifier') and self.neural_classifier:
            try:
                neural_results = self.neural_classifier.predict(query, 0.1)
                explanation['neural_results'] = neural_results
                if neural_results:
                    explanation['reasoning'].append("Neural classifier detected domain indicators")
            except Exception as e:
                explanation['reasoning'].append(f"Neural classification failed: {e}")
        
        # ML explanation (if trained)
        if self.is_trained:
            try:
                ml_results = self.predict(query)
                explanation['ml_results'] = [(domain.value, conf) for domain, conf in ml_results]
                if ml_results:
                    explanation['reasoning'].append("Traditional ML classifier provided additional confidence")
            except Exception as e:
                explanation['reasoning'].append(f"ML classification failed: {e}")
        
        # Final hybrid decision
        explanation['final_decision'] = self.predict_hybrid(query)
        
        return explanation

# For backward compatibility, create an alias
DomainClassifier = HybridDomainClassifier
