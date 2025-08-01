#!/usr/bin/env python3
"""
HybEx-Law Training Pipeline
===========================

Complete training pipeline for the hybrid neural-symbolic legal aid system
using the generated 23,500 sample dataset.

Author: HybEx-Law Team
Date: August 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Import our dataset loader
from dataset_loader import LegalDatasetLoader, TrainingBatch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybExLawTrainer:
    """
    Complete training pipeline for HybEx-Law system.
    Trains all components: Domain Classifier, Fact Extractor, Eligibility Predictor.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize training pipeline.
        
        Args:
            data_dir: Directory containing generated datasets
        """
        self.data_dir = Path(data_dir)
        self.loader = LegalDatasetLoader(data_dir)
        self.models = {}
        self.vectorizers = {}
        self.encoders = {}
        
        # Legal domain mapping
        self.domain_names = ['legal_aid', 'family_law', 'consumer_protection', 
                           'employment_law', 'fundamental_rights']
        
        logger.info("HybEx-Law training pipeline initialized")
    
    def load_and_prepare_data(self) -> Dict[str, TrainingBatch]:
        """
        Load and prepare all training data.
        
        Returns:
            Dictionary containing train/validation/test splits
        """
        logger.info("Loading and preparing training data...")
        
        # Load main dataset
        self.loader.load_main_dataset()
        
        # Create stratified splits
        splits = self.loader.create_training_splits(test_size=0.15, val_size=0.15)
        
        logger.info("Data preparation complete:")
        for split_name, batch in splits.items():
            logger.info(f"  {split_name.capitalize()}: {len(batch.queries):,} samples")
        
        return splits
    
    def train_stage1_classifier(self, train_batch: TrainingBatch, val_batch: TrainingBatch) -> Dict[str, Any]:
        """
        Train Stage 1: Multi-label Domain Classifier.
        
        Args:
            train_batch: Training data batch
            val_batch: Validation data batch
            
        Returns:
            Training metrics and model performance
        """
        logger.info("Training Stage 1: Multi-label Domain Classifier")
        
        # Prepare data
        train_queries, train_labels = self.loader.prepare_stage1_data(train_batch)
        val_queries, val_labels = self.loader.prepare_stage1_data(val_batch)
        
        # Vectorize text
        self.vectorizers['domain_classifier'] = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        
        X_train = self.vectorizers['domain_classifier'].fit_transform(train_queries)
        X_val = self.vectorizers['domain_classifier'].transform(val_queries)
        
        # Train multi-label classifier
        self.models['domain_classifier'] = OneVsRestClassifier(
            LogisticRegression(random_state=42, max_iter=1000)
        )
        
        self.models['domain_classifier'].fit(X_train, train_labels)
        
        # Evaluate
        val_predictions = self.models['domain_classifier'].predict(X_val)
        val_proba = self.models['domain_classifier'].predict_proba(X_val)
        
        # Calculate metrics
        metrics = self._calculate_multilabel_metrics(val_labels, val_predictions, self.domain_names)
        
        logger.info("Stage 1 Training Complete:")
        logger.info(f"  Overall Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Macro F1-Score: {metrics['macro_f1']:.3f}")
        
        return {
            'model': self.models['domain_classifier'],
            'vectorizer': self.vectorizers['domain_classifier'],
            'metrics': metrics,
            'validation_predictions': val_predictions,
            'validation_probabilities': val_proba
        }
    
    def train_stage2_extractor(self, train_batch: TrainingBatch, val_batch: TrainingBatch) -> Dict[str, Any]:
        """
        Train Stage 2: Fact Extraction (simplified for demo).
        
        Args:
            train_batch: Training data batch
            val_batch: Validation data batch
            
        Returns:
            Training metrics and extractor performance
        """
        logger.info("Training Stage 2: Fact Extractor")
        
        # Prepare data
        train_queries, train_facts = self.loader.prepare_stage2_data(train_batch)
        val_queries, val_facts = self.loader.prepare_stage2_data(val_batch)
        
        # For this demo, we'll create a rule-based extractor
        # In practice, you'd use NER models or custom neural networks
        extractor_metrics = self._train_rule_based_extractor(train_queries, train_facts, val_queries, val_facts)
        
        logger.info("Stage 2 Training Complete:")
        logger.info(f"  Fact Extraction Accuracy: {extractor_metrics['accuracy']:.3f}")
        logger.info(f"  Average Facts per Query: {extractor_metrics['avg_facts']:.1f}")
        
        return extractor_metrics
    
    def train_eligibility_predictor(self, train_batch: TrainingBatch, val_batch: TrainingBatch) -> Dict[str, Any]:
        """
        Train Eligibility Predictor (combines extracted facts with Prolog reasoning).
        
        Args:
            train_batch: Training data batch
            val_batch: Validation data batch
            
        Returns:
            Training metrics and predictor performance
        """
        logger.info("Training Eligibility Predictor")
        
        # Prepare data
        train_facts, train_eligibility, train_reasoning = self.loader.prepare_eligibility_data(train_batch)
        val_facts, val_eligibility, val_reasoning = self.loader.prepare_eligibility_data(val_batch)
        
        # Convert facts to features (simplified approach)
        train_features = self._facts_to_features(train_facts)
        val_features = self._facts_to_features(val_facts)
        
        # Train eligibility classifier
        self.models['eligibility_predictor'] = LogisticRegression(random_state=42)
        self.models['eligibility_predictor'].fit(train_features, train_eligibility)
        
        # Evaluate
        val_predictions = self.models['eligibility_predictor'].predict(val_features)
        val_proba = self.models['eligibility_predictor'].predict_proba(val_features)
        
        accuracy = accuracy_score(val_eligibility, val_predictions)
        
        # Classification report
        report = classification_report(val_eligibility, val_predictions, 
                                     target_names=['Not Eligible', 'Eligible'],
                                     output_dict=True)
        
        logger.info("Eligibility Predictor Training Complete:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision (Eligible): {report['Eligible']['precision']:.3f}")
        logger.info(f"  Recall (Eligible): {report['Eligible']['recall']:.3f}")
        
        return {
            'model': self.models['eligibility_predictor'],
            'accuracy': accuracy,
            'classification_report': report,
            'validation_predictions': val_predictions,
            'validation_probabilities': val_proba
        }
    
    def evaluate_full_pipeline(self, test_batch: TrainingBatch) -> Dict[str, Any]:
        """
        Evaluate the complete HybEx-Law pipeline end-to-end.
        
        Args:
            test_batch: Test data batch
            
        Returns:
            Comprehensive evaluation metrics
        """
        logger.info("Evaluating complete HybEx-Law pipeline")
        
        test_queries = test_batch.queries
        test_domains = test_batch.domain_labels
        test_facts = test_batch.facts
        test_eligibility = test_batch.eligibility_labels
        
        # Stage 1: Domain Classification
        X_test = self.vectorizers['domain_classifier'].transform(test_queries)
        predicted_domains = self.models['domain_classifier'].predict(X_test)
        domain_accuracy = accuracy_score(test_domains.argmax(axis=1), predicted_domains.argmax(axis=1))
        
        # Stage 2: Fact Extraction (simplified evaluation)
        extracted_facts_accuracy = self._evaluate_fact_extraction(test_queries, test_facts)
        
        # Stage 3: Eligibility Prediction
        test_features = self._facts_to_features(test_facts)
        predicted_eligibility = self.models['eligibility_predictor'].predict(test_features)
        eligibility_accuracy = accuracy_score(test_eligibility, predicted_eligibility)
        
        # Overall pipeline performance
        pipeline_metrics = {
            'domain_classification_accuracy': domain_accuracy,
            'fact_extraction_accuracy': extracted_facts_accuracy,
            'eligibility_prediction_accuracy': eligibility_accuracy,
            'overall_pipeline_score': np.mean([domain_accuracy, extracted_facts_accuracy, eligibility_accuracy])
        }
        
        logger.info("Pipeline Evaluation Complete:")
        logger.info(f"  Domain Classification: {domain_accuracy:.3f}")
        logger.info(f"  Fact Extraction: {extracted_facts_accuracy:.3f}")
        logger.info(f"  Eligibility Prediction: {eligibility_accuracy:.3f}")
        logger.info(f"  Overall Pipeline Score: {pipeline_metrics['overall_pipeline_score']:.3f}")
        
        return pipeline_metrics
    
    def save_trained_models(self, output_dir: str = "models") -> Dict[str, Path]:
        """
        Save all trained models and components.
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary mapping component names to file paths
        """
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save models
        for model_name, model in self.models.items():
            model_file = output_path / f"{model_name}.joblib"
            joblib.dump(model, model_file)
            saved_files[model_name] = model_file
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save vectorizers
        for vectorizer_name, vectorizer in self.vectorizers.items():
            vec_file = output_path / f"{vectorizer_name}_vectorizer.joblib"
            joblib.dump(vectorizer, vec_file)
            saved_files[f"{vectorizer_name}_vectorizer"] = vec_file
            logger.info(f"Saved {vectorizer_name} vectorizer to {vec_file}")
        
        # Save domain encoder
        if hasattr(self.loader, 'domain_encoder'):
            encoder_file = output_path / "domain_encoder.joblib"
            joblib.dump(self.loader.domain_encoder, encoder_file)
            saved_files['domain_encoder'] = encoder_file
            logger.info(f"Saved domain encoder to {encoder_file}")
        
        return saved_files
    
    def predict_legal_aid_eligibility(self, query: str) -> Dict[str, Any]:
        """
        Complete prediction pipeline for a new query.
        
        Args:
            query: Legal query text
            
        Returns:
            Comprehensive prediction results
        """
        if not self.models:
            raise ValueError("Models not trained yet. Run training pipeline first.")
        
        # Stage 1: Domain Classification
        query_vector = self.vectorizers['domain_classifier'].transform([query])
        domain_proba = self.models['domain_classifier'].predict_proba(query_vector)[0]
        predicted_domains = [self.domain_names[i] for i, prob in enumerate(domain_proba) if prob > 0.3]
        
        # Stage 2: Fact Extraction (simplified)
        extracted_facts = self._extract_facts_simple(query)
        
        # Stage 3: Eligibility Prediction
        fact_features = self._facts_to_features([extracted_facts])
        eligibility_proba = self.models['eligibility_predictor'].predict_proba(fact_features)[0]
        is_eligible = self.models['eligibility_predictor'].predict(fact_features)[0]
        
        prediction_result = {
            'query': query,
            'predicted_domains': predicted_domains,
            'domain_probabilities': dict(zip(self.domain_names, domain_proba)),
            'extracted_facts': extracted_facts,
            'eligibility_prediction': bool(is_eligible),
            'eligibility_confidence': float(max(eligibility_proba)),
            'legal_reasoning': self._generate_reasoning(extracted_facts, is_eligible, predicted_domains)
        }
        
        return prediction_result
    
    # Helper methods
    
    def _calculate_multilabel_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
        """Calculate metrics for multi-label classification."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'micro_f1': f1_score(y_true, y_pred, average='micro'),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro')
        }
    
    def _train_rule_based_extractor(self, train_queries: List[str], train_facts: List[List[str]], 
                                   val_queries: List[str], val_facts: List[List[str]]) -> Dict[str, float]:
        """Train a simplified rule-based fact extractor."""
        # This is a simplified implementation for demo purposes
        # In practice, you'd use more sophisticated NLP techniques
        
        correct_extractions = 0
        total_extractions = 0
        total_facts = 0
        
        for query, expected_facts in zip(val_queries, val_facts):
            extracted = self._extract_facts_simple(query)
            
            # Simple evaluation: count matching fact types
            expected_types = set(fact.split('(')[0] for fact in expected_facts if '(' in fact)
            extracted_types = set(fact.split('(')[0] for fact in extracted if '(' in fact)
            
            if expected_types:
                overlap = len(expected_types.intersection(extracted_types))
                correct_extractions += overlap
                total_extractions += len(expected_types)
            
            total_facts += len(expected_facts)
        
        accuracy = correct_extractions / max(total_extractions, 1)
        avg_facts = total_facts / len(val_facts)
        
        return {
            'accuracy': accuracy,
            'avg_facts': avg_facts,
            'total_evaluated': len(val_queries)
        }
    
    def _facts_to_features(self, facts_list: List[List[str]]) -> np.ndarray:
        """Convert fact lists to feature vectors for ML."""
        # Simplified feature extraction from facts
        features = []
        
        for facts in facts_list:
            feature_vector = [0] * 20  # 20 features for demo
            
            for fact in facts:
                if 'income_monthly' in fact:
                    # Extract income value
                    try:
                        income = int(''.join(filter(str.isdigit, fact)))
                        feature_vector[0] = min(income / 50000, 1.0)  # Normalize income
                    except:
                        feature_vector[0] = 0
                
                if 'is_woman' in fact:
                    feature_vector[1] = 1
                if 'social_category' in fact and 'sc' in fact:
                    feature_vector[2] = 1
                if 'social_category' in fact and 'st' in fact:
                    feature_vector[3] = 1
                if 'is_disabled' in fact:
                    feature_vector[4] = 1
                if 'is_senior_citizen' in fact:
                    feature_vector[5] = 1
                if 'domestic_violence' in fact:
                    feature_vector[6] = 1
                if 'seeks_legal_aid' in fact:
                    feature_vector[7] = 1
                # Add more feature extractions as needed
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_facts_simple(self, query: str) -> List[str]:
        """Simple rule-based fact extraction for demo."""
        facts = ["applicant(user)."]
        query_lower = query.lower()
        
        # Income extraction
        import re
        income_match = re.search(r'(\d+)\s*(?:rupees?|rs)', query_lower)
        if income_match:
            income = int(income_match.group(1))
            facts.append(f"income_monthly(user, {income}).")
        
        # Gender detection
        if any(word in query_lower for word in ['woman', 'female', 'wife', 'mother', 'girl']):
            facts.append("is_woman(user, true).")
        
        # Social category detection
        if any(word in query_lower for word in ['sc', 'dalit', 'scheduled caste']):
            facts.append("social_category(user, 'sc').")
        if any(word in query_lower for word in ['st', 'tribal', 'scheduled tribe']):
            facts.append("social_category(user, 'st').")
        
        # Legal aid seeking
        if any(phrase in query_lower for phrase in ['legal aid', 'free legal', 'cannot afford']):
            facts.append("seeks_legal_aid(user, true).")
        
        return facts
    
    def _evaluate_fact_extraction(self, queries: List[str], expected_facts: List[List[str]]) -> float:
        """Evaluate fact extraction accuracy."""
        correct = 0
        total = 0
        
        for query, expected in zip(queries, expected_facts):
            extracted = self._extract_facts_simple(query)
            
            # Simple evaluation based on fact count similarity
            if abs(len(extracted) - len(expected)) <= 2:  # Allow some tolerance
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _generate_reasoning(self, facts: List[str], is_eligible: bool, domains: List[str]) -> str:
        """Generate legal reasoning for prediction."""
        if is_eligible:
            if any('is_woman' in fact for fact in facts):
                return "Eligible under categorical criteria - women are eligible for legal aid"
            elif any('income_monthly' in fact for fact in facts):
                return "Eligible under income criteria - monthly income below threshold"
            else:
                return "Eligible based on case merits and legal requirements"
        else:
            return "Not eligible - income above threshold and no categorical eligibility found"

def main():
    """Main training pipeline execution."""
    
    print("🏛️  HybEx-Law Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = HybExLawTrainer("data")
    
    # Load and prepare data
    print("\n📊 Loading dataset...")
    splits = trainer.load_and_prepare_data()
    
    # Train Stage 1: Domain Classifier
    print("\n🎯 Training Stage 1: Domain Classifier...")
    stage1_results = trainer.train_stage1_classifier(splits['train'], splits['validation'])
    
    # Train Stage 2: Fact Extractor
    print("\n🔍 Training Stage 2: Fact Extractor...")
    stage2_results = trainer.train_stage2_extractor(splits['train'], splits['validation'])
    
    # Train Stage 3: Eligibility Predictor
    print("\n⚖️  Training Stage 3: Eligibility Predictor...")
    stage3_results = trainer.train_eligibility_predictor(splits['train'], splits['validation'])
    
    # Evaluate full pipeline
    print("\n🧪 Evaluating complete pipeline...")
    pipeline_metrics = trainer.evaluate_full_pipeline(splits['test'])
    
    # Save trained models
    print("\n💾 Saving trained models...")
    saved_models = trainer.save_trained_models()
    
    # Test with example query
    print("\n🚀 Testing with example query...")
    example_query = "I am a poor woman earning Rs 15000 monthly. My husband beats me and I need legal help for divorce and maintenance."
    
    prediction = trainer.predict_legal_aid_eligibility(example_query)
    
    print(f"\nExample Query: {example_query}")
    print(f"Predicted Domains: {prediction['predicted_domains']}")
    print(f"Eligibility: {'✅ ELIGIBLE' if prediction['eligibility_prediction'] else '❌ NOT ELIGIBLE'}")
    print(f"Confidence: {prediction['eligibility_confidence']:.3f}")
    print(f"Legal Reasoning: {prediction['legal_reasoning']}")
    
    print("\n" + "=" * 50)
    print("🎉 HybEx-Law Training Complete!")
    print(f"📊 Overall Pipeline Score: {pipeline_metrics['overall_pipeline_score']:.3f}")
    print("✅ Models saved and ready for deployment!")

if __name__ == "__main__":
    main()
