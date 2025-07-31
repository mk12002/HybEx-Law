"""
Neural Training Script for Multi-Domain Legal AI System.

This script trains the neural components of the hybrid system including
domain classification and fact extraction models.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_training_data(num_samples_per_domain: int = 200) -> Tuple[List[str], List[List[str]]]:
    """
    Generate synthetic training data for neural domain classification.
    
    Args:
        num_samples_per_domain: Number of training samples per domain
        
    Returns:
        Tuple of (queries, domain_labels) for training
    """
    logger.info(f"Generating {num_samples_per_domain} samples per domain...")
    
    # Extended training templates for each domain
    training_templates = {
        'legal_aid': [
            "I am {income_status} and need {legal_service} for {case_type}",
            "Being {category}, can I get free legal help for {issue}?",
            "I cannot afford a lawyer for my {case_type} case",
            "Need government legal aid for {issue} as I am {financial_status}",
            "Legal services authority help needed for {category} person",
            "Free lawyer required for {case_type} - I am {income_status}",
            "Legal aid eligibility for {category} in {case_type} matter",
            "Government legal assistance needed - {financial_status} family",
            "Can {category} get legal aid for {issue}?",
            "Lok adalat help needed for {case_type} dispute"
        ],
        'family_law': [
            "My {relation} is {behavior} and I want {remedy}",
            "{case_type} proceedings against {relation} for {ground}",
            "Need {remedy} from {relation} after {situation}",
            "Child {issue} matter after {family_event}",
            "Property {dispute_type} between {family_members}",
            "Marriage {issue} under {personal_law}",
            "{domestic_issue} by {family_member} - need legal help",
            "Divorce case for {ground} against {spouse}",
            "Maintenance dispute with {family_member}",
            "Inheritance rights of {family_member} in {property_type}"
        ],
        'consumer_protection': [
            "My {product} is {defect} and {seller} is refusing {remedy}",
            "{service_provider} provided {poor_service} - want {compensation}",
            "Online purchase of {product} resulted in {fraud_type}",
            "{company} violated {consumer_right} - need complaint guidance",
            "Defective {product} under warranty - {seller} not helping",
            "Builder delayed {property} possession by {time_period}",
            "Restaurant gave {food_issue} - suffered {health_impact}",
            "E-commerce fraud in {product} purchase - {loss_amount}",
            "Insurance claim rejected unfairly for {claim_type}",
            "Medical negligence by {provider} caused {harm}"
        ],
        'fundamental_rights': [
            "{authority} violated my {right} by {violation_act}",
            "Discrimination based on {ground} by {discriminator}",
            "RTI application {rti_issue} by {government_office}",
            "Police {misconduct} without {legal_requirement}",
            "Government {action} violating {constitutional_right}",
            "Illegal detention by {authority} for {duration}",
            "Freedom of {freedom_type} restricted by {restrictor}",
            "Caste discrimination in {context} by {discriminator}",
            "Right to information denied about {subject}",
            "Constitutional remedy needed for {rights_violation}"
        ],
        'employment_law': [
            "My {employer} {action} me {circumstances}",
            "{workplace_issue} at job by {perpetrator}",
            "Not getting {benefit} despite {entitlement}",
            "Workplace {discrimination_type} based on {ground}",
            "{termination_type} from job after {service_period}",
            "Salary {wage_issue} - company paying {amount_issue}",
            "{harassment_type} by {workplace_person}",
            "Denied {leave_type} by employer despite {reason}",
            "Contract violation by employer in {contract_aspect}",
            "Labor dispute over {dispute_subject} with {employer_type}"
        ]
    }
    
    # Template variable fillers
    template_vars = {
        'income_status': ['poor', 'below poverty line', 'BPL family', 'earning less than 2 lakh'],
        'legal_service': ['legal help', 'lawyer', 'court representation', 'legal advice'],
        'case_type': ['family dispute', 'property case', 'criminal matter', 'civil case'],
        'category': ['SC/ST', 'woman', 'child', 'disabled person', 'senior citizen'],
        'issue': ['domestic violence', 'property dispute', 'criminal case', 'civil matter'],
        'financial_status': ['financially weak', 'poor', 'below poverty line', 'cannot afford'],
        
        'relation': ['husband', 'wife', 'spouse', 'father', 'mother', 'son', 'daughter'],
        'behavior': ['cruel', 'abusive', 'violent', 'harassing', 'not supporting'],
        'remedy': ['divorce', 'maintenance', 'custody', 'protection', 'separation'],
        'ground': ['cruelty', 'domestic violence', 'desertion', 'adultery', 'dowry harassment'],
        'situation': ['separation', 'divorce', 'marriage breakdown', 'domestic violence'],
        'family_event': ['divorce', 'death', 'separation', 'marriage'],
        'dispute_type': ['inheritance', 'partition', 'ownership', 'succession'],
        'family_members': ['siblings', 'parents', 'children', 'relatives'],
        'personal_law': ['Hindu law', 'Muslim law', 'Christian law', 'civil law'],
        'domestic_issue': ['violence', 'harassment', 'cruelty', 'abuse'],
        'family_member': ['husband', 'wife', 'father', 'mother', 'in-laws'],
        'spouse': ['husband', 'wife'],
        'property_type': ['ancestral property', 'self-acquired property', 'family home'],
        
        'product': ['mobile phone', 'laptop', 'car', 'washing machine', 'TV', 'refrigerator'],
        'defect': ['not working', 'damaged', 'faulty', 'broken', 'malfunctioning'],
        'seller': ['shop', 'dealer', 'company', 'manufacturer', 'retailer'],
        'remedy': ['refund', 'replacement', 'repair', 'compensation'],
        'service_provider': ['restaurant', 'hotel', 'bank', 'telecom company', 'hospital'],
        'poor_service': ['bad service', 'delayed service', 'wrong service', 'no service'],
        'compensation': ['compensation', 'refund', 'damages', 'apology'],
        'fraud_type': ['payment fraud', 'fake product', 'non-delivery', 'wrong product'],
        'company': ['bank', 'insurance company', 'mobile company', 'e-commerce site'],
        'consumer_right': ['warranty terms', 'refund policy', 'service standards'],
        'property': ['apartment', 'house', 'flat', 'plot'],
        'time_period': ['1 year', '2 years', '6 months', '18 months'],
        'food_issue': ['food poisoning', 'stale food', 'foreign object', 'contaminated food'],
        'health_impact': ['illness', 'hospitalization', 'medical expenses', 'suffering'],
        'loss_amount': ['Rs 50000', 'Rs 25000', 'Rs 1 lakh', 'significant money'],
        'claim_type': ['accident claim', 'health claim', 'life insurance', 'vehicle insurance'],
        'provider': ['doctor', 'hospital', 'clinic', 'medical professional'],
        'harm': ['injury', 'worsened condition', 'complications', 'disability'],
        
        'authority': ['police', 'government office', 'municipal corporation', 'collector'],
        'right': ['fundamental right', 'constitutional right', 'human right', 'legal right'],
        'violation_act': ['illegal arrest', 'discrimination', 'denial of service', 'harassment'],
        'ground': ['caste', 'religion', 'gender', 'disability', 'economic status'],
        'discriminator': ['government office', 'private company', 'school', 'hospital'],
        'rti_issue': ['rejected', 'delayed', 'incomplete response', 'ignored'],
        'government_office': ['collector office', 'municipal corporation', 'police station'],
        'misconduct': ['arrested without warrant', 'harassment', 'illegal detention', 'torture'],
        'legal_requirement': ['proper procedure', 'warrant', 'grounds', 'notice'],
        'action': ['order', 'policy', 'decision', 'action'],
        'constitutional_right': ['equality', 'freedom', 'life and liberty', 'religious freedom'],
        'duration': ['3 days', '1 week', '24 hours', 'several days'],
        'freedom_type': ['speech', 'religion', 'movement', 'assembly'],
        'restrictor': ['government', 'local authority', 'police', 'administration'],
        'context': ['workplace', 'school', 'hospital', 'government office'],
        'subject': ['government scheme', 'public project', 'policy details', 'expenditure'],
        'rights_violation': ['discrimination', 'illegal detention', 'denial of rights'],
        
        'employer': ['company', 'boss', 'management', 'supervisor'],
        'action': ['fired', 'terminated', 'dismissed', 'removed', 'harassed'],
        'circumstances': ['without notice', 'without reason', 'illegally', 'unfairly'],
        'workplace_issue': ['harassment', 'discrimination', 'unsafe conditions', 'exploitation'],
        'perpetrator': ['boss', 'colleague', 'supervisor', 'manager'],
        'benefit': ['salary', 'overtime pay', 'bonus', 'PF', 'gratuity'],
        'entitlement': ['working extra hours', 'completing targets', 'policy rules'],
        'discrimination_type': ['gender discrimination', 'caste discrimination', 'age discrimination'],
        'termination_type': ['wrongful termination', 'illegal dismissal', 'unfair removal'],
        'service_period': ['5 years', '3 years', '2 years', 'long service'],
        'wage_issue': ['not paid', 'delayed', 'less than minimum wage', 'deducted'],
        'amount_issue': ['below minimum wage', 'less than agreed', 'irregular payment'],
        'harassment_type': ['sexual harassment', 'mental harassment', 'physical harassment'],
        'workplace_person': ['boss', 'senior colleague', 'manager', 'team lead'],
        'leave_type': ['maternity leave', 'medical leave', 'casual leave', 'earned leave'],
        'reason': ['medical condition', 'family emergency', 'pregnancy', 'illness'],
        'contract_aspect': ['salary terms', 'working hours', 'benefits', 'job security'],
        'dispute_subject': ['wages', 'working conditions', 'benefits', 'termination'],
        'employer_type': ['private company', 'government office', 'factory', 'organization']
    }
    
    queries = []
    labels = []
    
    for domain, templates in training_templates.items():
        for _ in range(num_samples_per_domain):
            # Select random template
            template = random.choice(templates)
            
            # Fill template variables
            filled_query = template
            for var_name, options in template_vars.items():
                if f'{{{var_name}}}' in filled_query:
                    filled_query = filled_query.replace(f'{{{var_name}}}', random.choice(options))
            
            queries.append(filled_query)
            labels.append([domain])
            
            # Generate some multi-domain samples (10% of data)
            if random.random() < 0.1:
                # Create cross-domain scenarios
                cross_domain_combinations = {
                    'legal_aid': ['family_law', 'employment_law', 'consumer_protection'],
                    'family_law': ['legal_aid', 'fundamental_rights'],
                    'employment_law': ['legal_aid', 'fundamental_rights'],
                    'consumer_protection': ['legal_aid', 'fundamental_rights'],
                    'fundamental_rights': ['employment_law', 'family_law']
                }
                
                if domain in cross_domain_combinations:
                    additional_domain = random.choice(cross_domain_combinations[domain])
                    labels[-1].append(additional_domain)
    
    logger.info(f"Generated {len(queries)} training samples")
    return queries, labels

def train_neural_components(queries: List[str], labels: List[List[str]]):
    """Train neural components with generated data"""
    
    try:
        from src.core.neural_components import NeuralDomainClassifier, NeuralFactExtractor
        
        logger.info("Training Neural Domain Classifier...")
        
        # Train domain classifier
        neural_classifier = NeuralDomainClassifier()
        neural_classifier.train(queries, labels, epochs=5, batch_size=32)
        
        logger.info("Training Neural Fact Extractor...")
        
        # Prepare fact extraction training data
        fact_training_data = []
        for query, query_labels in zip(queries[:100], labels[:100]):  # Use subset for fact training
            expected_facts = []
            
            # Generate expected facts based on query content
            if 'legal_aid' in query_labels:
                if any(word in query.lower() for word in ['poor', 'bpl', 'cannot afford']):
                    expected_facts.append('financial_distress(user, true).')
                if any(word in query.lower() for word in ['sc', 'st', 'woman', 'child']):
                    expected_facts.append('categorical_eligibility(user, true).')
            
            if 'employment_law' in query_labels:
                if any(word in query.lower() for word in ['fired', 'terminated', 'dismissed']):
                    expected_facts.append('termination_occurred(user, true).')
                if any(word in query.lower() for word in ['harassment', 'discrimination']):
                    expected_facts.append('workplace_issue(user, true).')
            
            # Add more fact generation logic for other domains...
            
            fact_training_data.append({
                'query': query,
                'expected_facts': expected_facts,
                'domains': query_labels
            })
        
        # Train fact extractor
        neural_fact_extractor = NeuralFactExtractor()
        neural_fact_extractor.train_on_domain_data(fact_training_data)
        
        logger.info("Neural components training completed successfully!")
        
        return neural_classifier, neural_fact_extractor
        
    except ImportError as e:
        logger.error(f"Could not import neural components: {e}")
        logger.info("Please install required dependencies: torch, transformers, scikit-learn")
        return None, None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None, None

def create_neural_training_config():
    """Create configuration for neural training"""
    config = {
        "domain_classifier": {
            "model_name": "distilbert-base-uncased",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "max_length": 512
        },
        "fact_extractor": {
            "model_name": "microsoft/DialoGPT-small",
            "use_ner_pipeline": True,
            "confidence_threshold": 0.7
        },
        "training_data": {
            "samples_per_domain": 200,
            "multi_domain_ratio": 0.1,
            "test_split": 0.2
        }
    }
    
    # Save config
    config_path = Path("config/neural_training_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Neural training configuration saved to {config_path}")
    return config

def main():
    """Main training pipeline"""
    logger.info("Starting Neural Training Pipeline for Multi-Domain Legal AI")
    
    # Create training configuration
    config = create_neural_training_config()
    
    # Generate training data
    queries, labels = generate_training_data(
        num_samples_per_domain=config["training_data"]["samples_per_domain"]
    )
    
    # Save training data
    training_data_path = Path("data/neural_training_data.json")
    training_data_path.parent.mkdir(exist_ok=True)
    
    training_data = {
        "queries": queries,
        "labels": labels,
        "config": config,
        "total_samples": len(queries),
        "domains": ["legal_aid", "family_law", "consumer_protection", "fundamental_rights", "employment_law"]
    }
    
    with open(training_data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Training data saved to {training_data_path}")
    
    # Train neural components
    neural_classifier, neural_fact_extractor = train_neural_components(queries, labels)
    
    if neural_classifier and neural_fact_extractor:
        logger.info("✅ Neural training completed successfully!")
        logger.info("The system now has hybrid neural-symbolic capabilities")
    else:
        logger.warning("⚠️ Neural training failed - system will use rule-based fallback")
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main()
