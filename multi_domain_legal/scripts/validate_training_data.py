"""
Data validation and preparation script for the HybEx-Law system.

This script validates generated training data, prepares training splits,
and ensures data quality for neural component training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDataValidator:
    """Validates legal training data for quality and consistency"""
    
    def __init__(self):
        self.validation_results = {}
        self.required_fields = [
            'query', 'domains', 'extracted_facts', 'expected_eligibility',
            'legal_reasoning', 'confidence_factors', 'user_demographics'
        ]
        
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Comprehensive validation of legal training dataset.
        
        Args:
            dataset_path: Path to the JSON dataset file
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating dataset: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        
        validation_results = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'invalid_samples': 0,
            'validation_errors': [],
            'quality_metrics': {},
            'data_distribution': {}
        }
        
        valid_samples = []
        
        for i, sample in enumerate(samples):
            errors = self._validate_sample(sample, i)
            
            if errors:
                validation_results['invalid_samples'] += 1
                validation_results['validation_errors'].extend(errors)
            else:
                validation_results['valid_samples'] += 1
                valid_samples.append(sample)
        
        # Calculate quality metrics
        validation_results['quality_metrics'] = self._calculate_quality_metrics(valid_samples)
        validation_results['data_distribution'] = self._analyze_data_distribution(valid_samples)
        
        # Save validation report
        self._save_validation_report(validation_results, dataset_path)
        
        return validation_results
    
    def _validate_sample(self, sample: Dict, index: int) -> List[str]:
        """Validate individual sample"""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in sample:
                errors.append(f"Sample {index}: Missing required field '{field}'")
            elif not sample[field]:
                errors.append(f"Sample {index}: Empty value for field '{field}'")
        
        # Validate query
        if 'query' in sample:
            query = sample['query']
            if len(query) < 10:
                errors.append(f"Sample {index}: Query too short ({len(query)} chars)")
            elif len(query) > 1000:
                errors.append(f"Sample {index}: Query too long ({len(query)} chars)")
        
        # Validate domains
        if 'domains' in sample:
            domains = sample['domains']
            if not isinstance(domains, list) or not domains:
                errors.append(f"Sample {index}: Invalid domains format")
            else:
                valid_domains = {'legal_aid', 'family_law', 'consumer_protection', 'fundamental_rights', 'employment_law'}
                for domain in domains:
                    if domain not in valid_domains:
                        errors.append(f"Sample {index}: Invalid domain '{domain}'")
        
        # Validate extracted facts
        if 'extracted_facts' in sample:
            facts = sample['extracted_facts']
            if not isinstance(facts, list) or not facts:
                errors.append(f"Sample {index}: No extracted facts")
            else:
                for fact in facts:
                    if not fact.endswith('.'):
                        errors.append(f"Sample {index}: Malformed Prolog fact '{fact}'")
        
        # Validate eligibility
        if 'expected_eligibility' in sample:
            if not isinstance(sample['expected_eligibility'], bool):
                errors.append(f"Sample {index}: Eligibility must be boolean")
        
        # Validate confidence factors
        if 'confidence_factors' in sample:
            confidence = sample['confidence_factors']
            if not isinstance(confidence, dict):
                errors.append(f"Sample {index}: Confidence factors must be dict")
            else:
                for key, value in confidence.items():
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        errors.append(f"Sample {index}: Invalid confidence value for '{key}': {value}")
        
        return errors
    
    def _calculate_quality_metrics(self, samples: List[Dict]) -> Dict[str, float]:
        """Calculate data quality metrics"""
        if not samples:
            return {}
        
        metrics = {}
        
        # Query length statistics
        query_lengths = [len(sample['query']) for sample in samples]
        metrics['avg_query_length'] = np.mean(query_lengths)
        metrics['min_query_length'] = np.min(query_lengths)
        metrics['max_query_length'] = np.max(query_lengths)
        
        # Facts count statistics
        facts_counts = [len(sample['extracted_facts']) for sample in samples]
        metrics['avg_facts_count'] = np.mean(facts_counts)
        metrics['min_facts_count'] = np.min(facts_counts)
        metrics['max_facts_count'] = np.max(facts_counts)
        
        # Domain coverage
        all_domains = [domain for sample in samples for domain in sample['domains']]
        unique_domains = set(all_domains)
        metrics['unique_domains_count'] = len(unique_domains)
        metrics['multi_domain_samples'] = sum(1 for sample in samples if len(sample['domains']) > 1)
        metrics['multi_domain_percentage'] = (metrics['multi_domain_samples'] / len(samples)) * 100
        
        # Eligibility distribution
        eligible_count = sum(1 for sample in samples if sample['expected_eligibility'])
        metrics['eligibility_percentage'] = (eligible_count / len(samples)) * 100
        
        return metrics
    
    def _analyze_data_distribution(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze data distribution across various dimensions"""
        distribution = {}
        
        # Domain distribution
        domain_counts = {}
        for sample in samples:
            for domain in sample['domains']:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        distribution['domains'] = domain_counts
        
        # Complexity distribution
        complexity_counts = {}
        for sample in samples:
            complexity = sample.get('case_complexity', 'unknown')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        distribution['complexity'] = complexity_counts
        
        # Priority distribution
        priority_counts = {}
        for sample in samples:
            priority = sample.get('priority_level', 'unknown')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        distribution['priority'] = priority_counts
        
        return distribution
    
    def _save_validation_report(self, results: Dict, dataset_path: str):
        """Save validation report"""
        report_path = dataset_path.replace('.json', '_validation_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")

class TrainingDataPreparator:
    """Prepares validated data for neural model training"""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        
    def prepare_training_splits(self, dataset_path: str, output_dir: str = "data/splits") -> Dict[str, str]:
        """
        Prepare training, validation, and test splits for neural training.
        
        Args:
            dataset_path: Path to validated dataset
            output_dir: Directory to save splits
            
        Returns:
            Dictionary with paths to split files
        """
        logger.info(f"Preparing training splits from {dataset_path}")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data['samples']
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare data for domain classification
        domain_data = self._prepare_domain_classification_data(samples)
        
        # Prepare data for fact extraction
        fact_extraction_data = self._prepare_fact_extraction_data(samples)
        
        # Prepare data for confidence estimation
        confidence_data = self._prepare_confidence_estimation_data(samples)
        
        # Create splits for each task
        splits = {}
        
        # Domain classification splits
        splits.update(self._create_splits(
            domain_data, 
            f"{output_dir}/domain_classification", 
            "domain_classification"
        ))
        
        # Fact extraction splits
        splits.update(self._create_splits(
            fact_extraction_data,
            f"{output_dir}/fact_extraction",
            "fact_extraction"
        ))
        
        # Confidence estimation splits
        splits.update(self._create_splits(
            confidence_data,
            f"{output_dir}/confidence_estimation",
            "confidence_estimation"
        ))
        
        # Save split information
        split_info = {
            'total_samples': len(samples),
            'splits_created': splits,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
        
        with open(f"{output_dir}/split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Training splits prepared in {output_dir}")
        return splits
    
    def _prepare_domain_classification_data(self, samples: List[Dict]) -> List[Dict]:
        """Prepare data for domain classification task"""
        domain_data = []
        
        for sample in samples:
            domain_data.append({
                'text': sample['query'],
                'labels': sample['domains'],
                'sample_id': len(domain_data)
            })
        
        return domain_data
    
    def _prepare_fact_extraction_data(self, samples: List[Dict]) -> List[Dict]:
        """Prepare data for fact extraction task"""
        fact_data = []
        
        for sample in samples:
            # Convert Prolog facts to structured format
            structured_facts = []
            for fact in sample['extracted_facts']:
                # Parse Prolog fact (simplified parsing)
                if '(' in fact and fact.endswith('.'):
                    predicate = fact.split('(')[0]
                    structured_facts.append({
                        'predicate': predicate,
                        'full_fact': fact[:-1]  # Remove the trailing dot
                    })
            
            fact_data.append({
                'text': sample['query'],
                'facts': structured_facts,
                'domains': sample['domains'],
                'sample_id': len(fact_data)
            })
        
        return fact_data
    
    def _prepare_confidence_estimation_data(self, samples: List[Dict]) -> List[Dict]:
        """Prepare data for confidence estimation task"""
        confidence_data = []
        
        for sample in samples:
            confidence_data.append({
                'text': sample['query'],
                'domains': sample['domains'],
                'extracted_facts': sample['extracted_facts'],
                'confidence_scores': sample['confidence_factors'],
                'case_complexity': sample.get('case_complexity', 'medium'),
                'sample_id': len(confidence_data)
            })
        
        return confidence_data
    
    def _create_splits(self, data: List[Dict], output_prefix: str, task_name: str) -> Dict[str, str]:
        """Create train/val/test splits for a specific task"""
        # First split: 70% train, 30% temp
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        
        # Second split: 15% val, 15% test (from the 30% temp)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        splits = {}
        
        # Save splits
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            filepath = f"{output_prefix}_{split_name}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'task': task_name,
                    'split': split_name,
                    'count': len(split_data),
                    'data': split_data
                }, f, indent=2, ensure_ascii=False)
            
            splits[f"{task_name}_{split_name}"] = filepath
            logger.info(f"Saved {len(split_data)} {split_name} samples for {task_name}")
        
        return splits

def main():
    """Main function for data validation and preparation"""
    logger.info("Starting data validation and preparation")
    
    # Paths
    dataset_path = "data/comprehensive_legal_training_data.json"
    
    # Step 1: Validate dataset
    validator = LegalDataValidator()
    validation_results = validator.validate_dataset(dataset_path)
    
    print("\n📊 Validation Results:")
    print(f"✅ Valid samples: {validation_results['valid_samples']}")
    print(f"❌ Invalid samples: {validation_results['invalid_samples']}")
    print(f"📈 Data quality score: {(validation_results['valid_samples'] / validation_results['total_samples']) * 100:.1f}%")
    
    if validation_results['invalid_samples'] > 0:
        print(f"\n⚠️  Found {len(validation_results['validation_errors'])} validation errors")
        for error in validation_results['validation_errors'][:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(validation_results['validation_errors']) > 5:
            print(f"   ... and {len(validation_results['validation_errors']) - 5} more errors")
    
    # Step 2: Prepare training splits (only if validation passes)
    if validation_results['valid_samples'] > 0:
        preparator = TrainingDataPreparator()
        splits = preparator.prepare_training_splits(dataset_path)
        
        print("\n📁 Training Splits Created:")
        for split_name, filepath in splits.items():
            print(f"   {split_name}: {filepath}")
    
    logger.info("Data validation and preparation completed!")

if __name__ == "__main__":
    main()
