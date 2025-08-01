#!/usr/bin/env python3
"""
Dataset Loader for HybEx-Law Training
=====================================

This script provides utilities to load and prepare the generated legal dataset
for training the hybrid neural-symbolic system.

Author: HybEx-Law Team
Date: August 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingBatch:
    """Structured training batch for different model components"""
    
    # Stage 1: Domain Classification Data
    queries: List[str]
    domain_labels: np.ndarray  # Multi-label binary matrix
    
    # Stage 2: Fact Extraction Data
    facts: List[List[str]]
    
    # Eligibility Prediction Data
    eligibility_labels: List[bool]
    legal_reasoning: List[str]
    
    # Metadata
    complexity_levels: List[str]
    priority_levels: List[str]
    demographics: List[Dict]

class LegalDatasetLoader:
    """
    Comprehensive dataset loader for HybEx-Law training pipeline.
    Handles the generated 23,500 sample dataset efficiently.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing the generated datasets
        """
        self.data_dir = Path(data_dir)
        self.domain_encoder = MultiLabelBinarizer()
        self.datasets = {}
        
        # Legal domain mapping
        self.domain_mapping = {
            'legal_aid': 0,
            'family_law': 1, 
            'consumer_protection': 2,
            'employment_law': 3,
            'fundamental_rights': 4
        }
        
        logger.info(f"Initialized dataset loader for directory: {self.data_dir}")
    
    def load_all_datasets(self) -> Dict[str, Any]:
        """
        Load ALL generated datasets (23,500 total samples).
        
        Returns:
            Dictionary containing complete consolidated dataset
        """
        all_samples = []
        dataset_info = {}
        
        # Load main dataset (20,000 samples)
        main_file = self.data_dir / "comprehensive_legal_training_data.json"
        if main_file.exists():
            with open(main_file, 'r', encoding='utf-8') as f:
                main_data = json.load(f)
            all_samples.extend(main_data['samples'])
            dataset_info['main'] = len(main_data['samples'])
            logger.info(f"Loaded {len(main_data['samples'])} main training samples")
        
        # Load edge cases (1,000 samples)
        edge_cases_file = self.data_dir / "edge_cases_dataset.json"
        if edge_cases_file.exists():
            with open(edge_cases_file, 'r', encoding='utf-8') as f:
                edge_data = json.load(f)
            all_samples.extend(edge_data['samples'])
            dataset_info['edge_cases'] = len(edge_data['samples'])
            logger.info(f"Loaded {len(edge_data['samples'])} edge case samples")
        
        # Load domain-specific validation sets (500 each × 5 = 2,500 samples)
        domains = ['legal_aid', 'family_law', 'consumer_protection', 
                   'employment_law', 'fundamental_rights']
        
        for domain in domains:
            validation_file = self.data_dir / f"{domain}_validation_set.json"
            if validation_file.exists():
                with open(validation_file, 'r', encoding='utf-8') as f:
                    domain_data = json.load(f)
                all_samples.extend(domain_data['samples'])
                dataset_info[f"{domain}_validation"] = len(domain_data['samples'])
                logger.info(f"Loaded {len(domain_data['samples'])} {domain} validation samples")
        
        # Create consolidated dataset
        consolidated_data = {
            'metadata': {
                'total_samples': len(all_samples),
                'dataset_breakdown': dataset_info,
                'generation_date': '2025-08-01',
                'domains_covered': ['legal_aid', 'family_law', 'consumer_protection', 
                                  'employment_law', 'fundamental_rights']
            },
            'samples': all_samples
        }
        
        self.datasets['consolidated'] = consolidated_data
        logger.info(f"Consolidated dataset: {len(all_samples)} total samples from {len(dataset_info)} sources")
        
        return consolidated_data
    
    def load_validation_datasets(self) -> Dict[str, Any]:
        """
        Load domain-specific validation datasets and edge cases.
        
        Returns:
            Dictionary containing all validation datasets
        """
        validation_data = {}
        
        # Load edge cases
        edge_cases_file = self.data_dir / "edge_cases_dataset.json"
        if edge_cases_file.exists():
            with open(edge_cases_file, 'r', encoding='utf-8') as f:
                validation_data['edge_cases'] = json.load(f)
            logger.info("Loaded edge cases dataset")
        
        # Load domain-specific validation sets
        domains = ['legal_aid', 'family_law', 'consumer_protection', 
                   'employment_law', 'fundamental_rights']
        
        for domain in domains:
            validation_file = self.data_dir / f"{domain}_validation_set.json"
            if validation_file.exists():
                with open(validation_file, 'r', encoding='utf-8') as f:
                    validation_data[f"{domain}_validation"] = json.load(f)
                logger.info(f"Loaded {domain} validation dataset")
        
        self.datasets['validation'] = validation_data
        return validation_data
    
    def create_training_splits(self, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, TrainingBatch]:
        """
        Create proper train/validation/test splits from ALL datasets (23,500 samples).
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            
        Returns:
            Dictionary with train/val/test splits as TrainingBatch objects
        """
        if 'consolidated' not in self.datasets:
            self.load_all_datasets()
        
        samples = self.datasets['consolidated']['samples']
        logger.info(f"Creating splits from {len(samples)} total samples")
        
        # Extract features
        queries = [sample['query'] for sample in samples]
        domains = [sample['domains'] for sample in samples]
        facts = [sample['extracted_facts'] for sample in samples]
        eligibility = [sample['expected_eligibility'] for sample in samples]
        reasoning = [sample['legal_reasoning'] for sample in samples]
        complexity = [sample['case_complexity'] for sample in samples]
        priority = [sample['priority_level'] for sample in samples]
        demographics = [sample['user_demographics'] for sample in samples]
        
        # Encode domain labels for multi-label classification
        domain_labels = self.domain_encoder.fit_transform(domains)
        
        # Create stratified splits based on primary domain AND eligibility
        primary_domains = [d[0] if d else 'unknown' for d in domains]
        stratify_labels = [f"{domain}_{elig}" for domain, elig in zip(primary_domains, eligibility)]
        
        # First split: train + val vs test
        indices = list(range(len(samples)))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=stratify_labels, random_state=42
        )
        
        # Second split: train vs val
        train_val_stratify = [stratify_labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size, stratify=train_val_stratify, random_state=42
        )
        
        # Create training batches
        splits = {}
        for split_name, indices in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
            splits[split_name] = TrainingBatch(
                queries=[queries[i] for i in indices],
                domain_labels=domain_labels[indices],
                facts=[facts[i] for i in indices],
                eligibility_labels=[eligibility[i] for i in indices],
                legal_reasoning=[reasoning[i] for i in indices],
                complexity_levels=[complexity[i] for i in indices],
                priority_levels=[priority[i] for i in indices],
                demographics=[demographics[i] for i in indices]
            )
            
            logger.info(f"Created {split_name} split with {len(indices)} samples")
        
        return splits
    
    def prepare_stage1_data(self, training_batch: TrainingBatch) -> Tuple[List[str], np.ndarray]:
        """
        Prepare data for Stage 1: Multi-label Domain Classifier.
        
        Args:
            training_batch: TrainingBatch object
            
        Returns:
            Tuple of (queries, domain_labels) for classification training
        """
        logger.info("Preparing Stage 1 domain classification data")
        return training_batch.queries, training_batch.domain_labels
    
    def prepare_stage2_data(self, training_batch: TrainingBatch) -> Tuple[List[str], List[List[str]]]:
        """
        Prepare data for Stage 2: Fact Extraction.
        
        Args:
            training_batch: TrainingBatch object
            
        Returns:
            Tuple of (queries, facts) for extraction training
        """
        logger.info("Preparing Stage 2 fact extraction data")
        return training_batch.queries, training_batch.facts
    
    def prepare_eligibility_data(self, training_batch: TrainingBatch) -> Tuple[List[List[str]], List[bool], List[str]]:
        """
        Prepare data for Eligibility Prediction (Prolog + Neural).
        
        Args:
            training_batch: TrainingBatch object
            
        Returns:
            Tuple of (facts, eligibility_labels, reasoning) for eligibility training
        """
        logger.info("Preparing eligibility prediction data")
        return training_batch.facts, training_batch.eligibility_labels, training_batch.legal_reasoning
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive dataset statistics for analysis.
        
        Returns:
            Dictionary containing detailed statistics
        """
        if 'consolidated' not in self.datasets:
            self.load_all_datasets()
        
        samples = self.datasets['consolidated']['samples']
        
        stats = {
            'total_samples': len(samples),
            'dataset_breakdown': self.datasets['consolidated']['metadata']['dataset_breakdown'],
            'domain_distribution': {},
            'eligibility_distribution': {'eligible': 0, 'not_eligible': 0},
            'complexity_distribution': {},
            'priority_distribution': {},
            'average_facts_per_sample': 0,
            'average_query_length': 0,
            'cross_domain_cases': 0,
            'multi_domain_breakdown': {},
            'income_range_analysis': {'low': 0, 'medium': 0, 'high': 0, 'unknown': 0}
        }
        
        total_facts = 0
        total_words = 0
        
        for sample in samples:
            # Domain distribution (count each domain occurrence)
            for domain in sample['domains']:
                stats['domain_distribution'][domain] = stats['domain_distribution'].get(domain, 0) + 1
            
            # Multi-domain analysis (fixed logic)
            domain_count = len(sample['domains'])
            if domain_count > 1:
                stats['cross_domain_cases'] += 1
                domain_key = f"{domain_count}_domains"
                stats['multi_domain_breakdown'][domain_key] = stats['multi_domain_breakdown'].get(domain_key, 0) + 1
            else:
                stats['multi_domain_breakdown']['1_domain'] = stats['multi_domain_breakdown'].get('1_domain', 0) + 1
            
            # Eligibility distribution
            key = 'eligible' if sample['expected_eligibility'] else 'not_eligible'
            stats['eligibility_distribution'][key] += 1
            
            # Complexity distribution
            complexity = sample['case_complexity']
            stats['complexity_distribution'][complexity] = stats['complexity_distribution'].get(complexity, 0) + 1
            
            # Priority distribution
            priority = sample['priority_level']
            stats['priority_distribution'][priority] = stats['priority_distribution'].get(priority, 0) + 1
            
            # Facts and query length
            total_facts += len(sample['extracted_facts'])
            total_words += len(sample['query'].split())
            
            # Income analysis
            query_lower = sample['query'].lower()
            if any(str(income) in query_lower for income in range(5000, 15000, 1000)):
                stats['income_range_analysis']['low'] += 1
            elif any(str(income) in query_lower for income in range(15000, 30000, 1000)):
                stats['income_range_analysis']['medium'] += 1
            elif any(str(income) in query_lower for income in range(30000, 60000, 1000)):
                stats['income_range_analysis']['high'] += 1
            else:
                stats['income_range_analysis']['unknown'] += 1
        
        stats['average_facts_per_sample'] = total_facts / len(samples)
        stats['average_query_length'] = total_words / len(samples)
        stats['cross_domain_percentage'] = (stats['cross_domain_cases'] / len(samples)) * 100
        stats['eligibility_percentage'] = {
            'eligible': (stats['eligibility_distribution']['eligible'] / len(samples)) * 100,
            'not_eligible': (stats['eligibility_distribution']['not_eligible'] / len(samples)) * 100
        }
        
        return stats
    
    def visualize_dataset(self, save_plots: bool = True) -> None:
        """
        Create visualization plots for dataset analysis.
        
        Args:
            save_plots: Whether to save plots to files
        """
        stats = self.get_dataset_statistics()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('HybEx-Law Comprehensive Dataset Analysis (23,500 Samples)', fontsize=16)
        
        # Domain distribution
        domains = list(stats['domain_distribution'].keys())
        counts = list(stats['domain_distribution'].values())
        axes[0, 0].bar(domains, counts, color='skyblue')
        axes[0, 0].set_title('Domain Distribution')
        axes[0, 0].set_xlabel('Legal Domains')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Eligibility distribution
        eligibility_data = stats['eligibility_distribution']
        colors = ['lightgreen', 'lightcoral']
        axes[0, 1].pie(eligibility_data.values(), labels=eligibility_data.keys(), 
                       autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('Eligibility Distribution')
        
        # Complexity distribution
        complexity_data = stats['complexity_distribution']
        axes[0, 2].bar(complexity_data.keys(), complexity_data.values(), color='orange')
        axes[0, 2].set_title('Case Complexity Distribution')
        axes[0, 2].set_xlabel('Complexity Level')
        axes[0, 2].set_ylabel('Number of Cases')
        
        # Multi-domain breakdown
        multi_domain_data = stats['multi_domain_breakdown']
        axes[1, 0].bar(multi_domain_data.keys(), multi_domain_data.values(), color='purple')
        axes[1, 0].set_title('Multi-Domain Case Analysis')
        axes[1, 0].set_xlabel('Domain Count')
        axes[1, 0].set_ylabel('Number of Cases')
        
        # Income range analysis
        income_data = stats['income_range_analysis']
        axes[1, 1].bar(income_data.keys(), income_data.values(), color='gold')
        axes[1, 1].set_title('Income Range Distribution')
        axes[1, 1].set_xlabel('Income Category')
        axes[1, 1].set_ylabel('Number of Cases')
        
        # Dataset overview metrics
        metrics_text = f"""
        Total Samples: {stats['total_samples']:,}
        
        Cross-domain Cases: {stats['cross_domain_percentage']:.1f}%
        Eligible Cases: {stats['eligibility_percentage']['eligible']:.1f}%
        Not Eligible: {stats['eligibility_percentage']['not_eligible']:.1f}%
        
        Avg Facts/Sample: {stats['average_facts_per_sample']:.1f}
        Avg Query Length: {stats['average_query_length']:.1f} words
        
        Dataset Sources:
        • Main: {stats['dataset_breakdown'].get('main', 0):,}
        • Edge Cases: {stats['dataset_breakdown'].get('edge_cases', 0):,}
        • Validation Sets: {sum(v for k, v in stats['dataset_breakdown'].items() if 'validation' in k):,}
        """
        axes[1, 2].text(0.05, 0.95, metrics_text, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 2].set_title('Dataset Overview')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.data_dir / 'comprehensive_dataset_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dataset visualization saved to {plot_path}")
        
        plt.show()
    
    def export_for_training(self, output_format: str = 'json') -> Dict[str, Path]:
        """
        Export dataset in format optimized for neural training.
        
        Args:
            output_format: 'json', 'csv', or 'pickle'
            
        Returns:
            Dictionary mapping split names to file paths
        """
        splits = self.create_training_splits()
        exported_files = {}
        
        for split_name, batch in splits.items():
            if output_format == 'json':
                file_path = self.data_dir / f"training_{split_name}.json"
                data = {
                    'queries': batch.queries,
                    'domain_labels': batch.domain_labels.tolist(),
                    'facts': batch.facts,
                    'eligibility': batch.eligibility_labels,
                    'reasoning': batch.legal_reasoning,
                    'complexity': batch.complexity_levels,
                    'priority': batch.priority_levels
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif output_format == 'csv':
                file_path = self.data_dir / f"training_{split_name}.csv"
                df = pd.DataFrame({
                    'query': batch.queries,
                    'domains': [','.join(map(str, row)) for row in batch.domain_labels],
                    'facts': [';'.join(facts) for facts in batch.facts],
                    'eligibility': batch.eligibility_labels,
                    'reasoning': batch.legal_reasoning,
                    'complexity': batch.complexity_levels,
                    'priority': batch.priority_levels
                })
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            exported_files[split_name] = file_path
            logger.info(f"Exported {split_name} split to {file_path}")
        
        return exported_files

# Example usage and testing
def main():
    """Demonstrate dataset loading and preparation."""
    
    print("🏛️  HybEx-Law Dataset Loader Demo")
    print("=" * 50)
    
    # Initialize loader
    loader = LegalDatasetLoader("data")
    
    # Load ALL datasets (23,500 samples)
    print("\n📁 Loading all datasets...")
    consolidated_data = loader.load_all_datasets()
    
    print(f"\n📊 Dataset Breakdown:")
    for dataset_name, count in consolidated_data['metadata']['dataset_breakdown'].items():
        print(f"   • {dataset_name}: {count:,} samples")
    
    # Create training splits
    print(f"\n🎯 Creating training splits from {consolidated_data['metadata']['total_samples']:,} total samples...")
    splits = loader.create_training_splits()
    
    # Show split information
    for split_name, batch in splits.items():
        eligible_count = sum(batch.eligibility_labels)
        eligible_pct = (eligible_count / len(batch.eligibility_labels)) * 100
        print(f"   • {split_name.capitalize()}: {len(batch.queries):,} samples ({eligible_pct:.1f}% eligible)")
    
    # Prepare data for different training stages
    print("\n🧠 Preparing data for training stages...")
    
    # Stage 1: Domain Classification
    train_queries, train_domains = loader.prepare_stage1_data(splits['train'])
    print(f"   • Stage 1 (Domain Classification): {len(train_queries):,} samples")
    
    # Stage 2: Fact Extraction
    train_queries_facts, train_facts = loader.prepare_stage2_data(splits['train'])
    print(f"   • Stage 2 (Fact Extraction): {len(train_facts):,} samples")
    
    # Eligibility Prediction
    train_facts_elig, train_eligibility, train_reasoning = loader.prepare_eligibility_data(splits['train'])
    print(f"   • Eligibility Prediction: {len(train_eligibility):,} samples")
    
    # Generate comprehensive statistics
    print("\n📊 Comprehensive Dataset Statistics:")
    stats = loader.get_dataset_statistics()
    print(f"   • Total samples: {stats['total_samples']:,}")
    print(f"   • Cross-domain cases: {stats['cross_domain_percentage']:.1f}%")
    print(f"   • Average facts per sample: {stats['average_facts_per_sample']:.1f}")
    print(f"   • Eligible cases: {stats['eligibility_percentage']['eligible']:.1f}%")
    print(f"   • Not eligible cases: {stats['eligibility_percentage']['not_eligible']:.1f}%")
    
    print(f"\n🏛️ Domain Distribution:")
    for domain, count in stats['domain_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"   • {domain}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🔀 Multi-Domain Analysis:")
    for domain_type, count in stats['multi_domain_breakdown'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"   • {domain_type}: {count:,} ({percentage:.1f}%)")
    
    # Export for training
    print("\n💾 Exporting training-ready datasets...")
    exported_files = loader.export_for_training('json')
    
    for split_name, file_path in exported_files.items():
        print(f"   • {split_name.capitalize()}: {file_path}")
    
    # Visualize dataset
    print("\n📈 Generating dataset visualization...")
    loader.visualize_dataset(save_plots=True)
    
    print("\n✅ Dataset preparation complete!")
    print(f"🚀 Ready for HybEx-Law training pipeline with {stats['total_samples']:,} samples!")
    print(f"📊 Balanced eligibility: {stats['eligibility_percentage']['eligible']:.1f}% eligible, {stats['eligibility_percentage']['not_eligible']:.1f}% not eligible")
    print(f"🔀 Cross-domain coverage: {stats['cross_domain_percentage']:.1f}% multi-domain cases")

if __name__ == "__main__":
    main()
