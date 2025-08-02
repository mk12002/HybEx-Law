"""
HybEx-Law Training Data Organization Script
==========================================

This script properly organizes the training data into clean train/validation/test splits
avoiding data leakage and ensuring proper dataset separation.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import random
from datetime import datetime

def load_json_data(file_path: Path) -> Dict:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data: Dict, file_path: Path):
    """Save JSON data to file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def organize_training_data():
    """Organize data into proper train/validation/test splits"""
    
    data_dir = Path("data")
    organized_dir = data_dir / "organized"
    backup_dir = data_dir / "backup_original"
    
    print("🚀 Starting HybEx-Law Data Organization")
    print("="*50)
    
    # Create directories
    organized_dir.mkdir(exist_ok=True)
    backup_dir.mkdir(exist_ok=True)
    
    # Define split ratios
    TRAIN_RATIO = 0.70  # 70%
    VAL_RATIO = 0.15    # 15% 
    TEST_RATIO = 0.15   # 15%
    
    print(f"📊 Split Ratios: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
    
    # 1. Load main comprehensive dataset (20,000 samples)
    print("\n📁 Loading comprehensive training data...")
    comprehensive_data = load_json_data(data_dir / "comprehensive_legal_training_data.json")
    main_samples = comprehensive_data["samples"]
    print(f"   Loaded {len(main_samples)} samples from comprehensive dataset")
    
    # 2. Load edge cases (should go to validation/test)
    print("\n🔍 Loading edge cases...")
    edge_cases_data = load_json_data(data_dir / "edge_cases_dataset.json")
    edge_samples = edge_cases_data["samples"]
    print(f"   Loaded {len(edge_samples)} edge case samples")
    
    # 3. Shuffle main samples for random splitting
    print("\n🔀 Shuffling main dataset...")
    random.seed(42)  # For reproducibility
    random.shuffle(main_samples)
    
    # 4. Calculate split indices for main dataset
    total_main = len(main_samples)
    train_end = int(total_main * TRAIN_RATIO)
    val_end = train_end + int(total_main * VAL_RATIO)
    
    train_samples = main_samples[:train_end]
    val_samples_main = main_samples[train_end:val_end]
    test_samples_main = main_samples[val_end:]
    
    print(f"   Main dataset split: Train={len(train_samples)}, Val={len(val_samples_main)}, Test={len(test_samples_main)}")
    
    # 5. Split edge cases between validation and test (no training)
    random.shuffle(edge_samples)
    edge_split = len(edge_samples) // 2
    val_edge_samples = edge_samples[:edge_split]
    test_edge_samples = edge_samples[edge_split:]
    
    # 6. Combine validation and test sets with edge cases
    val_samples_final = val_samples_main + val_edge_samples
    test_samples_final = test_samples_main + test_edge_samples
    
    print(f"   Edge cases split: Val={len(val_edge_samples)}, Test={len(test_edge_samples)}")
    print(f"   Final counts: Train={len(train_samples)}, Val={len(val_samples_final)}, Test={len(test_samples_final)}")
    
    # 7. Create organized datasets with proper metadata
    train_data = {
        "metadata": {
            "total_samples": len(train_samples),
            "split_type": "train",
            "source_datasets": ["comprehensive_legal_training_data"],
            "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
            "creation_date": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Training split for HybEx-Law system - includes main comprehensive data only"
        },
        "samples": train_samples
    }
    
    val_data = {
        "metadata": {
            "total_samples": len(val_samples_final),
            "split_type": "validation", 
            "source_datasets": ["comprehensive_legal_training_data", "edge_cases_dataset"],
            "main_samples": len(val_samples_main),
            "edge_case_samples": len(val_edge_samples),
            "creation_date": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Validation split for HybEx-Law system - includes main data + edge cases"
        },
        "samples": val_samples_final
    }
    
    test_data = {
        "metadata": {
            "total_samples": len(test_samples_final),
            "split_type": "test",
            "source_datasets": ["comprehensive_legal_training_data", "edge_cases_dataset"], 
            "main_samples": len(test_samples_main),
            "edge_case_samples": len(test_edge_samples),
            "creation_date": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Test split for HybEx-Law system - includes main data + edge cases"
        },
        "samples": test_samples_final
    }
    
    # 8. Save organized datasets
    print("\n💾 Saving organized datasets...")
    save_json_data(train_data, organized_dir / "train_data.json")
    save_json_data(val_data, organized_dir / "validation_data.json") 
    save_json_data(test_data, organized_dir / "test_data.json")
    
    # 9. Create domain-specific validation sets for specialized evaluation
    print("\n🎯 Creating domain-specific validation sets...")
    domain_validation_sets = {}
    
    # Load existing domain validation sets and move them to organized folder
    domain_files = [
        "legal_aid_validation_set.json",
        "family_law_validation_set.json", 
        "consumer_protection_validation_set.json",
        "employment_law_validation_set.json",
        "fundamental_rights_validation_set.json"
    ]
    
    for domain_file in domain_files:
        if (data_dir / domain_file).exists():
            domain_data = load_json_data(data_dir / domain_file)
            domain_name = domain_file.replace("_validation_set.json", "")
            
            # Add metadata for organization
            domain_data["metadata"]["organized_date"] = datetime.now().isoformat()
            domain_data["metadata"]["usage"] = "domain_specific_validation"
            
            save_json_data(domain_data, organized_dir / "domain_validation" / domain_file)
            domain_validation_sets[domain_name] = len(domain_data["samples"])
    
    # 10. Backup original files that won't be used in training
    print("\n📦 Backing up original files...")
    backup_files = [
        "training_train.json",
        "training_validation.json", 
        "training_test.json",
        "sample_queries.json"
    ]
    
    for backup_file in backup_files:
        if (data_dir / backup_file).exists():
            shutil.copy2(data_dir / backup_file, backup_dir / backup_file)
    
    # 11. Generate summary report
    summary = {
        "organization_date": datetime.now().isoformat(),
        "data_splits": {
            "train": {
                "file": "organized/train_data.json",
                "samples": len(train_samples),
                "sources": ["comprehensive_dataset"]
            },
            "validation": {
                "file": "organized/validation_data.json", 
                "samples": len(val_samples_final),
                "main_samples": len(val_samples_main),
                "edge_cases": len(val_edge_samples),
                "sources": ["comprehensive_dataset", "edge_cases"]
            },
            "test": {
                "file": "organized/test_data.json",
                "samples": len(test_samples_final), 
                "main_samples": len(test_samples_main),
                "edge_cases": len(test_edge_samples),
                "sources": ["comprehensive_dataset", "edge_cases"]
            }
        },
        "domain_validation_sets": domain_validation_sets,
        "total_organized_samples": len(train_samples) + len(val_samples_final) + len(test_samples_final),
        "recommended_training_command": "python -m hybex_system.main train --data-dir data/organized"
    }
    
    save_json_data(summary, organized_dir / "organization_summary.json")
    
    # Print final summary
    print("\n" + "="*50)
    print("✅ DATA ORGANIZATION COMPLETE!")
    print("="*50)
    print(f"📊 Training Data: {len(train_samples):,} samples")
    print(f"📊 Validation Data: {len(val_samples_final):,} samples ({len(val_samples_main):,} main + {len(val_edge_samples):,} edge cases)")
    print(f"📊 Test Data: {len(test_samples_final):,} samples ({len(test_samples_main):,} main + {len(test_edge_samples):,} edge cases)")
    print(f"🎯 Domain-specific validation sets: {len(domain_validation_sets)} domains")
    print(f"📁 Organized data location: {organized_dir}")
    print(f"📁 Backup location: {backup_dir}")
    print("\n🚀 Ready to train with: python -m hybex_system.main train --data-dir data/organized")

if __name__ == "__main__":
    organize_training_data()
