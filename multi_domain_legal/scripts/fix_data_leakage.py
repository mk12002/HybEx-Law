"""
Fix Data Leakage in Pre-Split Files
====================================

This script removes duplicate queries across train/val/test splits
and creates clean, non-overlapping data splits.
"""

import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import shutil

def fix_data_leakage(data_dir):
    """
    Remove duplicate queries and re-split data to prevent leakage
    """
    data_path = Path(data_dir)
    
    # Load all split files
    train_file = data_path / "train_split.json"
    val_file = data_path / "val_split.json"
    test_file = data_path / "test_split.json"
    
    print("Loading pre-split data files...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Original split sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Combine all data
    all_data = train_data + val_data + test_data
    print(f"\nTotal samples loaded: {len(all_data)}")
    
    # Deduplicate by query text, keeping first occurrence
    seen_queries = set()
    unique_data = []
    duplicates_removed = 0
    
    for sample in all_data:
        query = sample.get('query', '')
        if query and query not in seen_queries:
            seen_queries.add(query)
            unique_data.append(sample)
        else:
            duplicates_removed += 1
    
    print(f"\nDeduplication results:")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Unique samples: {len(unique_data)}")
    
    # Re-split the deduplicated data
    # Extract labels for stratification
    labels = [s.get('expected_eligibility', False) for s in unique_data]
    
    # First split: train+val vs test (85% vs 15%)
    train_val, test = train_test_split(
        unique_data,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )
    
    # Second split: train vs val (82.35% vs 17.65% of train_val = 70% vs 15% of total)
    train_val_labels = [s.get('expected_eligibility', False) for s in train_val]
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,  # Proportional split to get 15% of total
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"\nNew split sizes:")
    print(f"  Train: {len(train)} ({len(train)/len(unique_data)*100:.1f}%)")
    print(f"  Val: {len(val)} ({len(val)/len(unique_data)*100:.1f}%)")
    print(f"  Test: {len(test)} ({len(test)/len(unique_data)*100:.1f}%)")
    
    # Verify no overlap
    train_queries = {s['query'] for s in train}
    val_queries = {s['query'] for s in val}
    test_queries = {s['query'] for s in test}
    
    train_val_overlap = train_queries & val_queries
    train_test_overlap = train_queries & test_queries
    val_test_overlap = val_queries & test_queries
    
    print(f"\nOverlap verification:")
    print(f"  Train-Val overlap: {len(train_val_overlap)}")
    print(f"  Train-Test overlap: {len(train_test_overlap)}")
    print(f"  Val-Test overlap: {len(val_test_overlap)}")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\nERROR: Data leakage still exists after deduplication!")
        if train_val_overlap:
            print(f"  Example train-val overlap: {list(train_val_overlap)[0][:100]}...")
        raise ValueError("Data leakage still exists after deduplication!")
    
    print("\n✓ No overlap detected - data is clean!")
    
    # Save deduplicated splits
    backup_dir = data_path / "backup_original_splits"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup original files
    print(f"\nBacking up original files to: {backup_dir}")
    shutil.copy(train_file, backup_dir / "train_split.json")
    shutil.copy(val_file, backup_dir / "val_split.json")
    shutil.copy(test_file, backup_dir / "test_split.json")
    print("  Original files backed up successfully")
    
    # Save new deduplicated splits
    print("\nSaving deduplicated split files...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {train_file}")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {val_file}")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {test_file}")
    
    print("\n" + "="*60)
    print("SUCCESS: Data leakage fixed!")
    print("="*60)
    print(f"Original splits backed up to: {backup_dir}")
    
    return {
        'total_original': len(all_data),
        'duplicates_removed': duplicates_removed,
        'unique_samples': len(unique_data),
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test)
    }

# Run the fix
if __name__ == "__main__":
    data_directory = r"d:\Code_stuff\HybEx-Law\multi_domain_legal\data"
    
    print("="*60)
    print("Data Leakage Fix Script")
    print("="*60)
    print(f"Data directory: {data_directory}\n")
    
    try:
        results = fix_data_leakage(data_directory)
        
        print("\nFinal Results:")
        print(f"  Total original samples: {results['total_original']}")
        print(f"  Duplicates removed: {results['duplicates_removed']}")
        print(f"  Unique samples: {results['unique_samples']}")
        print(f"  New train size: {results['train_size']}")
        print(f"  New val size: {results['val_size']}")
        print(f"  New test size: {results['test_size']}")
        
        print("\nYou can now run training with clean data:")
        print("  python -m hybex_system.main train --data-dir 'data'")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
