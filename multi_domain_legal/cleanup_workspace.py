#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files from HybEx-Law workspace
"""

import os
import shutil
from pathlib import Path

def cleanup_workspace():
    """Remove unnecessary files and keep only essential components"""
    
    base_dir = Path(__file__).parent
    print(f"Cleaning up workspace: {base_dir}")
    
    # Files to delete (test files, debug scripts, duplicates)
    files_to_delete = [
        # Test files
        "analyze_rules.py",
        "debug_prolog_facts.py", 
        "debug_reasoning_output.py",
        "detailed_rules_report.py",
        "fixes_summary.py",
        "legal_integration.py",
        "quick_predicate_test.py",
        "quick_test.py",
        "simple_prolog_test.py",
        "test_clean_prolog_direct.py",
        "test_clean_rules.py",
        "test_corrected_prolog.py",
        "test_corrected_rules.py",
        "test_debug_timeout.py",
        "test_direct_prolog.py",
        "test_extraction_fix.py",
        "test_fact_generation.py",
        "test_final_fixed_system.py",
        "test_fixed_evaluation.py",
        "test_fixed_senior.py",
        "test_hybrid_evaluation.py",
        "test_hybrid_system.py",
        "test_imports.py",
        "test_minimal_direct.py",
        "test_minimal_prolog.py",
        "test_modular_generation.py",
        "test_modular_system.py",
        "test_neural_loading.py",
        "test_primary_reason.py",
        "test_prolog_direct.py",
        "test_prolog_fix.py",
        "test_quick_senior_citizen.py",
        "test_regex_fixes.py",
        "test_simple_facts.py",
        "test_simple_prolog.py",
        "test_simple_system.py",
        "test_ultra_minimal.py",
        
        # Old/unnecessary config
        "config.json",
        "scraper_requirements.txt"
    ]
    
    # Knowledge base files to keep (remove others)
    kb_files_to_keep = [
        "multi_domain_rules.py",
        "foundational_rules_clean.pl", 
        "legal_aid_clean_v2.pl",
        "consumer_protection.pl",
        "cross_domain_rules.pl",
        "family_law.pl",
        "employment_law.pl",
        "__pycache__"  # Will be cleaned separately
    ]
    
    # Delete unnecessary files from root
    deleted_count = 0
    for file_name in files_to_delete:
        file_path = base_dir / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✓ Deleted: {file_name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Failed to delete {file_name}: {e}")
    
    # Clean up knowledge_base directory
    kb_dir = base_dir / "knowledge_base"
    if kb_dir.exists():
        for item in kb_dir.iterdir():
            if item.name not in kb_files_to_keep:
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"✓ Deleted KB file: {item.name}")
                        deleted_count += 1
                    elif item.is_dir() and item.name == "__pycache__":
                        shutil.rmtree(item)
                        print(f"✓ Deleted KB dir: {item.name}")
                        deleted_count += 1
                except Exception as e:
                    print(f"✗ Failed to delete KB item {item.name}: {e}")
    
    # Clean up __pycache__ directories
    for pycache_dir in base_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"✓ Deleted __pycache__: {pycache_dir.relative_to(base_dir)}")
            deleted_count += 1
        except Exception as e:
            print(f"✗ Failed to delete __pycache__ {pycache_dir}: {e}")
    
    # Create a README for the cleaned workspace
    readme_content = """# HybEx-Law: Hybrid Neural-Symbolic Legal AI System

## Directory Structure

### Core System Components
- `hybex_system/` - Main system modules
  - `main.py` - Primary system interface
  - `prolog_engine.py` - Symbolic reasoning engine  
  - `neural_models.py` - Neural network components
  - `data_processor.py` - Data preprocessing
  - `evaluator.py` - Model evaluation
  - `trainer.py` - Model training
  - `config.py` - System configuration

### Knowledge Base
- `knowledge_base/` - Legal knowledge and rules
  - `multi_domain_rules.py` - Main rule definitions
  - `foundational_rules_clean.pl` - Core Prolog predicates
  - `legal_aid_clean_v2.pl` - Legal aid specific rules

### Models & Data
- `models/hybex_system/` - Trained neural models
  - `domain_classifier/` - Domain classification model
  - `eligibility_predictor/` - Eligibility prediction model
- `data/` - Training and evaluation datasets

### Supporting
- `scripts/` - Utility scripts
- `logs/` - System logs
- `results/` - Evaluation results
- `requirements.txt` - Python dependencies

## Usage

```python
from hybex_system.main import HybExLawSystem

# Initialize system
system = HybExLawSystem()

# Predict legal eligibility
result = system.predict_legal_eligibility("I need legal aid for custody case")
print(f"Eligible: {result['final_decision']['eligible']}")
```

## Features
- Hybrid neural-symbolic reasoning
- Multi-domain legal knowledge
- Real-time Prolog inference  
- Transformer-based neural models
- Comprehensive evaluation framework
"""
    
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Updated README.md")
    
    print(f"\n🎉 Cleanup completed! Deleted {deleted_count} items.")
    print("\n📁 Essential files remaining:")
    print("   - hybex_system/ (core system)")
    print("   - knowledge_base/ (3 essential files)")
    print("   - models/ (trained neural models)")
    print("   - data/ (datasets)")
    print("   - scripts/ (utilities)")
    print("   - requirements.txt")
    print("   - README.md")

if __name__ == "__main__":
    cleanup_workspace()
