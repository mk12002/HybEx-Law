"""
╔════════════════════════════════════════════════════════════════╗
║           HybEx-Law Model Verification Script                  ║
║     Comprehensive check of all trained models and paths        ║
╚════════════════════════════════════════════════════════════════╝
"""
import os
from pathlib import Path
from datetime import datetime


def format_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:>7.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:>7.2f} TB"


def format_modified_time(path):
    """Get file modification time"""
    try:
        timestamp = path.stat().st_mtime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "Unknown"


def check_model_integrity(model_path):
    """Basic integrity check for PyTorch models"""
    try:
        size_mb = model_path.stat().st_size / (1024*1024)
        
        # Heuristic checks
        if size_mb < 0.1:
            return "⚠️  WARNING: File too small (< 0.1 MB)"
        elif size_mb > 2000:
            return "⚠️  WARNING: File unusually large (> 2 GB)"
        elif size_mb < 1:
            return "⚠️  CAUTION: Small file (might be placeholder)"
        else:
            return "✅ Size looks normal"
    except:
        return "❌ Cannot read file"


# ============================================
# CONFIGURATION
# ============================================
models_dir = Path('models/hybex_system')
knowledge_base_dir = Path('knowledge_base')


print("="*100)
print(" "*30 + "HybEx-Law MODEL VERIFICATION")
print("="*100)
print(f"📂 Base Directory: {models_dir.absolute()}")
print(f"🕐 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)


# Define models with priority search order
models_config = {
    'domain_classifier': {
        'dir': 'domain_classifier',
        'files': ['model.pt', 'domain_classifier.pt', 'best_model.pt'],
        'description': 'Multi-label legal domain classifier',
        'expected_size_range': (400, 450)  # MB
    },
    'eligibility_predictor': {
        'dir': 'eligibility_predictor',
        'files': ['model.pt', 'eligibility_predictor.pt', 'best_model.pt'],
        'description': 'Binary eligibility prediction model',
        'expected_size_range': (400, 450)
    },
    'enhanced_legal_bert': {
        'dir': 'enhanced_legal_bert',
        'files': [
            'enhanced_legal_bert_best.pt',
            'model.pt',
            'enhanced_legal_bert.pt',
            'best_model.pt',
            'enhanced_bert.pt'
        ],
        'description': 'Multi-task enhanced BERT (domain + eligibility)',
        'expected_size_range': (950, 1050)
    },
    'gnn_model': {
        'dir': 'gnn_model',
        'files': ['gnn_model.pt', 'model.pt', 'best_model.pt'],
        'description': 'Knowledge Graph Neural Network',
        'expected_size_range': (1, 5)
    }
}


# ============================================
# MODEL VERIFICATION
# ============================================
results = {
    'found': 0,
    'missing': 0,
    'warnings': 0,
    'total_size_mb': 0
}

model_paths_found = {}

for model_name, config in models_config.items():
    print(f"\n{'─'*100}")
    print(f"🔍 {model_name.upper().replace('_', ' ')}")
    print(f"{'─'*100}")
    print(f"   Description: {config['description']}")
    print(f"   Expected Size: {config['expected_size_range'][0]}-{config['expected_size_range'][1]} MB")
    
    model_dir = models_dir / config['dir']
    print(f"\n   📁 Directory: {model_dir}")
    print(f"   📂 Exists: {'✅ YES' if model_dir.exists() else '❌ NO'}")
    
    if not model_dir.exists():
        print(f"   ❌ ERROR: Model directory not found!")
        results['missing'] += 1
        continue
    
    # List all files in directory
    print(f"\n   📄 Files in directory:")
    all_files = list(model_dir.iterdir())
    
    if not all_files:
        print(f"      ⚠️  Directory is empty!")
        results['warnings'] += 1
        continue
    
    # Check each file
    model_file_found = None
    for file_path in sorted(all_files):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024*1024)
            modified = format_modified_time(file_path)
            is_primary = file_path.name in config['files']
            
            # Determine priority
            priority = ""
            if is_primary:
                try:
                    idx = config['files'].index(file_path.name)
                    priority = f" [Priority: {idx + 1}]"
                    if model_file_found is None:
                        model_file_found = file_path
                except:
                    pass
            
            status_icon = "🎯" if is_primary else "ℹ️"
            print(f"      {status_icon} {file_path.name:<45} {size_mb:>8.2f} MB   Modified: {modified}{priority}")
            
            if is_primary:
                results['total_size_mb'] += size_mb
    
    # Primary model check
    print(f"\n   🎯 Primary Model File:")
    if model_file_found:
        print(f"      ✅ FOUND: {model_file_found.name}")
        model_paths_found[model_name] = model_file_found
        
        # Integrity check
        integrity = check_model_integrity(model_file_found)
        print(f"      {integrity}")
        
        # Size check
        actual_size = model_file_found.stat().st_size / (1024*1024)
        expected_min, expected_max = config['expected_size_range']
        if expected_min <= actual_size <= expected_max:
            print(f"      ✅ Size within expected range ({actual_size:.1f} MB)")
        else:
            print(f"      ⚠️  Size outside expected range: {actual_size:.1f} MB (expected {expected_min}-{expected_max} MB)")
            results['warnings'] += 1
        
        results['found'] += 1
    else:
        print(f"      ❌ NOT FOUND")
        print(f"      Looked for: {', '.join(config['files'][:3])}")
        results['missing'] += 1
    
    # Show alternative files
    print(f"\n   📋 Search priority order:")
    for i, filename in enumerate(config['files'], 1):
        exists = (model_dir / filename).exists()
        status = "✅" if exists else "❌"
        marker = "← FOUND" if exists and (model_dir / filename) == model_file_found else ""
        print(f"      {i}. {status} {filename} {marker}")


# ============================================
# PROLOG KNOWLEDGE BASE CHECK
# ============================================
print(f"\n{'─'*100}")
print(f"🔍 PROLOG KNOWLEDGE BASE")
print(f"{'─'*100}")

prolog_files = [
    'legal_aid_clean_v2.pl',
    'foundational_rules_clean.pl',
    'consumer_protection.pl',
    'employment_law.pl',
    'family_law.pl',
    'cross_domain_rules.pl'
]

prolog_found = 0
for pl_file in prolog_files:
    pl_path = knowledge_base_dir / pl_file
    if pl_path.exists():
        size_kb = pl_path.stat().st_size / 1024
        print(f"   ✅ {pl_file:<40} {size_kb:>6.1f} KB")
        prolog_found += 1
    else:
        print(f"   ❌ {pl_file:<40} NOT FOUND")


# ============================================
# SUMMARY & RECOMMENDATIONS
# ============================================
print(f"\n{'='*100}")
print(" "*40 + "VERIFICATION SUMMARY")
print(f"{'='*100}")

print(f"\n📊 Statistics:")
print(f"   Neural Models Found:    {results['found']}/4")
print(f"   Neural Models Missing:  {results['missing']}/4")
print(f"   Warnings:               {results['warnings']}")
print(f"   Total Model Size:       {results['total_size_mb']:.1f} MB")
print(f"   Prolog Files Found:     {prolog_found}/{len(prolog_files)}")

# Overall status
print(f"\n🎯 Overall Status:")
if results['missing'] == 0 and results['warnings'] == 0:
    print(f"   ✅ EXCELLENT: All models present and validated!")
    print(f"   ✅ Ready for evaluation!")
elif results['missing'] == 0:
    print(f"   ⚠️  GOOD: All models present but some warnings detected")
    print(f"   ✅ Can proceed with evaluation (check warnings)")
elif results['missing'] < 4:
    print(f"   ⚠️  PARTIAL: Some models missing!")
    print(f"   ⚠️  You can run evaluation, but {results['missing']} model(s) will be skipped")
else:
    print(f"   ❌ CRITICAL: No models found!")
    print(f"   ❌ Please run training first: python -m hybex_system.main train --data-dir data")


# ============================================
# ACTIONABLE RECOMMENDATIONS
# ============================================
print(f"\n{'='*100}")
print(" "*35 + "RECOMMENDATIONS")
print(f"{'='*100}")

if results['missing'] == 0:
    print(f"\n✅ ALL MODELS READY!")
    print(f"\n   To run complete evaluation:")
    print(f"   python -m hybex_system.main evaluate_hybrid --test-data data/val_split.json")
    print(f"\n   To run with ablation study (17 combinations):")
    print(f"   python -m hybex_system.main evaluate_hybrid --test-data data/val_split.json --ablation")
else:
    print(f"\n⚠️  MISSING MODELS DETECTED!")
    print(f"\n   To retrain all models:")
    print(f"   python -m hybex_system.main train --data-dir data")
    print(f"\n   Missing models: {[name for name, config in models_config.items() if name not in model_paths_found]}")


# ============================================
# MODEL PATHS FOR REFERENCE
# ============================================
if model_paths_found:
    print(f"\n{'='*100}")
    print(" "*30 + "MODEL PATHS FOR EVALUATION CODE")
    print(f"{'='*100}")
    print(f"\n# Use these paths in your evaluation code:\n")
    for model_name, path in model_paths_found.items():
        print(f"# {model_name}:")
        print(f"{model_name}_path = Path('{path}')  # {path.stat().st_size / (1024*1024):.1f} MB\n")


print(f"{'='*100}\n")
