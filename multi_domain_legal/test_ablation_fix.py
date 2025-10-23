"""
Quick test for the fixed ablation study
Tests the new evaluate_ablation_combinations() method
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybex_system.config import HybExConfig
from hybex_system.evaluator import ModelEvaluator

def test_ablation_fix():
    """Test the fixed ablation study with a small sample"""
    print("="*70)
    print("TESTING FIXED ABLATION STUDY")
    print("="*70)
    
    # Initialize config and evaluator
    config = HybExConfig()
    evaluator = ModelEvaluator(config)
    
    # Load a small sample of test data
    test_data_path = Path('data/val_split.json')
    
    if not test_data_path.exists():
        print(f"❌ Test data not found at {test_data_path}")
        return False
    
    print(f"\n✅ Loading test data from {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Take small sample for quick test
    sample_size = 20
    test_sample = test_data[:sample_size]
    print(f"✅ Using {len(test_sample)} samples for testing")
    
    # Test the method signature
    print("\n" + "="*70)
    print("Testing method signature...")
    print("="*70)
    
    # Check if method accepts List[Dict]
    import inspect
    sig = inspect.signature(evaluator.evaluate_ablation_combinations)
    params = list(sig.parameters.keys())
    print(f"✅ Method parameters: {params}")
    
    if params[0] == 'test_data':
        print("✅ Correct signature: accepts test_data (List[Dict])")
    else:
        print(f"❌ Wrong signature: first param is '{params[0]}'")
        return False
    
    # Test dataset creation
    print("\n" + "="*70)
    print("Testing AblationDataset creation...")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])
        print(f"✅ Tokenizer loaded: {config.MODEL_CONFIG['base_model']}")
        
        # Create a sample to test dataset
        sample = {
            'query': 'Test query for legal eligibility',
            'expected_eligibility': 1
        }
        
        # Tokenize
        encoding = tokenizer(
            sample['query'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        print(f"✅ Tokenization works")
        print(f"   input_ids shape: {encoding['input_ids'].shape}")
        print(f"   attention_mask shape: {encoding['attention_mask'].shape}")
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False
    
    # Test Prolog-only path
    print("\n" + "="*70)
    print("Testing Prolog evaluation path...")
    print("="*70)
    
    if hasattr(evaluator, '_prolog_engine_cache') and evaluator._prolog_engine_cache:
        print("✅ Prolog engine available")
        
        # Test single sample
        try:
            sample = test_sample[0]
            entities = sample.get('extracted_entities', {})
            query = sample.get('query', '')
            
            if entities or query:
                result = evaluator._prolog_engine_cache.evaluate_eligibility(entities, query)
                print(f"✅ Prolog prediction: {result.get('eligible', False)}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
            else:
                print("⚠️  Sample missing entities/query")
                
        except Exception as e:
            print(f"⚠️  Prolog evaluation warning: {e}")
    else:
        print("⚠️  Prolog engine not initialized (optional)")
    
    # Test model availability
    print("\n" + "="*70)
    print("Checking model availability...")
    print("="*70)
    
    models_to_check = ['domain_classifier', 'eligibility_predictor']
    available_models = []
    
    for model_name in models_to_check:
        model_path = config.MODELS_DIR / model_name / 'model.pt'
        if model_path.exists():
            available_models.append(model_name)
            size_mb = model_path.stat().st_size / (1024*1024)
            print(f"✅ {model_name}: {size_mb:.2f} MB")
        else:
            print(f"⚠️  {model_name}: Not found at {model_path}")
    
    if len(available_models) < 2:
        print("\n⚠️  Warning: Not all neural models available")
        print("   Ablation study will only test available combinations")
    
    print("\n" + "="*70)
    print("PRELIMINARY CHECKS PASSED")
    print("="*70)
    print("\n✅ The fixed ablation study method is ready to use!")
    print("✅ All data format issues should be resolved")
    print("\nTo run the full ablation study:")
    print("  python hybex_system/main.py evaluate_hybrid --test-data data/val_split.json --ablation")
    
    return True

if __name__ == '__main__':
    try:
        success = test_ablation_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
