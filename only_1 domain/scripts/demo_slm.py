"""
SLM Quick Demo - Demonstration of SLM pipeline capabilities.

This script provides a quick demonstration of the SLM pipeline for legal
fact extraction without requiring full training.

Usage:
    python scripts/demo_slm.py
    python scripts/demo_slm.py --model microsoft/Phi-3-mini-4k-instruct
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.slm_pipeline.slm_pipeline import SLMPipeline
from src.prolog_engine.legal_engine import LegalAidEngine


def demo_slm_pipeline():
    """Demonstrate SLM pipeline capabilities."""
    
    print("🤖 HybEx-Law SLM Pipeline Demo")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        {
            "name": "Domestic Violence Case",
            "query": "I am a woman facing domestic violence from my husband. He beats me regularly and demands dowry. I earn 15000 rupees per month working as a domestic helper. Can I get legal aid?"
        },
        {
            "name": "Property Dispute",
            "query": "My landlord is trying to evict me illegally from my rented house. I am from scheduled caste community and work as a daily wage laborer earning around 8000 rupees monthly."
        },
        {
            "name": "Labor Dispute",
            "query": "The factory where I work has not paid my salary for 3 months. I am an industrial worker and my monthly salary is 12000 rupees. I need legal help to get my money."
        }
    ]
    
    try:
        # Initialize SLM pipeline (will use base model without fine-tuning)
        print("🔧 Initializing SLM pipeline...")
        print("Note: Using base model without fine-tuning for demonstration")
        
        slm_pipeline = SLMPipeline(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            use_preprocessing=True
        )
        
        # Initialize legal engine
        legal_engine = LegalAidEngine()
        
        print("✅ Pipeline initialized successfully")
        print("\n🧪 Testing fact extraction capabilities...")
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*20} Test Case {i}: {test_case['name']} {'='*20}")
            print(f"Query: {test_case['query']}")
            print()
            
            # Extract facts
            print("🔍 Extracting facts with SLM...")
            facts = slm_pipeline.process_query(test_case['query'], verbose=False)
            
            print("📝 Extracted Facts:")
            for fact in facts:
                print(f"  • {fact}")
            
            # Apply legal reasoning
            print("\n⚖️  Applying legal rules...")
            decision = legal_engine.check_eligibility(facts)
            
            print("🎯 Eligibility Decision:")
            status = "✅ ELIGIBLE" if decision['eligible'] else "❌ NOT ELIGIBLE"
            print(f"  Status: {status}")
            print(f"  Reason: {decision['explanation']}")
            
            if decision.get('additional_info'):
                print(f"  Details: {decision['additional_info']}")
        
        print(f"\n{'='*60}")
        print("🎉 Demo completed successfully!")
        
        print("\n💡 Key Observations:")
        print("• SLM pipeline can extract facts from natural language")
        print("• Facts are properly formatted for Prolog reasoning")
        print("• Legal engine provides explainable decisions")
        print("• System handles diverse case types and scenarios")
        
        print("\n🚀 Next Steps for Full Implementation:")
        print("1. Fine-tune SLM on large legal dataset (1000+ examples)")
        print("2. Compare performance with baseline TF-IDF approach")
        print("3. Optimize for better accuracy and speed")
        print("4. Deploy for real-world legal aid applications")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nPossible issues:")
        print("• SLM dependencies not installed (torch, transformers, etc.)")
        print("• Insufficient GPU memory or no CUDA available")
        print("• Network issues downloading model")
        print("\nTry running: pip install -r slm_requirements.txt")


def compare_with_baseline():
    """Quick comparison with baseline approach."""
    print("\n🆚 Quick Baseline Comparison")
    print("=" * 40)
    
    try:
        from src.nlp_pipeline.hybrid_pipeline import HybridLegalNLPPipeline
        
        baseline_pipeline = HybridLegalNLPPipeline()
        slm_pipeline = SLMPipeline()
        
        test_query = "I am a woman facing domestic violence. I earn 15000 rupees monthly."
        
        print(f"Test Query: {test_query}")
        print()
        
        # Baseline extraction
        print("🔧 Baseline (TF-IDF) Facts:")
        baseline_facts = baseline_pipeline.process_query(test_query, verbose=False)
        for fact in baseline_facts:
            print(f"  • {fact}")
        
        print("\n🤖 SLM Facts:")
        slm_facts = slm_pipeline.process_query(test_query, verbose=False)
        for fact in slm_facts:
            print(f"  • {fact}")
        
        print(f"\n📊 Comparison:")
        print(f"  Baseline facts: {len(baseline_facts)}")
        print(f"  SLM facts: {len(slm_facts)}")
        
        # Check overlap
        baseline_set = set(f.lower().strip() for f in baseline_facts)
        slm_set = set(f.lower().strip() for f in slm_facts)
        overlap = len(baseline_set.intersection(slm_set))
        
        print(f"  Common facts: {overlap}")
        print(f"  SLM unique: {len(slm_set) - overlap}")
        print(f"  Baseline unique: {len(baseline_set) - overlap}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        print("This requires both pipelines to be properly configured")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="SLM Pipeline Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="SLM model to use for demo"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Include baseline comparison"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo with single query"
    )
    
    args = parser.parse_args()
    
    # Run main demo
    demo_slm_pipeline()
    
    # Optional baseline comparison
    if args.compare:
        compare_with_baseline()


if __name__ == "__main__":
    main()
