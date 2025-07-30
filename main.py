"""
Main entry point for the HybEx-Law system.
Demonstrates the complete pipeline from natural language query to legal aid eligibility decision.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.nlp_pipeline.hybrid_pipeline import HybridLegalNLPPipeline
from src.prolog_engine.legal_engine import LegalAidEngine


def main():
    parser = argparse.ArgumentParser(description="HybEx-Law Legal Aid Eligibility System")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True,
        help="Natural language query describing the legal situation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed processing steps"
    )
    
    args = parser.parse_args()
    
    print("🏛️  HybEx-Law: Legal Aid Eligibility System")
    print("=" * 50)
    print(f"Query: {args.query}")
    print()
    
    try:
        # Initialize the pipeline
        if args.verbose:
            print("Initializing NLP pipeline...")
        
        pipeline = HybridLegalNLPPipeline()
        legal_engine = LegalAidEngine()
        
        # Process the query
        if args.verbose:
            print("Processing natural language query...")
        
        extracted_facts = pipeline.process_query(args.query, verbose=args.verbose)
        
        print("Extracted Legal Facts:")
        for fact in extracted_facts:
            print(f"  • {fact}")
        print()
        
        # Get eligibility decision
        if args.verbose:
            print("Applying legal rules...")
        
        decision = legal_engine.check_eligibility(extracted_facts)
        
        print("🔍 Eligibility Decision:")
        print(f"  Status: {'✅ ELIGIBLE' if decision['eligible'] else '❌ NOT ELIGIBLE'}")
        print(f"  Reason: {decision['explanation']}")
        
        if decision.get('additional_info'):
            print(f"  Details: {decision['additional_info']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
