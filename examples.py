"""
Example usage and testing script for the HybEx-Law system.

This script demonstrates how to use the complete pipeline and
provides examples for testing different components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.nlp_pipeline.hybrid_pipeline import HybridLegalNLPPipeline
from src.prolog_engine.legal_engine import LegalAidEngine
from src.evaluation.evaluator import HybExEvaluator
from data.sample_data import SAMPLE_QUERIES


def demo_single_query():
    """Demonstrate processing a single query."""
    print("🔍 HybEx-Law Single Query Demo")
    print("=" * 50)
    
    # Initialize components
    pipeline = HybridLegalNLPPipeline()
    legal_engine = LegalAidEngine()
    
    # Example query
    query = "I am a woman. My husband beats me and demands dowry. I earn 20000 rupees per month. Can I get legal aid?"
    
    print(f"Query: {query}")
    print()
    
    # Process query
    print("📝 Extracting facts...")
    facts = pipeline.process_query(query, verbose=True)
    
    print("\n🧠 Applying legal rules...")
    decision = legal_engine.check_eligibility(facts)
    
    print("\n🎯 Final Decision:")
    print(f"  Eligible: {'✅ YES' if decision['eligible'] else '❌ NO'}")
    print(f"  Reason: {decision['explanation']}")
    
    if decision.get('additional_info'):
        print(f"  Additional Info: {decision['additional_info']}")


def demo_batch_evaluation():
    """Demonstrate batch evaluation on sample data."""
    print("\n📊 HybEx-Law Batch Evaluation Demo")
    print("=" * 50)
    
    evaluator = HybExEvaluator()
    
    # Use first 5 sample queries for demo
    test_data = SAMPLE_QUERIES[:5]
    
    print(f"Evaluating on {len(test_data)} sample queries...")
    results = evaluator.evaluate_pipeline(test_data)
    
    print("\n📈 Results Summary:")
    metrics = results['aggregate_metrics']
    print(f"  Task Success Rate: {metrics['task_success_rate']:.2%}")
    print(f"  Average Fact F1: {metrics['avg_fact_f1']:.3f}")
    print(f"  Average Processing Time: {metrics['avg_processing_time']:.3f}s")
    
    print("\n📋 Individual Results:")
    for result in results['individual_results']:
        status = "✅" if result.task_success else "❌"
        print(f"  Query {result.query_id}: {status} (F1: {result.fact_f1:.3f})")


def demo_component_testing():
    """Demonstrate testing individual components."""
    print("\n🧪 Component Testing Demo")
    print("=" * 50)
    
    # Test text preprocessing
    from src.utils.text_preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor()
    
    sample_text = "I can't afford a lawyer. I earn Rs. 15,000 per month."
    processed = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    
    # Test income extraction
    from src.extractors.income_extractor import IncomeExtractor
    income_extractor = IncomeExtractor()
    
    income_query = "I lost my job and have no income. I used to earn 25000 rupees monthly."
    income_facts = income_extractor.extract(income_query, income_query)
    print(f"\nIncome extraction from: '{income_query}'")
    print(f"Extracted facts: {income_facts}")
    
    # Test case type classification
    from src.extractors.case_type_classifier import CaseTypeClassifier
    case_classifier = CaseTypeClassifier()
    
    case_query = "My landlord is trying to evict me from my apartment"
    case_type = case_classifier.classify_case_type(case_query)
    print(f"\nCase type from: '{case_query}'")
    print(f"Classified as: {case_type}")
    
    # Test social category extraction
    from src.extractors.social_category_extractor import SocialCategoryExtractor
    social_extractor = SocialCategoryExtractor()
    
    social_query = "I am a woman from scheduled caste community"
    social_facts = social_extractor.extract(social_query, social_query)
    print(f"\nSocial categories from: '{social_query}'")
    print(f"Extracted facts: {social_facts}")


def demo_knowledge_base_testing():
    """Demonstrate testing the Prolog knowledge base."""
    print("\n⚖️  Knowledge Base Testing Demo")
    print("=" * 50)
    
    legal_engine = LegalAidEngine()
    
    print("Testing knowledge base with sample cases...")
    test_results = legal_engine.test_knowledge_base()
    
    for test_name, result in test_results.items():
        status = "✅" if result else "❌" 
        print(f"  {test_name}: {status}")
    
    # Test a custom case
    print("\nTesting custom case:")
    custom_facts = [
        'applicant(custom_user)',
        'income_monthly(custom_user, 5000)',
        'is_woman(custom_user, true)',
        'case_type(custom_user, "domestic_violence")'
    ]
    
    decision = legal_engine.check_eligibility(custom_facts)
    print(f"  Facts: {custom_facts}")
    print(f"  Decision: {'Eligible' if decision['eligible'] else 'Not Eligible'}")
    print(f"  Reason: {decision['explanation']}")


def main():
    """Run all demonstrations."""
    print("🏛️  HybEx-Law: Legal Aid Eligibility System")
    print("=" * 60)
    print("Comprehensive demonstration of the hybrid NLP framework")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demo_single_query()
        demo_batch_evaluation()
        demo_component_testing()
        demo_knowledge_base_testing()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install SWI-Prolog for full functionality")
        print("3. Run individual components or full pipeline")
        print("4. Develop additional training data")
        print("5. Train machine learning components")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This is expected if dependencies are not installed.")
        print("Please install requirements and try again.")


if __name__ == "__main__":
    main()
