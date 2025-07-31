"""
Hybrid System Demo - Multi-Domain Legal AI

This script demonstrates the hybrid neural-symbolic capabilities of the
multi-domain legal AI system.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

def check_hybrid_system():
    """Check hybrid system components"""
    print("🔍 CHECKING HYBRID SYSTEM COMPONENTS")
    print("="*50)
    
    # Check neural components
    try:
        from src.core.neural_components import NeuralDomainClassifier, NeuralFactExtractor, HybridConfidenceEstimator
        print("✅ Neural Components: Available")
        neural_available = True
    except ImportError as e:
        print(f"⚠️  Neural Components: Not available ({e})")
        neural_available = False
    
    # Check hybrid domain classifier
    try:
        from src.core.domain_classifier import HybridDomainClassifier
        classifier = HybridDomainClassifier(use_neural=neural_available)
        print("✅ Hybrid Domain Classifier: Initialized")
    except Exception as e:
        print(f"❌ Hybrid Domain Classifier: Failed ({e})")
    
    # Check hybrid pipeline
    try:
        from src.core.multi_domain_pipeline import MultiDomainLegalPipeline
        pipeline = MultiDomainLegalPipeline()
        print("✅ Multi-Domain Pipeline: Initialized")
        
        # Get pipeline status
        status = pipeline.get_pipeline_status()
        print(f"   • Active Processors: {status['total_domains']}")
        print(f"   • Confidence Threshold: {status['confidence_threshold']}")
        
    except Exception as e:
        print(f"❌ Multi-Domain Pipeline: Failed ({e})")
    
    return neural_available

def demonstrate_hybrid_classification():
    """Demonstrate hybrid domain classification"""
    print("\n🤖 HYBRID DOMAIN CLASSIFICATION DEMO")
    print("="*50)
    
    try:
        from src.core.domain_classifier import HybridDomainClassifier
        
        classifier = HybridDomainClassifier(use_neural=True)
        
        # Test queries
        test_queries = [
            "I was fired without notice after 5 years of work",
            "My husband is cruel and I want divorce",
            "Shop sold me defective phone, refusing refund", 
            "Need legal aid as I cannot afford lawyer",
            "Police arrested me without warrant"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            
            # Rule-based classification
            rule_results = classifier.classify_rule_based(query)
            if rule_results:
                print(f"📊 Rule-based: {rule_results[0][0].value} ({rule_results[0][1]:.2f})")
            else:
                print("📊 Rule-based: No match")
            
            # Hybrid classification
            try:
                hybrid_results = classifier.predict_hybrid(query)
                if hybrid_results:
                    top_domain = max(hybrid_results.items(), key=lambda x: x[1])
                    print(f"🧠 Hybrid: {top_domain[0]} ({top_domain[1]:.2f})")
                else:
                    print("🧠 Hybrid: No confident predictions")
            except Exception as e:
                print(f"🧠 Hybrid: Error ({e})")
    
    except Exception as e:
        print(f"❌ Classification demo failed: {e}")

def demonstrate_fact_extraction():
    """Demonstrate hybrid fact extraction"""
    print("\n🔍 HYBRID FACT EXTRACTION DEMO")
    print("="*50)
    
    try:
        from src.core.neural_components import NeuralFactExtractor
        
        extractor = NeuralFactExtractor()
        
        test_query = "I earn Rs 25000 monthly and was fired without notice by my company after 3 years"
        print(f"📝 Query: {test_query}")
        
        # Extract entities
        entities = extractor.extract_entities_neural(test_query)
        print(f"\n🏷️  Extracted Entities:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   • {entity_type}: {[e['text'] for e in entity_list]}")
        
        # Extract legal facts
        facts = extractor.extract_legal_facts_neural(test_query, 'employment_law')
        print(f"\n⚖️  Legal Facts:")
        for fact in facts[:5]:  # Show first 5 facts
            print(f"   • {fact}")
    
    except Exception as e:
        print(f"❌ Fact extraction demo failed: {e}")

def demonstrate_full_pipeline():
    """Demonstrate complete hybrid pipeline"""
    print("\n🚀 COMPLETE HYBRID PIPELINE DEMO")
    print("="*50)
    
    try:
        from src.core.multi_domain_pipeline import MultiDomainLegalPipeline
        
        pipeline = MultiDomainLegalPipeline()
        
        test_query = "I am a poor woman earning Rs 15000 monthly. My husband is cruel and violent, and I also lost my job unfairly. Need legal help."
        
        print(f"📝 Complex Query: {test_query}")
        print("\n🔄 Processing through hybrid pipeline...")
        
        result = pipeline.process_legal_query(test_query)
        
        if 'error' not in result:
            print(f"\n✅ Processing Complete!")
            print(f"🎯 Relevant Domains: {', '.join(result['relevant_domains'])}")
            
            # Show system info
            system_info = result.get('system_info', {})
            if system_info:
                print(f"🤖 Processing Mode: {system_info.get('classification_method', 'unknown')}")
                print(f"🧠 Neural Active: {system_info.get('neural_components_active', False)}")
            
            # Show unified analysis
            unified = result.get('unified_analysis', {})
            if unified:
                print(f"\n🔍 Unified Analysis:")
                print(f"   • Primary Domain: {unified.get('primary_domain', 'N/A')}")
                print(f"   • Complexity: {unified.get('legal_complexity', 'N/A')}")
                print(f"   • Urgency: {unified.get('urgency_level', 'N/A')}")
                print(f"   • Cross-Domain Issues: {len(unified.get('cross_domain_issues', []))}")
            
            # Show recommendations
            recommendations = result.get('recommendations', {})
            immediate = recommendations.get('immediate_actions', [])
            if immediate:
                print(f"\n💡 Top Recommendations:")
                for action in immediate[:3]:
                    print(f"   • {action}")
        else:
            print(f"❌ Pipeline Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Pipeline demo failed: {e}")

def main():
    """Main demo function"""
    print("🏛️  MULTI-DOMAIN LEGAL AI - HYBRID SYSTEM DEMO")
    print("="*60)
    print("🤖 Demonstrating Neural-Symbolic Legal AI Capabilities")
    print("⚖️  Covering 5 Legal Domains with Hybrid Intelligence")
    print("="*60)
    
    # Check system components
    neural_available = check_hybrid_system()
    
    if neural_available:
        print(f"\n🎉 HYBRID MODE: Neural + Symbolic")
    else:
        print(f"\n⚠️  FALLBACK MODE: Rule-based + Symbolic")
        print("   Install torch, transformers, scikit-learn for full hybrid capabilities")
    
    # Run demonstrations
    demonstrate_hybrid_classification()
    demonstrate_fact_extraction()
    demonstrate_full_pipeline()
    
    print("\n" + "="*60)
    print("✅ HYBRID SYSTEM DEMO COMPLETE")
    print("🚀 Ready for production legal query processing!")
    print("="*60)

if __name__ == "__main__":
    main()
