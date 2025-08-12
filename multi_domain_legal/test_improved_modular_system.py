#!/usr/bin/env python3
"""
Test the modular system with a focus on entity extraction and Prolog reasoning.
Updated to address entity extraction failures.
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add project to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybex_system.main import HybExLawSystem

def test_entity_extraction():
    """Test entity extraction separately first"""
    print("🧪 Testing Entity Extraction")
    print("=" * 40)
    
    try:
        system = HybExLawSystem()
        data_processor = system.data_processor
        
        test_queries = [
            "I am a 75-year-old senior citizen with an annual income of Rs. 15,000. Am I eligible for legal aid?",
            "I have a family dispute and my monthly income is 20,000 rupees. I am a woman.",
            "I am from SC category with annual income of 200000. Am I eligible for legal aid?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            entities = data_processor.extract_entities(query)
            print(f"✅ Extracted: {entities}")
            
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        import traceback
        traceback.print_exc()

def test_modular_system():
    """
    Test the modular system with a focus on entity extraction and Prolog reasoning.
    """
    print("\n🔧 Initializing HybEx-Law system...")
    try:
        system = HybExLawSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return

    print("\n" + "=" * 80)
    print("🧪 TESTING MODULAR PROLOG SYSTEM")
    print("=" * 80 + "\n")

    # --- TEST CASE 1: LEGAL_AID ---
    print("📊 TEST CASE 1: LEGAL_AID (Senior Citizen)")
    query = "I am a 75-year-old senior citizen with an annual income of Rs. 15,000. Am I eligible for legal aid?"
    
    print(f"Query: {query}")
    print("-" * 60)
    try:
        result = system.predict_legal_eligibility(query)
        
        print("\n✅ RESULTS:")
        print(f"   Eligible: {result.get('prolog_reasoning', {}).get('eligible', 'Unknown')}")
        print(f"   Confidence: {result.get('prolog_reasoning', {}).get('confidence', 0.0):.2f}")
        print(f"   Method: {result.get('prolog_reasoning', {}).get('method', 'Unknown')}")
        print(f"   Primary Reason: {result.get('prolog_reasoning', {}).get('primary_reason', 'Unknown')}")
        print(f"   Reasoning: {str(result.get('prolog_reasoning', {}).get('detailed_reasoning', 'Unknown'))}")
        
        if result.get('prolog_reasoning', {}).get('eligible'):
            print("🎉 SUCCESS: System correctly identified eligibility!")
        else:
            print("❌ FAILED: System did not identify eligibility as expected.")
        
    except Exception as e:
        print(f"❌ ERROR running legal aid test: {e}")
        import traceback
        traceback.print_exc()

    # --- TEST CASE 2: FAMILY_LAW ---
    print("\n\n📊 TEST CASE 2: FAMILY_LAW")
    query_family = "I have a family dispute and my monthly income is 20,000 rupees. I am a woman."

    print(f"Query: {query_family}")
    print("-" * 60)
    try:
        result = system.predict_legal_eligibility(query_family)
        
        print("\n✅ RESULTS:")
        print(f"   Eligible: {result.get('prolog_reasoning', {}).get('eligible', 'Unknown')}")
        print(f"   Confidence: {result.get('prolog_reasoning', {}).get('confidence', 0.0):.2f}")
        print(f"   Method: {result.get('prolog_reasoning', {}).get('method', 'Unknown')}")
        print(f"   Primary Reason: {result.get('prolog_reasoning', {}).get('primary_reason', 'Unknown')}")
        print(f"   Reasoning: {str(result.get('prolog_reasoning', {}).get('detailed_reasoning', 'Unknown'))}")
        
        if result.get('prolog_reasoning', {}).get('eligible'):
            print("🎉 SUCCESS: System correctly identified eligibility!")
        else:
            print("❌ FAILED: System did not identify eligibility as expected.")

    except Exception as e:
        print(f"❌ ERROR running family law test: {e}")
        import traceback
        traceback.print_exc()

    # --- TEST CASE 3: SC CATEGORY ---
    print("\n\n📊 TEST CASE 3: SC CATEGORY")
    query_sc = "I am from SC category with annual income of 200000. Am I eligible for legal aid?"

    print(f"Query: {query_sc}")
    print("-" * 60)
    try:
        result = system.predict_legal_eligibility(query_sc)
        
        print("\n✅ RESULTS:")
        print(f"   Eligible: {result.get('prolog_reasoning', {}).get('eligible', 'Unknown')}")
        print(f"   Confidence: {result.get('prolog_reasoning', {}).get('confidence', 0.0):.2f}")
        print(f"   Method: {result.get('prolog_reasoning', {}).get('method', 'Unknown')}")
        print(f"   Primary Reason: {result.get('prolog_reasoning', {}).get('primary_reason', 'Unknown')}")
        print(f"   Reasoning: {str(result.get('prolog_reasoning', {}).get('detailed_reasoning', 'Unknown'))}")
        
        if result.get('prolog_reasoning', {}).get('eligible'):
            print("🎉 SUCCESS: System correctly identified eligibility!")
        else:
            print("❌ FAILED: System did not identify eligibility as expected.")

    except Exception as e:
        print(f"❌ ERROR running SC category test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("🏁 MODULAR SYSTEM TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    # First test entity extraction separately
    test_entity_extraction()
    
    # Then test the full system
    test_modular_system()
    #!/usr/bin/env python3
"""
Test the modular system with a focus on entity extraction and Prolog reasoning.
Updated to address entity extraction failures.
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add project to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybex_system.main import HybExLawSystem

def test_entity_extraction():
    """Test entity extraction separately first"""
    print("🧪 Testing Entity Extraction")
    print("=" * 40)
    
    try:
        system = HybExLawSystem()
        data_processor = system.data_processor
        
        test_queries = [
            "I am a 75-year-old senior citizen with an annual income of Rs. 15,000. Am I eligible for legal aid?",
            "I have a family dispute and my monthly income is 20,000 rupees. I am a woman.",
            "I am from SC category with annual income of 200000. Am I eligible for legal aid?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            entities = data_processor.extract_entities(query)
            print(f"✅ Extracted: {entities}")
            
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        import traceback
        traceback.print_exc()

def test_modular_system():
    """
    Test the modular system with a focus on entity extraction and Prolog reasoning.
    """
    print("\n🔧 Initializing HybEx-Law system...")
    try:
        system = HybExLawSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return

    print("\n" + "=" * 80)
    print("🧪 TESTING MODULAR PROLOG SYSTEM")
    print("=" * 80 + "\n")

    # --- TEST CASE 1: LEGAL_AID ---
    print("📊 TEST CASE 1: LEGAL_AID (Senior Citizen)")
    query = "I am a 75-year-old senior citizen with an annual income of Rs. 15,000. Am I eligible for legal aid?"
    
    print(f"Query: {query}")
    print("-" * 60)
    try:
        result = system.predict_legal_eligibility(query)
        
        print("\n✅ RESULTS:")
        print(f"   Eligible: {result.get('prolog_reasoning', {}).get('eligible', 'Unknown')}")
        print(f"   Confidence: {result.get('prolog_reasoning', {}).get('confidence', 0.0):.2f}")
        print(f"   Method: {result.get('prolog_reasoning', {}).get('method', 'Unknown')}")
        print(f"   Primary Reason: {result.get('prolog_reasoning', {}).get('primary_reason', 'Unknown')}")
        print(f"   Reasoning: {str(result.get('prolog_reasoning', {}).get('detailed_reasoning', 'Unknown'))}")
        
        if result.get('prolog_reasoning', {}).get('eligible'):
            print("🎉 SUCCESS: System correctly identified eligibility!")
        else:
            print("❌ FAILED: System did not identify eligibility as expected.")
        
    except Exception as e:
        print(f"❌ ERROR running legal aid test: {e}")
        import traceback
        traceback.print_exc()

    # --- TEST CASE 2: FAMILY_LAW ---
    print("\n\n📊 TEST CASE 2: FAMILY_LAW")
    query_family = "I have a family dispute and my monthly income is 20,000 rupees. I am a woman."

    print(f"Query: {query_family}")
    print("-" * 60)
    try:
        result = system.predict_legal_eligibility(query_family)
        
        print("\n✅ RESULTS:")
        print(f"   Eligible: {result.get('prolog_reasoning', {}).get('eligible', 'Unknown')}")
        print(f"   Confidence: {result.get('prolog_reasoning', {}).get('confidence', 0.0):.2f}")
        print(f"   Method: {result.get('prolog_reasoning', {}).get('method', 'Unknown')}")
        print(f"   Primary Reason: {result.get('prolog_reasoning', {}).get('primary_reason', 'Unknown')}")
        print(f"   Reasoning: {str(result.get('prolog_reasoning', {}).get('detailed_reasoning', 'Unknown'))}")
        
        if result.get('prolog_reasoning', {}).get('eligible'):
            print("🎉 SUCCESS: System correctly identified eligibility!")
        else:
            print("❌ FAILED: System did not identify eligibility as expected.")

    except Exception as e:
        print(f"❌ ERROR running family law test: {e}")
        import traceback
        traceback.print_exc()

    # --- TEST CASE 3: SC CATEGORY ---
    print("\n\n📊 TEST CASE 3: SC CATEGORY")
    query_sc = "I am from SC category with annual income of 200000. Am I eligible for legal aid?"

    print(f"Query: {query_sc}")
    print("-" * 60)
    try:
        result = system.predict_legal_eligibility(query_sc)
        
        print("\n✅ RESULTS:")
        print(f"   Eligible: {result.get('prolog_reasoning', {}).get('eligible', 'Unknown')}")
        print(f"   Confidence: {result.get('prolog_reasoning', {}).get('confidence', 0.0):.2f}")
        print(f"   Method: {result.get('prolog_reasoning', {}).get('method', 'Unknown')}")
        print(f"   Primary Reason: {result.get('prolog_reasoning', {}).get('primary_reason', 'Unknown')}")
        print(f"   Reasoning: {str(result.get('prolog_reasoning', {}).get('detailed_reasoning', 'Unknown'))}")
        
        if result.get('prolog_reasoning', {}).get('eligible'):
            print("🎉 SUCCESS: System correctly identified eligibility!")
        else:
            print("❌ FAILED: System did not identify eligibility as expected.")

    except Exception as e:
        print(f"❌ ERROR running SC category test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("🏁 MODULAR SYSTEM TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    # First test entity extraction separately
    test_entity_extraction()
    
    # Then test the full system
    test_modular_system()