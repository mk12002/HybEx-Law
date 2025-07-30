"""
Data Expansion Script for HybEx-Law

This script demonstrates how to expand your training data using various techniques
to create a more robust and diverse dataset for the legal NLP pipeline.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.sample_data import (
    SAMPLE_QUERIES, expand_sample_data, save_expanded_data,
    ADDITIONAL_SCENARIOS, get_realistic_income_scenarios, get_social_category_variations
)


def demonstrate_data_expansion():
    """Demonstrate various data expansion techniques."""
    
    print("🏭 HybEx-Law Data Expansion Demo")
    print("=" * 50)
    
    # Show current sample data statistics
    print(f"📊 Current Sample Data:")
    print(f"   Total queries: {len(SAMPLE_QUERIES)}")
    
    # Count by case type
    case_types = {}
    eligible_count = 0
    
    for query in SAMPLE_QUERIES:
        # Extract case type from facts
        case_type = None
        for fact in query['expected_facts']:
            if 'case_type(user,' in fact:
                case_type = fact.split('"')[1]
                break
        
        if case_type:
            case_types[case_type] = case_types.get(case_type, 0) + 1
        
        if query['expected_eligible']:
            eligible_count += 1
    
    print(f"   Eligible cases: {eligible_count} ({eligible_count/len(SAMPLE_QUERIES):.1%})")
    print(f"   Case types:")
    for case_type, count in case_types.items():
        print(f"     {case_type}: {count}")
    
    # Expand the data
    print(f"\n🔧 Expanding dataset...")
    try:
        expanded_data = expand_sample_data(multiplier=3)
        print(f"   Expanded to: {len(expanded_data)} queries")
        
        # Save expanded data
        splits = save_expanded_data(expanded_data)
        
        print(f"\n📋 Data Splits Created:")
        for split_name, split_data in splits.items():
            print(f"   {split_name}: {len(split_data)} samples")
        
    except ImportError as e:
        print(f"⚠️  Data generator not available: {e}")
        print("Using template-based expansion instead...")
        
        # Use template-based expansion as fallback
        template_expanded = create_template_variations()
        print(f"   Created {len(template_expanded)} template variations")
        
        # Save template-based data
        import json
        with open("data/template_expanded_data.json", 'w', encoding='utf-8') as f:
            json.dump(template_expanded, f, indent=2, ensure_ascii=False)
        
        print("💾 Saved template-based expanded data")


def create_template_variations():
    """Create variations using templates and the existing data."""
    
    variations = []
    
    # Get variation components
    income_scenarios = get_realistic_income_scenarios()
    social_variations = get_social_category_variations()
    
    # Create variations for each original query
    for i, original in enumerate(SAMPLE_QUERIES):
        base_id = original['id']
        
        # Create income variations
        for income_category, income_phrases in income_scenarios.items():
            for j, income_phrase in enumerate(income_phrases[:2]):  # Limit to 2 per category
                
                # Modify the original query
                new_query = _create_income_variation(original['query'], income_phrase)
                
                # Create new expected facts
                new_facts = _modify_facts_for_income(original['expected_facts'], income_category)
                
                # Determine new eligibility
                new_eligible = _determine_eligibility_from_facts(new_facts)
                
                variation = {
                    'id': len(variations) + len(SAMPLE_QUERIES) + 1,
                    'query': new_query,
                    'expected_facts': new_facts,
                    'expected_eligible': new_eligible,
                    'source': f'income_variation_of_{base_id}',
                    'variation_type': 'income_modification'
                }
                
                variations.append(variation)
    
    return variations


def _create_income_variation(original_query: str, income_phrase: str) -> str:
    """Create a variation by modifying income information."""
    
    # Simple approach: append income information if not present
    # More sophisticated approach would replace existing income mentions
    
    if any(word in original_query.lower() for word in ['earn', 'income', 'salary', 'rupees']):
        # Income already mentioned, return original
        return original_query
    else:
        # Add income information
        sentences = original_query.split('.')
        # Insert income info before the last sentence (usually the request for help)
        if len(sentences) > 1:
            sentences.insert(-1, f" I {income_phrase}.")
            return '.'.join(sentences)
        else:
            return f"{original_query} I {income_phrase}."


def _modify_facts_for_income(original_facts: List[str], income_category: str) -> List[str]:
    """Modify facts based on income category."""
    
    new_facts = []
    
    for fact in original_facts:
        if 'income_monthly(user,' in fact:
            # Replace with new income based on category
            if income_category == 'unemployed':
                new_facts.append('income_monthly(user, 0)')
            elif income_category == 'low_income':
                new_facts.append('income_monthly(user, 12000)')
            elif income_category == 'moderate_income':
                new_facts.append('income_monthly(user, 25000)')
            elif income_category == 'above_threshold':
                new_facts.append('income_monthly(user, 60000)')
        else:
            new_facts.append(fact)
    
    # Add income fact if not present
    if not any('income_monthly(user,' in fact for fact in new_facts):
        if income_category == 'unemployed':
            new_facts.append('income_monthly(user, 0)')
        elif income_category == 'low_income':
            new_facts.append('income_monthly(user, 12000)')
        elif income_category == 'moderate_income':
            new_facts.append('income_monthly(user, 25000)')
        elif income_category == 'above_threshold':
            new_facts.append('income_monthly(user, 60000)')
    
    return new_facts


def _determine_eligibility_from_facts(facts: List[str]) -> bool:
    """Determine eligibility based on facts."""
    
    # Check for categorical eligibility
    categorical_eligible = any(
        'true' in fact for fact in facts 
        if any(category in fact for category in [
            'is_woman', 'is_child', 'is_sc_st', 'is_disabled',
            'is_industrial_workman', 'is_in_custody', 'is_disaster_victim'
        ])
    )
    
    # Check for income eligibility
    income_eligible = False
    for fact in facts:
        if 'income_monthly(user,' in fact:
            try:
                income = int(fact.split(',')[1].strip(' ).'))
                income_eligible = income <= 25000
            except:
                pass
    
    # Check for excluded case types
    excluded = any(
        excluded_type in fact for fact in facts
        for excluded_type in ['defamation', 'business_dispute', 'election_offense']
        if 'case_type(user,' in fact
    )
    
    return (categorical_eligible or income_eligible) and not excluded


def show_data_sources_guide():
    """Show guidance on authentic legal data sources."""
    
    print("\n🏛️  Authentic Legal Data Sources")
    print("=" * 50)
    
    sources = {
        "🇮🇳 Official Government Sources": [
            "NALSA (nalsa.gov.in) - Legal aid schemes and guidelines",
            "Supreme Court of India - Public judgments and case law",
            "High Court websites - State-specific legal information",
            "District Court records - Local case examples",
            "Legislative.gov.in - Acts and legal frameworks"
        ],
        
        "📚 Academic & Research Sources": [
            "Law university case study databases",
            "Legal research journals and publications",
            "Bar council publications and reports",
            "Legal aid organization annual reports",
            "Pro bono legal clinic case studies"
        ],
        
        "🤝 Partnership Opportunities": [
            "Legal aid organizations (anonymized cases)",
            "Law schools (student clinic cases)",
            "Bar associations (member contributed cases)",
            "NGOs working on access to justice",
            "Government legal aid boards"
        ],
        
        "⚖️  Ethical Considerations": [
            "Always anonymize personal information",
            "Obtain proper permissions for data use",
            "Respect confidentiality and privacy",
            "Follow data protection regulations",
            "Get ethical review for research use"
        ]
    }
    
    for category, items in sources.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")


def main():
    """Run the data expansion demonstration."""
    
    # Import type annotations
    from typing import List, Dict, Any
    
    # Demonstrate data expansion
    demonstrate_data_expansion()
    
    # Show guidance on real data sources
    show_data_sources_guide()
    
    print("\n" + "=" * 50)
    print("💡 Next Steps for Data Collection:")
    print("1. Set up partnerships with legal aid organizations")
    print("2. Create data collection agreements")
    print("3. Develop anonymization procedures")
    print("4. Build web scrapers for public legal databases")
    print("5. Implement quality control and validation")
    print("6. Create continuous data update procedures")
    
    print("\n🎯 Quality Targets:")
    print("• 1000+ training queries across all case types")
    print("• 200+ validation queries for hyperparameter tuning")
    print("• 300+ test queries for final evaluation")
    print("• Balanced representation of eligible/ineligible cases")
    print("• Diverse linguistic expressions and demographics")


if __name__ == "__main__":
    main()
