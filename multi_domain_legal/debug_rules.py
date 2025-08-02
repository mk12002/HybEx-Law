#!/usr/bin/env python3
"""Debug script to check rule loading"""

from hybex_system.prolog_engine import PrologEngine
from hybex_system.config import HybExConfig

def main():
    print("🔍 Debugging Rule Loading...")
    print("=" * 50)
    
    config = HybExConfig()
    engine = PrologEngine(config)
    
    print(f"Total rules loaded: {sum(len(rules) for rules in engine.legal_rules.values())}")
    print("\nRule counts by category:")
    
    for category, rules in engine.legal_rules.items():
        print(f"  {category}: {len(rules)} rules")
        
        # Show first rule content for categories with few rules
        if len(rules) <= 3 and rules:
            for i, rule in enumerate(rules):
                preview = rule[:150] + "..." if len(rule) > 150 else rule
                print(f"    Rule {i+1}: {preview}")
        elif rules:
            # Show just first rule for larger categories
            preview = rules[0][:150] + "..." if len(rules[0]) > 150 else rules[0]
            print(f"    First rule: {preview}")
    
    print(f"\nRules loaded from KB: {engine.rules_loaded}")
    
    # Show some sample content from the knowledge base
    print("\n📖 Sample from knowledge base:")
    rules_str = engine._load_multi_domain_rules()
    if rules_str:
        lines = rules_str.split('\n')
        print(f"Total lines in KB: {len(lines)}")
        print("First 10 lines:")
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1}: {line}")

if __name__ == "__main__":
    main()
