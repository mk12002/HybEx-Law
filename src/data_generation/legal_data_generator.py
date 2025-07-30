"""
Data Generation Utilities for HybEx-Law

This module provides tools for generating realistic legal aid queries
and expanding the training dataset with diverse, legally accurate examples.
"""

import random
import itertools
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class LegalScenarioTemplate:
    """Template for generating legal scenarios."""
    case_type: str
    income_ranges: List[tuple]
    social_categories: List[str]
    scenario_templates: List[str]
    outcome_logic: str


class LegalDataGenerator:
    """
    Generate realistic legal aid queries using templates and variations.
    
    This generator creates diverse training data while maintaining
    legal accuracy and realistic language patterns.
    """
    
    def __init__(self):
        """Initialize the data generator with legal templates."""
        
        # Income brackets (monthly in INR)
        self.income_brackets = {
            'very_low': (0, 5000),
            'low': (5001, 15000), 
            'moderate': (15001, 25000),
            'above_threshold': (25001, 50000),
            'high': (50001, 100000)
        }
        
        # Social categories with eligibility impact
        self.social_categories = {
            'woman': {'eligible': True, 'patterns': ['woman', 'female', 'wife', 'mother']},
            'child': {'eligible': True, 'patterns': ['16 years old', 'minor', 'under 18', 'student']},
            'sc_st': {'eligible': True, 'patterns': ['scheduled caste', 'scheduled tribe', 'SC', 'ST', 'dalit']},
            'disabled': {'eligible': True, 'patterns': ['disabled', 'wheelchair', 'blind', 'handicapped']},
            'industrial_worker': {'eligible': True, 'patterns': ['factory worker', 'industrial worker', 'mill worker']},
            'in_custody': {'eligible': True, 'patterns': ['in custody', 'arrested', 'jail', 'police custody']},
            'disaster_victim': {'eligible': True, 'patterns': ['flood victim', 'earthquake affected', 'disaster victim']},
            'general': {'eligible': False, 'patterns': ['general category', 'general caste', 'upper caste']}
        }
        
        # Employment status variations
        self.employment_status = {
            'unemployed': ['lost job', 'unemployed', 'no job', 'fired', 'terminated', 'laid off'],
            'employed': ['work in', 'employed at', 'job at', 'working as'],
            'self_employed': ['own business', 'self employed', 'run shop', 'small business'],
            'retired': ['retired', 'pension', 'senior citizen'],
            'student': ['student', 'studying', 'college student']
        }
        
        # Case type scenarios
        self.case_scenarios = self._initialize_case_scenarios()
        
        # Language variations
        self.money_expressions = [
            'earn {amount} rupees per month',
            'income is {amount} monthly', 
            'make {amount} per month',
            'salary of {amount}',
            'get {amount} rupees monthly',
            '{amount} income per month'
        ]
        
        self.help_requests = [
            'Can I get legal aid?',
            'Am I eligible for free lawyer?',
            'Can government provide lawyer?',
            'Do I qualify for legal help?',
            'Please help me get justice.',
            'I need legal assistance.'
        ]
    
    def _initialize_case_scenarios(self) -> Dict[str, LegalScenarioTemplate]:
        """Initialize case type scenario templates."""
        return {
            'property_dispute': LegalScenarioTemplate(
                case_type='property_dispute',
                income_ranges=[('very_low', 'low'), ('moderate',)],
                social_categories=['woman', 'sc_st', 'general'],
                scenario_templates=[
                    'My landlord is trying to evict me from my {housing_type}',
                    'Neighbor has occupied part of my land illegally',
                    'Property owner refusing to return security deposit',
                    'Landlord demanding illegal rent increase',
                    'Builder not giving possession of flat'
                ],
                outcome_logic='income_based OR categorical'
            ),
            
            'family_matter': LegalScenarioTemplate(
                case_type='family_matter',
                income_ranges=[('very_low', 'low', 'moderate'), ('above_threshold',)],
                social_categories=['woman', 'child', 'general'],
                scenario_templates=[
                    'My husband {domestic_issue} and demands dowry',
                    'Need help with child custody after divorce',
                    'Husband not paying maintenance for children',
                    'In-laws harassing me for dowry',
                    'Wife filed false case against me'
                ],
                outcome_logic='woman_always_eligible OR income_based'
            ),
            
            'domestic_violence': LegalScenarioTemplate(
                case_type='domestic_violence',
                income_ranges=[('very_low', 'low', 'moderate'), ('above_threshold',)],
                social_categories=['woman', 'general'],
                scenario_templates=[
                    'My husband physically abuses me daily',
                    'Facing domestic violence from in-laws',
                    'Husband threatens to throw me out of house',
                    'Being beaten and harassed for dowry',
                    'Need protection from violent husband'
                ],
                outcome_logic='woman_always_eligible OR income_based'
            ),
            
            'labor_dispute': LegalScenarioTemplate(
                case_type='labor_dispute',
                income_ranges=[('very_low', 'low'), ('moderate',)],
                social_categories=['industrial_worker', 'woman', 'general'],
                scenario_templates=[
                    'Employer has not paid my salary for {months} months',
                    'Factory fired me without proper notice',
                    'Not getting overtime payment from company',
                    'Workplace harassment by supervisor',
                    'Company not providing safety equipment'
                ],
                outcome_logic='industrial_worker_eligible OR income_based'
            ),
            
            'criminal_matter': LegalScenarioTemplate(
                case_type='criminal_matter',
                income_ranges=[('very_low', 'low', 'moderate'), ('above_threshold',)],
                social_categories=['in_custody', 'woman', 'general'],
                scenario_templates=[
                    'I was arrested and am in police custody',
                    'Police filed false case against me',
                    'Need bail for my {relation}',
                    'Victim of theft, need to file FIR',
                    'Someone cheated me of money'
                ],
                outcome_logic='in_custody_eligible OR income_based'
            ),
            
            'accident_compensation': LegalScenarioTemplate(
                case_type='accident_compensation',
                income_ranges=[('very_low', 'low', 'moderate'), ('above_threshold',)],
                social_categories=['disaster_victim', 'disabled', 'general'],
                scenario_templates=[
                    'My {relation} died in road accident, insurance not paying',
                    'House destroyed in floods, need compensation',
                    'Injured in factory accident, company denying liability',
                    'Medical negligence caused disability',
                    'Government not providing disaster relief'
                ],
                outcome_logic='disaster_victim_eligible OR income_based'
            ),
            
            'defamation': LegalScenarioTemplate(
                case_type='defamation',
                income_ranges=[('above_threshold', 'high')],
                social_categories=['general'],
                scenario_templates=[
                    'Someone wrote false things about me in newspaper',
                    'Neighbor spreading rumors about my character',
                    'Business rival damaging my reputation',
                    'False accusations on social media',
                    'Competitor making false claims about my business'
                ],
                outcome_logic='excluded_case_type'
            ),
            
            'business_dispute': LegalScenarioTemplate(
                case_type='business_dispute',
                income_ranges=[('above_threshold', 'high')],
                social_categories=['general'],
                scenario_templates=[
                    'Business partner cheating me out of profits',
                    'Supplier not delivering goods as per contract',
                    'Customer not paying for services provided',
                    'Investor backing out of agreed deal',
                    'Company merger dispute with shareholders'
                ],
                outcome_logic='excluded_case_type'
            )
        }
    
    def generate_query(self, case_type: str = None, complexity: str = 'simple') -> Dict[str, Any]:
        """
        Generate a single legal aid query.
        
        Args:
            case_type: Specific case type to generate, or None for random
            complexity: 'simple', 'moderate', or 'complex'
            
        Returns:
            Dictionary with query and expected annotations
        """
        # Map case types to available scenarios
        case_type_mapping = {
            'domestic_violence': 'family_matter',
            'property_dispute': 'property_dispute',
            'family_matter': 'family_matter',
            'labor_dispute': 'labor_dispute',
            'criminal_matter': 'criminal_matter',
            'accident_compensation': 'accident_compensation',
            'defamation': 'defamation',
            'business_dispute': 'business_dispute'
        }
        
        # Select case type
        if case_type is None:
            case_type = random.choice(list(self.case_scenarios.keys()))
        else:
            # Map to available scenario or use as-is
            case_type = case_type_mapping.get(case_type, case_type)
            if case_type not in self.case_scenarios:
                case_type = 'family_matter'  # Default fallback
        
        scenario = self.case_scenarios[case_type]
        
        # Select social category
        social_category = random.choice(scenario.social_categories)
        social_info = self.social_categories[social_category]
        
        # Select income bracket
        income_bracket = random.choice(scenario.income_ranges[0])
        income_range = self.income_brackets[income_bracket]
        income = random.randint(income_range[0], income_range[1]) if income_range[1] > 0 else 0
        
        # Generate scenario text
        scenario_template = random.choice(scenario.scenario_templates)
        
        # Fill in template variables
        scenario_text = self._fill_template_variables(scenario_template)
        
        # Generate complete query
        query_parts = []
        
        # Add social category information
        if social_category != 'general':
            social_pattern = random.choice(social_info['patterns'])
            query_parts.append(f"I am {social_pattern}.")
        
        # Add income information
        if income > 0:
            money_expr = random.choice(self.money_expressions).format(amount=income)
            query_parts.append(f"I {money_expr}.")
        else:
            unemployment_reason = random.choice(['lost my job', 'unemployed', 'have no income'])
            query_parts.append(f"I {unemployment_reason}.")
        
        # Add main scenario
        query_parts.append(scenario_text + ".")
        
        # Add help request
        if complexity in ['moderate', 'complex']:
            help_request = random.choice(self.help_requests)
            query_parts.append(help_request)
        
        # Combine parts
        query = " ".join(query_parts)
        
        # Generate expected facts
        expected_facts = self._generate_expected_facts(
            income, case_type, social_category, social_info
        )
        
        # Determine eligibility
        expected_eligible = self._determine_eligibility(
            income, case_type, social_category, scenario.outcome_logic
        )
        
        return {
            'query': query,
            'expected_facts': expected_facts,
            'expected_eligible': expected_eligible,
            'case_type': case_type,
            'social_category': social_category,
            'income': income,
            'complexity': complexity
        }
    
    def _fill_template_variables(self, template: str) -> str:
        """Fill template variables with realistic values."""
        variables = {
            'housing_type': ['house', 'apartment', 'flat', 'room'],
            'domestic_issue': ['beats me', 'harasses me', 'threatens me', 'abuses me'],
            'months': ['2', '3', '4', '5', '6'],
            'relation': ['son', 'husband', 'father', 'brother', 'friend']
        }
        
        filled_template = template
        for var, options in variables.items():
            if f'{{{var}}}' in template:
                value = random.choice(options)
                filled_template = filled_template.replace(f'{{{var}}}', value)
        
        return filled_template
    
    def _generate_expected_facts(self, income: int, case_type: str, 
                                social_category: str, social_info: Dict) -> List[str]:
        """Generate expected Prolog facts."""
        facts = ['applicant(user)']
        
        # Income facts
        if income >= 0:
            facts.append(f'income_monthly(user, {income})')
        
        # Case type fact
        facts.append(f'case_type(user, "{case_type}")')
        
        # Social category facts
        for category in ['woman', 'child', 'sc_st', 'disabled', 
                        'industrial_workman', 'in_custody', 'disaster_victim']:
            if category == social_category or (category == 'sc_st' and social_category == 'sc_st'):
                facts.append(f'is_{category}(user, true)')
            else:
                facts.append(f'is_{category}(user, false)')
        
        return facts
    
    def _determine_eligibility(self, income: int, case_type: str, 
                              social_category: str, outcome_logic: str) -> bool:
        """Determine expected eligibility based on legal rules."""
        # Excluded case types
        if case_type in ['defamation', 'business_dispute', 'election_offense']:
            return False
        
        # Categorical eligibility
        categorical_eligible = social_category in [
            'woman', 'child', 'sc_st', 'disabled', 
            'industrial_worker', 'in_custody', 'disaster_victim'
        ]
        
        # Income-based eligibility (threshold: 25,000 per month)
        income_eligible = income <= 25000
        
        # Apply logic
        return categorical_eligible or income_eligible
    
    def generate_dataset(self, size: int, case_type_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Generate a complete dataset of legal queries.
        
        Args:
            size: Number of queries to generate
            case_type_distribution: Distribution of case types (optional)
            
        Returns:
            List of generated queries with annotations
        """
        if case_type_distribution is None:
            # Default distribution
            case_type_distribution = {
                'property_dispute': 0.25,
                'family_matter': 0.20,
                'labor_dispute': 0.15,
                'criminal_matter': 0.15,
                'accident_compensation': 0.10,
                'defamation': 0.08,
                'business_dispute': 0.07
            }
        
        dataset = []
        
        for i in range(size):
            # Select case type based on distribution
            case_type = self._weighted_choice(case_type_distribution)
            
            # Vary complexity
            complexity = random.choices(
                ['simple', 'moderate', 'complex'],
                weights=[0.5, 0.3, 0.2]
            )[0]
            
            # Generate query
            query_data = self.generate_query(case_type, complexity)
            query_data['id'] = i + 1
            
            dataset.append(query_data)
        
        return dataset
    
    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Choose item based on weighted distribution."""
        items = list(weights.keys())
        weights_list = list(weights.values())
        return random.choices(items, weights=weights_list)[0]
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save generated dataset to file."""
        import json
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Dataset saved: {filename} ({len(dataset)} queries)")


def main():
    """Demonstrate data generation."""
    generator = LegalDataGenerator()
    
    print("🏭 Legal Data Generator Demo")
    print("=" * 40)
    
    # Generate individual examples
    print("📝 Sample Generated Queries:")
    for i in range(3):
        query_data = generator.generate_query()
        print(f"\n{i+1}. {query_data['case_type'].upper()}")
        print(f"   Query: {query_data['query']}")
        print(f"   Eligible: {'✅' if query_data['expected_eligible'] else '❌'}")
        print(f"   Facts: {len(query_data['expected_facts'])} extracted")
    
    # Generate dataset
    print(f"\n📊 Generating dataset...")
    dataset = generator.generate_dataset(size=50)
    
    # Statistics
    case_types = {}
    eligible_count = 0
    
    for item in dataset:
        case_type = item['case_type']
        case_types[case_type] = case_types.get(case_type, 0) + 1
        if item['expected_eligible']:
            eligible_count += 1
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total queries: {len(dataset)}")
    print(f"   Eligible cases: {eligible_count} ({eligible_count/len(dataset):.1%})")
    print(f"   Case type distribution:")
    for case_type, count in case_types.items():
        print(f"     {case_type}: {count} ({count/len(dataset):.1%})")
    
    # Save dataset
    generator.save_dataset(dataset, "data/generated_legal_queries.json")


if __name__ == "__main__":
    main()
