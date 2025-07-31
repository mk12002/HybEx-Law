"""
Legal Aid Domain Processor.

This module handles legal aid eligibility determination based on the
Legal Services Authorities Act, 1987 and related legislation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class LegalAidProcessor:
    """
    Processor for legal aid domain queries.
    
    Determines eligibility for legal aid based on income criteria,
    categorical criteria, and case type eligibility.
    """
    
    def __init__(self):
        self.income_thresholds = self._setup_income_thresholds()
        self.categorical_criteria = self._setup_categorical_criteria()
        self.excluded_cases = self._setup_excluded_cases()
        self.case_type_mapping = self._setup_case_type_mapping()
    
    def _setup_income_thresholds(self) -> Dict[str, int]:
        """Setup income thresholds for legal aid eligibility"""
        return {
            'rural_annual': 300000,      # 3 lakhs per annum for rural
            'urban_annual': 500000,      # 5 lakhs per annum for urban
            'rural_monthly': 25000,      # 25k per month for rural
            'urban_monthly': 41667,      # ~41.7k per month for urban
            'daily_rural': 833,          # ~833 per day for rural
            'daily_urban': 1389          # ~1389 per day for urban
        }
    
    def _setup_categorical_criteria(self) -> List[str]:
        """Setup categorical eligibility criteria"""
        return [
            'scheduled_caste',
            'scheduled_tribe', 
            'woman',
            'child',
            'mentally_ill',
            'disabled',
            'industrial_worker',
            'mass_disaster_victim',
            'transgender',
            'senior_citizen'
        ]
    
    def _setup_excluded_cases(self) -> List[str]:
        """Setup case types excluded from legal aid"""
        return [
            'business_dispute',
            'commercial_litigation',
            'corporate_matters',
            'tax_evasion',
            'election_petition',
            'defamation_by_press'
        ]
    
    def _setup_case_type_mapping(self) -> Dict[str, str]:
        """Map query terms to standard case types"""
        return {
            # Family law cases
            'domestic violence': 'domestic_violence',
            'dowry': 'dowry_harassment', 
            'divorce': 'matrimonial_dispute',
            'maintenance': 'maintenance_claim',
            'custody': 'child_custody',
            'inheritance': 'property_dispute',
            
            # Criminal cases
            'theft': 'theft',
            'assault': 'assault',
            'rape': 'sexual_offense',
            'harassment': 'harassment',
            'fraud': 'fraud',
            
            # Consumer cases
            'consumer': 'consumer_dispute',
            'defective product': 'consumer_dispute',
            'service deficiency': 'consumer_dispute',
            
            # Employment cases
            'termination': 'wrongful_termination',
            'workplace harassment': 'workplace_harassment',
            'minimum wage': 'wage_dispute', 
            
            # Property cases
            'property dispute': 'property_dispute',
            'land dispute': 'land_dispute',
            'tenant': 'landlord_tenant',
            
            # Constitutional cases
            'discrimination': 'discrimination',
            'fundamental rights': 'constitutional_violation',
            'police harassment': 'police_misconduct'
        }
    
    def extract_income_info(self, query: str) -> Dict[str, Any]:
        """
        Extract income information from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Dictionary with income information
        """
        income_info = {
            'monthly_income': None,
            'annual_income': None,
            'daily_income': None,
            'income_type': None,
            'location_type': 'urban'  # default to urban
        }
        
        query_lower = query.lower()
        
        # Detect location type
        rural_indicators = ['village', 'rural', 'gram', 'tehsil', 'block']
        if any(indicator in query_lower for indicator in rural_indicators):
            income_info['location_type'] = 'rural'
        
        # Extract income amounts
        income_patterns = [
            (r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)\s*(?:per\s+month|monthly|every\s+month)', 'monthly'),
            (r'(?:monthly|per\s+month)\s+(?:income|salary|earning|wage).*?(\d+(?:,\d+)*)', 'monthly'),
            (r'(?:earn|earning|income|salary|wage).*?(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)?\s*(?:per\s+month|monthly)', 'monthly'),
            (r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)\s*(?:per\s+year|annually|yearly)', 'annual'),
            (r'(?:annual|yearly)\s+(?:income|salary).*?(\d+(?:,\d+)*)', 'annual'),
            (r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)\s*(?:per\s+day|daily)', 'daily'),
            (r'(?:daily|per\s+day)\s+(?:wage|income).*?(\d+(?:,\d+)*)', 'daily'),
            # General income patterns
            (r'(?:earn|earning|income|salary|wage).*?(\d+(?:,\d+)*)', 'monthly'),  # assume monthly
            (r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)', 'monthly')  # assume monthly
        ]
        
        for pattern, income_type in income_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                amount = int(matches[0].replace(',', ''))
                
                if income_type == 'monthly':
                    income_info['monthly_income'] = amount
                    income_info['annual_income'] = amount * 12
                    income_info['daily_income'] = amount / 30
                elif income_type == 'annual':
                    income_info['annual_income'] = amount
                    income_info['monthly_income'] = amount / 12
                    income_info['daily_income'] = amount / 365
                elif income_type == 'daily':
                    income_info['daily_income'] = amount
                    income_info['monthly_income'] = amount * 30
                    income_info['annual_income'] = amount * 365
                
                income_info['income_type'] = income_type
                break
        
        return income_info
    
    def extract_categorical_info(self, query: str) -> List[str]:
        """
        Extract categorical eligibility information.
        
        Args:
            query: Legal query text
            
        Returns:
            List of applicable categorical criteria
        """
        categories = []
        query_lower = query.lower()
        
        # Check for each categorical criterion
        category_patterns = {
            'scheduled_caste': [r'\bsc\b', r'\bscheduled caste\b', r'\bdalit\b'],
            'scheduled_tribe': [r'\bst\b', r'\bscheduled tribe\b', r'\btribal\b'],
            'woman': [r'\bwoman\b', r'\bfemale\b', r'\bwife\b', r'\bmother\b', r'\bgirl\b'],
            'child': [r'\bchild\b', r'\bminor\b', r'\bjuvenile\b', r'\bkid\b', r'\byears? old\b.*\b(?:1[0-7]|[1-9])\b'],
            'mentally_ill': [r'\bmental\b.*\bill\b', r'\bpsychiatric\b', r'\bmental disorder\b'],
            'disabled': [r'\bdisabled\b', r'\bhandicapped\b', r'\bdisability\b', r'\bphysically challenged\b'],
            'industrial_worker': [r'\bindustrial worker\b', r'\bfactory worker\b', r'\blabour\b', r'\blaborer\b'],
            'mass_disaster_victim': [r'\bdisaster victim\b', r'\bearthquake\b', r'\bflood\b', r'\bcyclone\b'],
            'transgender': [r'\btransgender\b', r'\btrans\b', r'\bhijra\b'],
            'senior_citizen': [r'\bsenior citizen\b', r'\bolderly\b', r'\bold age\b', r'\b(?:6[0-9]|7[0-9]|8[0-9]|9[0-9])\s*years? old\b']
        }
        
        for category, patterns in category_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                categories.append(category)
        
        return categories
    
    def extract_case_type(self, query: str) -> Optional[str]:
        """
        Extract case type from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Identified case type or None
        """
        query_lower = query.lower()
        
        # Check for specific case type indicators
        for term, case_type in self.case_type_mapping.items():
            if term in query_lower:
                return case_type
        
        # Check for legal act mentions
        if any(term in query_lower for term in ['consumer protection', 'consumer court']):
            return 'consumer_dispute'
        
        if any(term in query_lower for term in ['domestic violence', 'protection of women']):
            return 'domestic_violence'
        
        if any(term in query_lower for term in ['industrial dispute', 'labor']):
            return 'employment_dispute'
        
        return None
    
    def check_income_eligibility(self, income_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if applicant meets income criteria for legal aid.
        
        Args:
            income_info: Income information extracted from query
            
        Returns:
            Dictionary with eligibility decision and details
        """
        result = {
            'eligible': False,
            'reason': '',
            'threshold_used': '',
            'income_amount': 0
        }
        
        if not income_info['monthly_income']:
            result['reason'] = 'no_income_information'
            return result
        
        location_type = income_info['location_type']
        monthly_income = income_info['monthly_income']
        
        # Determine applicable threshold
        if location_type == 'rural':
            threshold = self.income_thresholds['rural_monthly']
            threshold_name = 'rural_monthly'
        else:
            threshold = self.income_thresholds['urban_monthly']
            threshold_name = 'urban_monthly'
        
        result['threshold_used'] = threshold_name
        result['income_amount'] = monthly_income
        
        if monthly_income <= threshold:
            result['eligible'] = True
            result['reason'] = f'income_below_{threshold_name}_threshold'
        else:
            result['eligible'] = False
            result['reason'] = f'income_above_{threshold_name}_threshold'
        
        return result
    
    def check_categorical_eligibility(self, categories: List[str]) -> Dict[str, Any]:
        """
        Check if applicant meets categorical criteria for legal aid.
        
        Args:
            categories: List of applicable categories
            
        Returns:
            Dictionary with eligibility decision and details
        """
        result = {
            'eligible': False,
            'reason': '',
            'applicable_categories': categories
        }
        
        if not categories:
            result['reason'] = 'no_categorical_criteria'
            return result
        
        # Check if any category makes them eligible
        eligible_categories = []
        for category in categories:
            if category in self.categorical_criteria:
                eligible_categories.append(category)
        
        if eligible_categories:
            result['eligible'] = True  
            result['reason'] = f"categorical_eligibility_{eligible_categories[0]}"
        else:
            result['reason'] = 'no_valid_categorical_criteria'
        
        return result
    
    def check_case_eligibility(self, case_type: Optional[str]) -> Dict[str, Any]:
        """
        Check if case type is eligible for legal aid.
        
        Args:
            case_type: Identified case type
            
        Returns:
            Dictionary with case eligibility decision
        """
        result = {
            'eligible': True,
            'reason': '',
            'case_type': case_type
        }
        
        if not case_type:
            result['reason'] = 'case_type_not_identified'
            return result
        
        if case_type in self.excluded_cases:
            result['eligible'] = False
            result['reason'] = f'excluded_case_type_{case_type}'
        else:
            result['reason'] = 'eligible_case_type'
        
        return result
    
    def extract_facts(self, query: str) -> List[str]:
        """
        Extract legal facts from query for Prolog reasoning.
        
        Args:
            query: Legal query text
            
        Returns:
            List of Prolog facts
        """
        facts = ['applicant(user).']
        
        # Extract income information
        income_info = self.extract_income_info(query)
        if income_info['monthly_income']:
            facts.append(f"income_monthly(user, {int(income_info['monthly_income'])}).")
            facts.append(f"location_type(user, '{income_info['location_type']}').")
        
        # Extract categorical information
        categories = self.extract_categorical_info(query)
        for category in categories:
            if category == 'woman':
                facts.append('is_woman(user, true).')
            elif category == 'scheduled_caste':
                facts.append('is_sc_st(user, true).')
                facts.append("social_category(user, 'sc').")
            elif category == 'scheduled_tribe':
                facts.append('is_sc_st(user, true).')
                facts.append("social_category(user, 'st').")
            elif category == 'child':
                facts.append('is_minor(user, true).')
            elif category == 'disabled':
                facts.append('is_disabled(user, true).')
            elif category == 'senior_citizen':
                facts.append('is_senior_citizen(user, true).')
        
        # Extract case type
        case_type = self.extract_case_type(query)
        if case_type:
            facts.append(f"case_type(user, '{case_type}').")
        
        return facts
    
    def analyze_legal_position(self, facts: List[str]) -> Dict[str, Any]:
        """
        Analyze legal position for legal aid eligibility.
        
        Args:
            facts: List of extracted legal facts
            
        Returns:
            Dictionary with legal analysis
        """
        analysis = {
            'eligible_for_legal_aid': False,
            'eligibility_basis': [],
            'income_eligible': False,
            'categorically_eligible': False,
            'case_eligible': True,
            'recommended_actions': [],
            'applicable_acts': []
        }
        
        facts_str = ' '.join(facts)
        
        # Check income eligibility
        income_match = re.search(r'income_monthly\(user, (\d+)\)', facts_str)
        location_match = re.search(r"location_type\(user, '(\w+)'\)", facts_str)
        
        if income_match:
            monthly_income = int(income_match.group(1))
            location = location_match.group(1) if location_match else 'urban'
            
            threshold = (self.income_thresholds['rural_monthly'] if location == 'rural' 
                        else self.income_thresholds['urban_monthly'])
            
            if monthly_income <= threshold:
                analysis['income_eligible'] = True
                analysis['eligibility_basis'].append('income_criteria')
        
        # Check categorical eligibility
        categorical_indicators = [
            'is_woman(user, true)',
            'is_sc_st(user, true)', 
            'is_minor(user, true)',
            'is_disabled(user, true)',
            'is_senior_citizen(user, true)'
        ]
        
        for indicator in categorical_indicators:
            if indicator in facts_str:
                analysis['categorically_eligible'] = True
                category = indicator.split('(')[0].replace('is_', '')
                analysis['eligibility_basis'].append(f'categorical_{category}')
                break
        
        # Check case eligibility
        case_match = re.search(r"case_type\(user, '(\w+)'\)", facts_str)
        if case_match:
            case_type = case_match.group(1)
            if case_type in self.excluded_cases:
                analysis['case_eligible'] = False
                analysis['eligibility_basis'].append('excluded_case_type')
        
        # Overall eligibility
        analysis['eligible_for_legal_aid'] = (
            (analysis['income_eligible'] or analysis['categorically_eligible']) 
            and analysis['case_eligible']
        )
        
        # Recommendations
        if analysis['eligible_for_legal_aid']:
            analysis['recommended_actions'] = [
                'Apply to District Legal Services Authority',
                'Gather income/category proof documents',
                'File legal aid application with case details'
            ]
        else:
            analysis['recommended_actions'] = [
                'Consider private legal counsel',
                'Check for other legal aid schemes',
                'Seek advice from legal aid clinic'
            ]
        
        # Applicable acts
        analysis['applicable_acts'] = [
            'Legal Services Authorities Act, 1987'
        ]
        
        if 'is_woman' in facts_str:
            analysis['applicable_acts'].append('Legal Aid to Women (Protection of Rights) Act')
        
        if 'is_minor' in facts_str:
            analysis['applicable_acts'].append('Juvenile Justice (Care and Protection of Children) Act, 2015')
        
        if 'is_disabled' in facts_str:
            analysis['applicable_acts'].append('Rights of Persons with Disabilities Act, 2016')
        
        if 'is_sc_st' in facts_str:
            analysis['applicable_acts'].append('SC/ST (Prevention of Atrocities) Act, 1989')
        
        return analysis
