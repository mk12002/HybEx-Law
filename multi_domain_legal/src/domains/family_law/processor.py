"""
Family Law Domain Processor.

This module handles family law queries including marriage, divorce,
maintenance, custody, and inheritance matters across different personal laws.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

class FamilyLawProcessor:
    """
    Processor for family law domain queries.
    
    Handles marriage, divorce, maintenance, custody, and inheritance
    matters under Hindu, Muslim, Christian, and Parsi personal laws.
    """
    
    def __init__(self):
        self.personal_laws = self._setup_personal_laws()
        self.divorce_grounds = self._setup_divorce_grounds()
        self.maintenance_criteria = self._setup_maintenance_criteria()
        self.custody_factors = self._setup_custody_factors()
        self.inheritance_rules = self._setup_inheritance_rules()
    
    def _setup_personal_laws(self) -> Dict[str, Dict[str, Any]]:
        """Setup personal law frameworks"""
        return {
            'hindu': {
                'acts': ['Hindu Marriage Act, 1955', 'Hindu Succession Act, 1956'],
                'keywords': ['hindu', 'hinduism', 'hindu marriage', 'hindu family'],
                'divorce_grounds': ['cruelty', 'adultery', 'conversion', 'mental_disorder', 'renunciation', 'desertion'],
                'inheritance': 'coparcenary_rights'
            },
            'muslim': {
                'acts': ['Muslim Personal Law (Shariat) Application Act, 1937'],
                'keywords': ['muslim', 'islamic', 'shariat', 'nikah', 'talaq', 'mehr'],
                'divorce_grounds': ['talaq', 'khula', 'mubarat', 'zihar'],
                'inheritance': 'shariat_law'
            },
            'christian': {
                'acts': ['Indian Christian Marriage Act, 1872'],
                'keywords': ['christian', 'christianity', 'church', 'christian marriage'],
                'divorce_grounds': ['adultery', 'conversion', 'cruelty', 'desertion'],
                'inheritance': 'indian_succession_act'
            },
            'parsi': {
                'acts': ['Parsi Marriage and Divorce Act, 1936'],
                'keywords': ['parsi', 'zoroastrian', 'parsi marriage'],
                'divorce_grounds': ['adultery', 'cruelty', 'desertion', 'conversion'],
                'inheritance': 'parsi_intestate_succession'
            }
        }
    
    def _setup_divorce_grounds(self) -> Dict[str, List[str]]:
        """Setup divorce grounds with their indicators"""
        return {
            'cruelty': [
                'physical violence', 'beating', 'hitting', 'assault', 'abuse', 
                'mental cruelty', 'torture', 'harassment', 'threats'
            ],
            'adultery': [
                'extramarital affair', 'cheating', 'relationship with another',
                'unfaithful', 'infidelity', 'another man', 'another woman'
            ],
            'desertion': [
                'abandoned', 'left home', 'living separately', 'deserted',
                'not living together', 'separated for years'
            ],
            'conversion': [
                'converted religion', 'changed religion', 'became christian',
                'became muslim', 'became hindu', 'religious conversion'
            ],
            'mental_disorder': [
                'mental illness', 'psychiatric disorder', 'insanity',
                'mental disease', 'psychological disorder'
            ],
            'impotency': [
                'impotent', 'sexual dysfunction', 'unable to consummate',
                'physical incapacity'
            ],
            'renunciation': [
                'became monk', 'renounced world', 'sannyasi', 'religious order'
            ]
        }
    
    def _setup_maintenance_criteria(self) -> Dict[str, Any]:
        """Setup maintenance eligibility criteria"""
        return {
            'eligible_persons': ['wife', 'divorced_wife', 'children', 'elderly_parents'],
            'factors': [
                'income_of_husband', 'needs_of_wife', 'standard_of_living',
                'age_and_health', 'earning_capacity', 'conduct_of_parties'
            ],
            'sections': {
                'hindu': 'Section 25 Hindu Marriage Act',
                'muslim': 'Section 125 CrPC + Muslim Personal Law',
                'christian': 'Section 37 Indian Christian Marriage Act',
                'general': 'Section 125 CrPC'
            }
        }
    
    def _setup_custody_factors(self) -> Dict[str, Any]:
        """Setup child custody consideration factors"""
        return {
            'primary_factors': [
                'welfare_of_child', 'age_of_child', 'preference_of_child',
                'financial_capacity', 'moral_fitness', 'stability'
            ],
            'age_guidelines': {
                'tender_years': {'age': 5, 'preference': 'mother'},
                'school_age': {'age': 12, 'preference': 'best_interest'},
                'adolescent': {'age': 16, 'preference': 'child_choice'}
            },
            'types': ['sole_custody', 'joint_custody', 'shared_custody', 'visitation_rights']
        }
    
    def _setup_inheritance_rules(self) -> Dict[str, Any]:
        """Setup inheritance and succession rules"""
        return {
            'hindu': {
                'act': 'Hindu Succession Act, 1956',
                'female_rights': 'coparcenary_rights',
                'succession_order': ['spouse', 'children', 'parents', 'siblings']
            },
            'muslim': {
                'act': 'Muslim Personal Law',
                'system': 'quranic_inheritance',
                'shares': {'wife': '1/8', 'daughter': '1/2', 'son': '2x_daughter'}
            },
            'general': {
                'act': 'Indian Succession Act, 1925',
                'applies_to': ['christian', 'parsi', 'other']
            }
        }
    
    def identify_personal_law(self, query: str) -> str:
        """
        Identify applicable personal law from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Identified personal law ('hindu', 'muslim', 'christian', 'parsi', or 'general')
        """
        query_lower = query.lower()
        
        for law_type, law_info in self.personal_laws.items():
            if any(keyword in query_lower for keyword in law_info['keywords']):
                return law_type
        
        # Default to Hindu law if Indian names/terms present
        indian_indicators = ['sharma', 'gupta', 'singh', 'kumar', 'devi', 'indian']
        if any(indicator in query_lower for indicator in indian_indicators):
            return 'hindu'
        
        return 'general'
    
    def extract_marriage_info(self, query: str) -> Dict[str, Any]:
        """
        Extract marriage-related information from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Dictionary with marriage information
        """
        marriage_info = {
            'marital_status': None,
            'marriage_duration': None,
            'has_children': False,
            'personal_law': 'general'
        }
        
        query_lower = query.lower()
        
        # Identify personal law
        marriage_info['personal_law'] = self.identify_personal_law(query)
        
        # Extract marital status
        if any(term in query_lower for term in ['married', 'husband', 'wife', 'spouse']):
            marriage_info['marital_status'] = 'married'
        elif any(term in query_lower for term in ['divorced', 'ex-husband', 'ex-wife']):
            marriage_info['marital_status'] = 'divorced'
        elif any(term in query_lower for term in ['separated', 'living separately']):
            marriage_info['marital_status'] = 'separated'
        
        # Extract marriage duration
        duration_pattern = r'(?:married|marriage).*?(\d+)\s*years?'
        duration_match = re.search(duration_pattern, query_lower)
        if duration_match:
            marriage_info['marriage_duration'] = int(duration_match.group(1))
        
        # Check for children
        if any(term in query_lower for term in ['child', 'children', 'son', 'daughter', 'kids']):
            marriage_info['has_children'] = True
        
        return marriage_info
    
    def extract_divorce_grounds(self, query: str) -> List[str]:
        """
        Extract divorce grounds from query.
        
        Args:
            query: Legal query text
            
        Returns:
            List of applicable divorce grounds
        """
        grounds = []
        query_lower = query.lower()
        
        for ground, indicators in self.divorce_grounds.items():
            if any(indicator in query_lower for indicator in indicators):
                grounds.append(ground)
        
        return grounds
    
    def extract_maintenance_details(self, query: str) -> Dict[str, Any]:
        """
        Extract maintenance-related details from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Dictionary with maintenance information
        """
        maintenance_info = {
            'seeking_maintenance': False,
            'maintenance_type': None,
            'income_spouse': None,
            'own_income': None,
            'children_involved': False
        }
        
        query_lower = query.lower()
        
        # Check if seeking maintenance
        maintenance_keywords = ['maintenance', 'alimony', 'financial support', 'monthly allowance']
        if any(keyword in query_lower for keyword in maintenance_keywords):
            maintenance_info['seeking_maintenance'] = True
        
        # Determine maintenance type
        if 'interim maintenance' in query_lower:
            maintenance_info['maintenance_type'] = 'interim'
        elif 'permanent maintenance' in query_lower:
            maintenance_info['maintenance_type'] = 'permanent'
        elif 'child support' in query_lower:
            maintenance_info['maintenance_type'] = 'child_support'
        
        # Extract income information
        income_patterns = [
            r'(?:husband|spouse|ex-husband).*?(?:earns|income|salary).*?(\d+(?:,\d+)*)',
            r'(?:earns|income|salary).*?(\d+(?:,\d+)*)'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, query_lower)
            if match:
                maintenance_info['income_spouse'] = int(match.group(1).replace(',', ''))
                break
        
        # Check for children
        if any(term in query_lower for term in ['child', 'children', 'custody']):
            maintenance_info['children_involved'] = True
        
        return maintenance_info
    
    def extract_custody_details(self, query: str) -> Dict[str, Any]:
        """
        Extract child custody details from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Dictionary with custody information
        """
        custody_info = {
            'seeking_custody': False,
            'custody_type': None,
            'child_age': None,
            'current_custody': None,
            'custody_dispute': False
        }
        
        query_lower = query.lower()
        
        # Check if custody-related
        custody_keywords = ['custody', 'guardianship', 'visitation', 'access to child']
        if any(keyword in query_lower for keyword in custody_keywords):
            custody_info['seeking_custody'] = True
        
        # Determine custody type
        if 'sole custody' in query_lower:
            custody_info['custody_type'] = 'sole'
        elif 'joint custody' in query_lower:
            custody_info['custody_type'] = 'joint'
        elif 'visitation' in query_lower:
            custody_info['custody_type'] = 'visitation'
        
        # Extract child age
        age_pattern = r'(?:child|son|daughter).*?(\d+)\s*years?\s*old'
        age_match = re.search(age_pattern, query_lower)
        if age_match:
            custody_info['child_age'] = int(age_match.group(1))
        
        # Check for custody disputes
        dispute_indicators = ['not allowing', 'preventing', 'denying access', 'hiding child']
        if any(indicator in query_lower for indicator in dispute_indicators):
            custody_info['custody_dispute'] = True
        
        return custody_info
    
    def extract_property_details(self, query: str) -> Dict[str, Any]:
        """
        Extract property and inheritance details from query.
        
        Args:
            query: Legal query text
            
        Returns:
            Dictionary with property information
        """
        property_info = {
            'inheritance_dispute': False,
            'property_type': None,
            'relationship_deceased': None,
            'other_heirs': False,
            'will_exists': None
        }
        
        query_lower = query.lower()
        
        # Check for inheritance issues
        inheritance_keywords = ['inheritance', 'succession', 'property rights', 'ancestral property']
        if any(keyword in query_lower for keyword in inheritance_keywords):
            property_info['inheritance_dispute'] = True
        
        # Identify property type
        property_types = {
            'house': ['house', 'home', 'residence'],
            'land': ['land', 'plot', 'agricultural land'],
            'business': ['business', 'shop', 'commercial'],
            'ancestral': ['ancestral property', 'family property']
        }
        
        for prop_type, indicators in property_types.items():
            if any(indicator in query_lower for indicator in indicators):
                property_info['property_type'] = prop_type
                break
        
        # Identify relationship to deceased
        relationships = ['father', 'mother', 'husband', 'grandfather', 'grandmother']
        for relationship in relationships:
            if relationship in query_lower:
                property_info['relationship_deceased'] = relationship
                break
        
        # Check for will
        if 'will' in query_lower:
            if 'no will' in query_lower or 'without will' in query_lower:
                property_info['will_exists'] = False
            else:
                property_info['will_exists'] = True
        
        # Check for other heirs
        heir_indicators = ['brother', 'sister', 'son', 'daughter', 'other legal heirs']
        if any(indicator in query_lower for indicator in heir_indicators):
            property_info['other_heirs'] = True
        
        return property_info
    
    def extract_facts(self, query: str) -> List[str]:
        """
        Extract legal facts from family law query.
        
        Args:
            query: Legal query text
            
        Returns:
            List of Prolog facts
        """
        facts = ['applicant(user).']
        
        # Extract marriage information
        marriage_info = self.extract_marriage_info(query)
        if marriage_info['marital_status']:
            facts.append(f"marital_status(user, {marriage_info['marital_status']}).")
        if marriage_info['marriage_duration']:
            facts.append(f"marriage_duration(user, {marriage_info['marriage_duration']}).")
        if marriage_info['has_children']:
            facts.append('has_children(user, true).')
        facts.append(f"personal_law(user, '{marriage_info['personal_law']}').")
        
        # Extract divorce grounds
        divorce_grounds = self.extract_divorce_grounds(query)
        for ground in divorce_grounds:
            facts.append(f"divorce_ground(user, '{ground}').")
        
        # Extract maintenance details
        maintenance_info = self.extract_maintenance_details(query)
        if maintenance_info['seeking_maintenance']:
            facts.append('seeks_maintenance(user, true).')
        if maintenance_info['maintenance_type']:
            facts.append(f"maintenance_type(user, '{maintenance_info['maintenance_type']}').")
        if maintenance_info['income_spouse']:
            facts.append(f"spouse_income(user, {maintenance_info['income_spouse']}).")
        
        # Extract custody details
        custody_info = self.extract_custody_details(query)
        if custody_info['seeking_custody']:
            facts.append('seeks_custody(user, true).')
        if custody_info['custody_type']:
            facts.append(f"custody_type(user, '{custody_info['custody_type']}').")
        if custody_info['child_age']:
            facts.append(f"child_age(user, {custody_info['child_age']}).")
        if custody_info['custody_dispute']:
            facts.append('custody_dispute(user, true).')
        
        # Extract property details
        property_info = self.extract_property_details(query)
        if property_info['inheritance_dispute']:
            facts.append('inheritance_dispute(user, true).')
        if property_info['property_type']:
            facts.append(f"property_type(user, '{property_info['property_type']}').")
        if property_info['relationship_deceased']:
            facts.append(f"relationship_deceased(user, '{property_info['relationship_deceased']}').")
        if property_info['will_exists'] is not None:
            facts.append(f"will_exists(user, {str(property_info['will_exists']).lower()}).")
        
        return facts
    
    def analyze_legal_position(self, facts: List[str]) -> Dict[str, Any]:
        """
        Analyze legal position for family law matters.
        
        Args:
            facts: List of extracted legal facts
            
        Returns:
            Dictionary with legal analysis
        """
        analysis = {
            'can_file_divorce': False,
            'entitled_to_maintenance': False,
            'child_custody_rights': False,
            'property_inheritance_rights': False,
            'recommended_actions': [],
            'applicable_acts': [],
            'legal_remedies': []
        }
        
        facts_str = ' '.join(facts)
        
        # Analyze divorce eligibility
        if 'divorce_ground(' in facts_str and 'marital_status(user, married)' in facts_str:
            analysis['can_file_divorce'] = True
            analysis['legal_remedies'].append('File divorce petition')
            analysis['recommended_actions'].append('Gather evidence of divorce grounds')
        
        # Analyze maintenance eligibility
        if ('marital_status(user, married)' in facts_str or 
            'marital_status(user, divorced)' in facts_str or 
            'marital_status(user, separated)' in facts_str):
            analysis['entitled_to_maintenance'] = True
            analysis['legal_remedies'].append('Apply for maintenance under Section 125 CrPC')
        
        # Analyze custody rights
        if 'has_children(user, true)' in facts_str:
            analysis['child_custody_rights'] = True
            analysis['legal_remedies'].append('File custody petition')
            analysis['recommended_actions'].append('Document child welfare factors')
        
        # Analyze inheritance rights
        if 'inheritance_dispute(user, true)' in facts_str:
            analysis['property_inheritance_rights'] = True
            analysis['legal_remedies'].append('File succession case')
            analysis['recommended_actions'].append('Gather property documents')
        
        # Determine applicable acts
        personal_law_match = re.search(r"personal_law\(user, '(\w+)'\)", facts_str)
        if personal_law_match:
            personal_law = personal_law_match.group(1)
            if personal_law in self.personal_laws:
                analysis['applicable_acts'].extend(self.personal_laws[personal_law]['acts'])
        
        # General recommendations
        if analysis['can_file_divorce']:
            analysis['recommended_actions'].extend([
                'Consider mediation/counseling first',
                'Prepare financial disclosures',
                'Arrange for child welfare if applicable'
            ])
        
        if analysis['entitled_to_maintenance']:
            analysis['recommended_actions'].extend([
                'Calculate monthly expenses',
                'Gather spouse income proof',
                'File interim maintenance application'
            ])
        
        return analysis
