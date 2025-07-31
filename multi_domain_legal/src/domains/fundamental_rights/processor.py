"""
Fundamental Rights Domain Processor.

This module handles fundamental rights violations, constitutional matters,
RTI queries, and human rights issues.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

class FundamentalRightsProcessor:
    """
    Processor for fundamental rights domain queries.
    
    Handles constitutional rights violations, RTI matters, discrimination cases,
    and human rights issues under various constitutional and statutory provisions.
    """
    
    def __init__(self):
        self.fundamental_rights = self._setup_fundamental_rights()
        self.discrimination_grounds = self._setup_discrimination_grounds()
        self.government_authorities = self._setup_government_authorities()
        self.rti_provisions = self._setup_rti_provisions()
        self.remedies = self._setup_remedies()
    
    def _setup_fundamental_rights(self) -> Dict[str, Dict[str, Any]]:
        """Setup fundamental rights with their provisions"""
        return {
            'equality': {
                'article': 14,
                'description': 'Right to Equality',
                'keywords': ['equality', 'equal treatment', 'discrimination', 'equal protection'],
                'violations': ['unequal_treatment', 'discriminatory_laws', 'arbitrary_action']
            },
            'freedom': {
                'article': 19,
                'description': 'Right to Freedom',
                'keywords': ['freedom', 'speech', 'expression', 'assembly', 'movement', 'profession'],
                'violations': ['censorship', 'restriction_movement', 'profession_ban']
            },
            'life_liberty': {
                'article': 21,
                'description': 'Right to Life and Personal Liberty',
                'keywords': ['life', 'liberty', 'personal liberty', 'dignity', 'privacy'],
                'violations': ['illegal_detention', 'police_custody', 'encounter', 'torture']
            },
            'religion': {
                'article': 25,
                'description': 'Freedom of Religion',
                'keywords': ['religion', 'religious freedom', 'worship', 'faith', 'belief'],
                'violations': ['religious_persecution', 'forced_conversion', 'religious_discrimination']
            },
            'cultural_educational': {
                'article': '29-30',
                'description': 'Cultural and Educational Rights',
                'keywords': ['culture', 'language', 'education', 'minority rights'],
                'violations': ['cultural_suppression', 'language_discrimination', 'educational_denial']
            },
            'constitutional_remedies': {
                'article': 32,
                'description': 'Right to Constitutional Remedies',
                'keywords': ['constitutional remedy', 'writ petition', 'habeas corpus', 'mandamus'],
                'violations': ['denial_of_remedy', 'court_access_denied']
            }
        }
    
    def _setup_discrimination_grounds(self) -> List[str]:
        """Setup prohibited grounds of discrimination"""
        return [
            'religion', 'race', 'caste', 'sex', 'place_of_birth', 'gender', 'disability',
            'economic_status', 'political_opinion', 'sexual_orientation', 'age'
        ]
    
    def _setup_government_authorities(self) -> Dict[str, List[str]]:
        """Setup government authorities and their keywords"""
        return {
            'police': ['police', 'constable', 'inspector', 'superintendent', 'commissioner'],
            'administration': ['collector', 'magistrate', 'commissioner', 'secretary', 'officer'],
            'court': ['judge', 'magistrate', 'court', 'judicial', 'tribunal'],
            'legislative': ['mla', 'mp', 'minister', 'legislature', 'assembly'],
            'executive': ['governor', 'chief minister', 'minister', 'bureaucrat'],
            'municipal': ['mayor', 'corporator', 'municipal', 'civic body', 'gram panchayat']
        }
    
    def _setup_rti_provisions(self) -> Dict[str, Any]:
        """Setup RTI Act provisions and procedures"""
        return {
            'time_limits': {
                'normal_response': 30,  # days
                'life_liberty_response': 48,  # hours
                'first_appeal': 30,  # days
                'second_appeal': 90  # days
            },
            'fees': {
                'application_fee': 10,  # rupees
                'additional_fee_per_page': 2,
                'inspection_fee_per_hour': 5
            },
            'exempt_information': [
                'national_security', 'foreign_relations', 'cabinet_papers',
                'investigation_records', 'personal_information', 'commercial_confidence'
            ],
            'authorities': {
                'pio': 'Public Information Officer',
                'apio': 'Assistant Public Information Officer', 
                'first_appellate': 'First Appellate Authority',
                'cic': 'Central Information Commission',
                'sic': 'State Information Commission'
            }
        }
    
    def _setup_remedies(self) -> Dict[str, List[str]]:
        """Setup available remedies for rights violations"""
        return {
            'constitutional': [
                'writ_petition_high_court', 'writ_petition_supreme_court',
                'habeas_corpus', 'mandamus', 'prohibition', 'certiorari', 'quo_warranto'
            ],
            'statutory': [
                'complaint_to_nhrc', 'complaint_to_shrc', 'rti_application',
                'complaint_to_police', 'complaint_to_magistrate'
            ],
            'administrative': [
                'departmental_complaint', 'grievance_redressal', 'ombudsman',
                'vigilance_complaint', 'audit_complaint'
            ]
        }
    
    def identify_fundamental_right(self, query: str) -> List[str]:
        """
        Identify which fundamental rights are involved.
        
        Args:
            query: Rights violation query
            
        Returns:
            List of applicable fundamental rights
        """
        rights = []
        query_lower = query.lower()
        
        for right_name, right_info in self.fundamental_rights.items():
            if any(keyword in query_lower for keyword in right_info['keywords']):
                rights.append(right_name)
        
        return rights
    
    def identify_discrimination_ground(self, query: str) -> List[str]:
        """
        Identify grounds of discrimination mentioned in query.
        
        Args:
            query: Discrimination query
            
        Returns:
            List of discrimination grounds
        """
        grounds = []
        query_lower = query.lower()
        
        discrimination_patterns = {
            'religion': ['religion', 'hindu', 'muslim', 'christian', 'sikh', 'buddhist'],
            'caste': ['caste', 'sc', 'st', 'obc', 'scheduled caste', 'scheduled tribe', 'brahmin', 'dalit'],
            'sex': ['gender', 'woman', 'female', 'male', 'sex', 'transgender'],
            'race': ['race', 'racial', 'ethnicity', 'tribal'],
            'place_of_birth': ['state', 'domicile', 'resident', 'outsider', 'migrant'],
            'disability': ['disabled', 'handicapped', 'physically challenged', 'disability'],
            'economic_status': ['poor', 'rich', 'economic', 'financial status', 'poverty'],
            'age': ['age', 'young', 'old', 'elderly', 'senior citizen']
        }
        
        for ground, indicators in discrimination_patterns.items():
            if any(indicator in query_lower for indicator in indicators):
                grounds.append(ground)
        
        return grounds
    
    def identify_violating_authority(self, query: str) -> Optional[str]:
        """
        Identify government authority involved in violation.
        
        Args:
            query: Rights violation query
            
        Returns:
            Type of government authority or None
        """
        query_lower = query.lower()
        
        for authority_type, keywords in self.government_authorities.items():
            if any(keyword in query_lower for keyword in keywords):
                return authority_type
        
        return None
    
    def extract_rti_details(self, query: str) -> Dict[str, Any]:
        """
        Extract RTI-specific details from query.
        
        Args:
            query: RTI-related query
            
        Returns:
            Dictionary with RTI information
        """
        rti_info = {
            'is_rti_query': False,
            'information_sought': None,
            'authority_involved': None,
            'application_status': None,
            'response_time_exceeded': False,
            'appeal_required': False
        }
        
        query_lower = query.lower()
        
        # Check if it's an RTI query
        rti_keywords = ['rti', 'right to information', 'information', 'government documents']
        if any(keyword in query_lower for keyword in rti_keywords):
            rti_info['is_rti_query'] = True
        
        # Extract information type
        info_patterns = {
            'government_schemes': ['scheme', 'yojana', 'program', 'benefit'],
            'expenditure': ['expenditure', 'spending', 'budget', 'allocation'],
            'appointments': ['appointment', 'recruitment', 'selection', 'hiring'],
            'policies': ['policy', 'rule', 'guideline', 'circular'],
            'complaints': ['complaint', 'grievance', 'petition', 'representation']
        }
        
        for info_type, indicators in info_patterns.items():
            if any(indicator in query_lower for indicator in indicators):
                rti_info['information_sought'] = info_type
                break
        
        # Check for time-related issues
        time_indicators = ['30 days', 'not replied', 'no response', 'delayed', 'pending']
        if any(indicator in query_lower for indicator in time_indicators):
            rti_info['response_time_exceeded'] = True
            rti_info['appeal_required'] = True
        
        # Check for rejection
        if any(term in query_lower for term in ['rejected', 'denied', 'refused']):
            rti_info['application_status'] = 'rejected'
            rti_info['appeal_required'] = True
        
        return rti_info
    
    def extract_violation_details(self, query: str) -> Dict[str, Any]:
        """
        Extract details about rights violation.
        
        Args:
            query: Rights violation query
            
        Returns:
            Dictionary with violation details
        """
        violation_info = {
            'violation_type': None,
            'location': None,
            'date_of_incident': None,
            'witnesses': False,
            'complaint_filed': False,
            'urgency_level': 'normal'
        }
        
        query_lower = query.lower()
        
        # Identify violation type
        violation_patterns = {
            'illegal_detention': ['detained', 'arrest', 'custody', 'locked up', 'imprisoned'],
            'police_harassment': ['police harassment', 'police torture', 'police beating'],
            'discrimination': ['discriminated', 'denied service', 'refused entry', 'unequal treatment'],
            'censorship': ['banned', 'censored', 'prohibited', 'stopped from speaking'],
            'religious_persecution': ['religious harassment', 'forced conversion', 'religious violence']
        }
        
        for violation_type, indicators in violation_patterns.items():
            if any(indicator in query_lower for indicator in indicators):
                violation_info['violation_type'] = violation_type
                break
        
        # Check urgency
        urgent_indicators = ['immediate', 'urgent', 'emergency', 'life threat', 'danger']
        if any(indicator in query_lower for indicator in urgent_indicators):
            violation_info['urgency_level'] = 'urgent'
        
        # Check if complaint filed
        if any(term in query_lower for term in ['complained', 'fir', 'police complaint', 'filed case']):
            violation_info['complaint_filed'] = True
        
        # Check for witnesses
        if any(term in query_lower for term in ['witness', 'people saw', 'others present']):
            violation_info['witnesses'] = True
        
        return violation_info
    
    def extract_facts(self, query: str) -> List[str]:
        """
        Extract legal facts from fundamental rights query.
        
        Args:
            query: Rights violation query
            
        Returns:
            List of Prolog facts
        """
        facts = ['citizen(user).']
        
        # Extract fundamental rights
        rights = self.identify_fundamental_right(query)
        for right in rights:
            facts.append(f"fundamental_right_involved(user, '{right}').")
        
        # Extract discrimination grounds
        discrimination_grounds = self.identify_discrimination_ground(query)
        for ground in discrimination_grounds:
            facts.append(f"discrimination_ground(user, '{ground}').")
        
        # Extract violating authority
        authority = self.identify_violating_authority(query)
        if authority:
            facts.append(f"violating_authority(user, '{authority}').")
        
        # Extract RTI details
        rti_info = self.extract_rti_details(query)
        if rti_info['is_rti_query']:
            facts.append('rti_query(user, true).')
            if rti_info['information_sought']:
                facts.append(f"information_type(user, '{rti_info['information_sought']}').")
            if rti_info['response_time_exceeded']:
                facts.append('rti_time_exceeded(user, true).')
            if rti_info['appeal_required']:
                facts.append('rti_appeal_required(user, true).')
        
        # Extract violation details
        violation_info = self.extract_violation_details(query)
        if violation_info['violation_type']:
            facts.append(f"violation_type(user, '{violation_info['violation_type']}').")
        if violation_info['urgency_level'] == 'urgent':
            facts.append('urgent_matter(user, true).')
        if violation_info['complaint_filed']:
            facts.append('complaint_filed(user, true).')
        if violation_info['witnesses']:
            facts.append('witnesses_available(user, true).')
        
        return facts
    
    def analyze_legal_position(self, facts: List[str]) -> Dict[str, Any]:
        """
        Analyze legal position for fundamental rights matters.
        
        Args:
            facts: List of extracted legal facts
            
        Returns:
            Dictionary with legal analysis
        """
        analysis = {
            'constitutional_violation': False,
            'available_writs': [],
            'appropriate_court': None,
            'rti_remedies': [],
            'nhrc_complaint_applicable': False,
            'urgency_level': 'normal',
            'recommended_actions': [],
            'applicable_acts': [],
            'estimated_timeline': None
        }
        
        facts_str = ' '.join(facts)
        
        # Check for constitutional violations
        if 'fundamental_right_involved(' in facts_str:
            analysis['constitutional_violation'] = True
            analysis['appropriate_court'] = 'high_court'
            analysis['available_writs'] = ['habeas_corpus', 'mandamus', 'prohibition']
        
        # Check for discrimination
        if 'discrimination_ground(' in facts_str:
            analysis['available_writs'].append('mandamus')
            analysis['nhrc_complaint_applicable'] = True
        
        # RTI specific remedies
        if 'rti_query(user, true)' in facts_str:
            analysis['rti_remedies'] = ['first_appeal', 'second_appeal_to_cic']
            if 'rti_time_exceeded(user, true)' in facts_str:
                analysis['recommended_actions'].append('File first appeal immediately')
                analysis['estimated_timeline'] = '30 days for first appeal response'
        
        # Check urgency
        if 'urgent_matter(user, true)' in facts_str:
            analysis['urgency_level'] = 'urgent'
            analysis['recommended_actions'].append('File urgent writ petition')
            analysis['estimated_timeline'] = '24-48 hours for urgent hearing'
        
        # Specific violation remedies
        violation_type_match = re.search(r"violation_type\(user, '([^']+)'\)", facts_str)
        if violation_type_match:
            violation_type = violation_type_match.group(1)
            
            if violation_type == 'illegal_detention':
                analysis['available_writs'] = ['habeas_corpus']
                analysis['appropriate_court'] = 'high_court'
                analysis['urgency_level'] = 'urgent'
            elif violation_type == 'discrimination':
                analysis['available_writs'] = ['mandamus']
                analysis['nhrc_complaint_applicable'] = True
            elif violation_type == 'police_harassment':
                analysis['nhrc_complaint_applicable'] = True
                analysis['recommended_actions'].append('File NHRC complaint')
        
        # General recommendations
        if analysis['constitutional_violation']:
            analysis['recommended_actions'].extend([
                'Consult constitutional lawyer',
                'Gather documentary evidence',
                'File writ petition in High Court'
            ])
        
        if analysis['nhrc_complaint_applicable']:
            analysis['recommended_actions'].append('File complaint with National Human Rights Commission')
        
        # Applicable acts and provisions
        analysis['applicable_acts'] = ['Constitution of India']
        
        if 'rti_query(user, true)' in facts_str:
            analysis['applicable_acts'].append('Right to Information Act, 2005')
        
        if analysis['nhrc_complaint_applicable']:
            analysis['applicable_acts'].append('Protection of Human Rights Act, 1993')
        
        return analysis
