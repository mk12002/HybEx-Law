"""
Employment Law Domain Processor.

This module handles employment and labor law queries including termination,
harassment, wage disputes, and other workplace-related legal issues.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

class EmploymentLawProcessor:
    """
    Processor for employment law domain queries.
    
    Handles workplace issues including wrongful termination, harassment,
    wage disputes, and other employment-related legal matters.
    """
    
    def __init__(self):
        self.employment_types = self._setup_employment_types()
        self.termination_categories = self._setup_termination_categories()
        self.harassment_types = self._setup_harassment_types()
        self.wage_laws = self._setup_wage_laws()
        self.labor_tribunals = self._setup_labor_tribunals()
        self.remedies = self._setup_remedies()
    
    def _setup_employment_types(self) -> Dict[str, List[str]]:
        """Setup employment categories with indicators"""
        return {
            'permanent': ['permanent', 'regular', 'confirmed', 'full-time'],
            'temporary': ['temporary', 'contractual', 'contract', 'fixed-term'],
            'casual': ['casual', 'daily wage', 'part-time', 'hourly'],
            'probationary': ['probation', 'probationary', 'trial period'],
            'apprentice': ['apprentice', 'trainee', 'intern']
        }
    
    def _setup_termination_categories(self) -> Dict[str, Dict[str, Any]]:
        """Setup termination types with legal requirements"""
        return {
            'wrongful_termination': {
                'indicators': ['fired without reason', 'terminated illegally', 'dismissed unfairly'],
                'requirements': ['proper_notice', 'valid_reason', 'due_process'],
                'compensation': 'reinstatement_or_compensation'
            },
            'retrenchment': {
                'indicators': ['layoff', 'retrenchment', 'downsizing', 'closure'],
                'requirements': ['notice_period', 'compensation', 'government_approval'],
                'compensation': 'retrenchment_compensation'
            },
            'misconduct_dismissal': {
                'indicators': ['dismissed for misconduct', 'terminated for cause', 'disciplinary action'],
                'requirements': ['domestic_inquiry', 'show_cause_notice', 'hearing'],
                'compensation': 'none_if_proven'
            },
            'voluntary_resignation': {
                'indicators': ['resigned', 'quit', 'left job'],
                'requirements': ['notice_period', 'handover'],
                'compensation': 'pending_dues_only'
            }
        }
    
    def _setup_harassment_types(self) -> Dict[str, Dict[str, Any]]:
        """Setup workplace harassment categories"""
        return {
            'sexual_harassment': {
                'indicators': ['sexual harassment', 'inappropriate behavior', 'unwelcome advances'],
                'applicable_act': 'Sexual Harassment of Women at Workplace Act, 2013',
                'complaint_mechanism': 'internal_complaints_committee',
                'timeline': '3 months from incident'
            },
            'workplace_bullying': {
                'indicators': ['bullying', 'humiliation', 'verbal abuse', 'mental harassment'],
                'applicable_act': 'Industrial Disputes Act, 1947',
                'complaint_mechanism': 'management_complaint',
                'timeline': 'no_specific_limit'
            },
            'discrimination': {
                'indicators': ['discrimination', 'biased treatment', 'unfair treatment'],
                'applicable_act': 'Equal Remuneration Act, 1976',
                'complaint_mechanism': 'labor_department',
                'timeline': '6 months'
            }
        }
    
    def _setup_wage_laws(self) -> Dict[str, Any]:
        """Setup wage-related laws and thresholds"""
        return {
            'minimum_wage': {
                'central_rates': {
                    'unskilled': 178,  # per day
                    'semi_skilled': 196,
                    'skilled': 215,
                    'highly_skilled': 236
                },
                'overtime_rate': 'double_normal_rate',
                'working_hours': 8,  # per day
                'weekly_limit': 48  # hours
            },
            'equal_pay': {
                'principle': 'equal_work_equal_pay',
                'applicable_to': ['men_women', 'contract_permanent', 'all_categories']
            },
            'bonus': {
                'minimum_percentage': 8.33,
                'maximum_percentage': 20,
                'eligibility': 'worked_30_days_minimum'
            }
        }
    
    def _setup_labor_tribunals(self) -> Dict[str, Dict[str, Any]]:
        """Setup labor dispute resolution mechanisms"""
        return {
            'conciliation': {
                'authority': 'conciliation_officer',
                'timeline': '2_months',
                'mandatory': True
            },
            'labor_court': {
                'jurisdiction': 'individual_disputes',
                'timeline': '6_months_to_2_years',
                'appeal_to': 'high_court'
            },
            'industrial_tribunal': {
                'jurisdiction': 'collective_disputes',
                'timeline': '1_to_3_years',
                'appeal_to': 'high_court'
            }
        }
    
    def _setup_remedies(self) -> Dict[str, List[str]]:
        """Setup available remedies for employment issues"""
        return {
            'wrongful_termination': [
                'reinstatement', 'back_wages', 'compensation', 'benefits_restoration'
            ],
            'sexual_harassment': [
                'written_apology', 'warning', 'transfer', 'termination_of_harasser',
                'compensation_to_victim'
            ],
            'wage_dispute': [
                'payment_of_dues', 'interest_on_delayed_payment', 'penalty_on_employer'
            ],
            'discrimination': [
                'equal_treatment', 'compensation', 'policy_changes', 'awareness_training'
            ]
        }
    
    def identify_employment_type(self, query: str) -> Optional[str]:
        """
        Identify type of employment from query.
        
        Args:
            query: Employment-related query
            
        Returns:
            Employment type or None
        """
        query_lower = query.lower()
        
        for emp_type, indicators in self.employment_types.items():
            if any(indicator in query_lower for indicator in indicators):
                return emp_type
        
        return None
    
    def extract_employment_details(self, query: str) -> Dict[str, Any]:
        """
        Extract employment details from query.
        
        Args:
            query: Employment query
            
        Returns:
            Dictionary with employment information
        """
        employment_info = {
            'employment_type': None,
            'company_name': None,
            'job_duration': None,
            'salary': None,
            'notice_period': None,
            'benefits': []
        }
        
        query_lower = query.lower()
        
        # Identify employment type
        employment_info['employment_type'] = self.identify_employment_type(query)
        
        # Extract salary information
        salary_patterns = [
            r'(?:salary|wage|pay|earning|income).*?(\d+(?:,\d+)*)',
            r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|inr)\s*(?:per month|monthly|salary)',
            r'(\d+(?:,\d+)*)\s*(?:per day|daily|day)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, query_lower)
            if match:
                employment_info['salary'] = int(match.group(1).replace(',', ''))
                break
        
        # Extract job duration
        duration_patterns = [
            r'(?:worked|working|employed).*?(\d+)\s*(?:years?|months?)',
            r'(\d+)\s*(?:years?|months?)\s*(?:of service|experience|job)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                duration = int(match.group(1))
                if 'year' in pattern:
                    employment_info['job_duration'] = f"{duration} years"
                else:
                    employment_info['job_duration'] = f"{duration} months"
                break
        
        # Extract notice period
        notice_patterns = [
            r'(?:notice period|notice).*?(\d+)\s*(?:days?|months?)',
            r'(\d+)\s*(?:days?|months?)\s*notice'
        ]
        
        for pattern in notice_patterns:
            match = re.search(pattern, query_lower)
            if match:
                notice_duration = int(match.group(1))
                if 'month' in pattern:
                    employment_info['notice_period'] = f"{notice_duration} months"
                else:
                    employment_info['notice_period'] = f"{notice_duration} days"
                break
        
        return employment_info
    
    def identify_termination_type(self, query: str) -> Optional[str]:
        """
        Identify type of employment termination.
        
        Args:
            query: Termination-related query
            
        Returns:
            Termination type or None
        """
        query_lower = query.lower()
        
        for termination_type, details in self.termination_categories.items():
            if any(indicator in query_lower for indicator in details['indicators']):
                return termination_type
        
        return None
    
    def identify_harassment_type(self, query: str) -> List[str]:
        """
        Identify types of workplace harassment.
        
        Args:
            query: Harassment-related query
            
        Returns:
            List of harassment types
        """
        harassment_types = []
        query_lower = query.lower()
        
        for harassment_type, details in self.harassment_types.items():
            if any(indicator in query_lower for indicator in details['indicators']):
                harassment_types.append(harassment_type)
        
        return harassment_types
    
    def extract_wage_dispute_details(self, query: str) -> Dict[str, Any]:
        """
        Extract wage dispute details from query.
        
        Args:
            query: Wage dispute query
            
        Returns:
            Dictionary with wage dispute information
        """
        wage_info = {
            'dispute_type': None,
            'unpaid_amount': None,
            'unpaid_duration': None,
            'overtime_claim': False,
            'bonus_claim': False,
            'minimum_wage_violation': False
        }
        
        query_lower = query.lower()
        
        # Identify dispute type
        if any(term in query_lower for term in ['not paid', 'unpaid', 'salary pending']):
            wage_info['dispute_type'] = 'unpaid_wages'
        elif any(term in query_lower for term in ['minimum wage', 'below minimum']):
            wage_info['dispute_type'] = 'minimum_wage_violation'
            wage_info['minimum_wage_violation'] = True
        elif 'overtime' in query_lower:
            wage_info['dispute_type'] = 'overtime_payment'
            wage_info['overtime_claim'] = True  
        elif 'bonus' in query_lower:
            wage_info['dispute_type'] = 'bonus_payment'
            wage_info['bonus_claim'] = True
        
        # Extract unpaid amount
        amount_patterns = [
            r'(?:unpaid|pending|owed).*?(\d+(?:,\d+)*)',
            r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?)\s*(?:unpaid|pending|owed)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                wage_info['unpaid_amount'] = int(match.group(1).replace(',', ''))
                break
        
        # Extract duration
        duration_patterns = [
            r'(\d+)\s*(?:months?)\s*(?:salary|wage|pay)',
            r'(?:salary|wage|pay).*?(\d+)\s*(?:months?|years?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                duration = int(match.group(1))
                wage_info['unpaid_duration'] = f"{duration} months"
                break
        
        return wage_info
    
    def extract_facts(self, query: str) -> List[str]:
        """
        Extract legal facts from employment law query.
        
        Args:
            query: Employment query
            
        Returns:
            List of Prolog facts
        """
        facts = ['employee(user).']
        
        # Extract employment details
        employment_info = self.extract_employment_details(query)
        if employment_info['employment_type']:
            facts.append(f"employment_type(user, '{employment_info['employment_type']}').")
        if employment_info['salary']:
            facts.append(f"monthly_salary(user, {employment_info['salary']}).")
        if employment_info['job_duration']:
            facts.append(f"service_duration(user, '{employment_info['job_duration']}').")
        if employment_info['notice_period']:
            facts.append(f"notice_period(user, '{employment_info['notice_period']}').")
        
        # Extract termination details
        termination_type = self.identify_termination_type(query)
        if termination_type:
            facts.append(f"termination_type(user, '{termination_type}').")
        
        # Extract harassment details
        harassment_types = self.identify_harassment_type(query)
        for harassment_type in harassment_types:
            facts.append(f"harassment_type(user, '{harassment_type}').")
        
        # Extract wage dispute details
        wage_info = self.extract_wage_dispute_details(query)
        if wage_info['dispute_type']:
            facts.append(f"wage_dispute_type(user, '{wage_info['dispute_type']}').")
        if wage_info['unpaid_amount']:
            facts.append(f"unpaid_amount(user, {wage_info['unpaid_amount']}).")
        if wage_info['minimum_wage_violation']:
            facts.append('minimum_wage_violation(user, true).')
        if wage_info['overtime_claim']:
            facts.append('overtime_claim(user, true).')
        if wage_info['bonus_claim']:
            facts.append('bonus_claim(user, true).')
        
        return facts
    
    def analyze_legal_position(self, facts: List[str]) -> Dict[str, Any]:
        """
        Analyze legal position for employment law matters.
        
        Args:
            facts: List of extracted legal facts
            
        Returns:
            Dictionary with legal analysis
        """
        analysis = {
            'wrongful_termination_case': False,
            'harassment_complaint_valid': False,
            'wage_claim_valid': False,
            'available_forums': [],
            'compensation_estimate': None,
            'required_procedures': [],
            'recommended_actions': [],
            'applicable_acts': [],
            'timeline_estimate': None
        }
        
        facts_str = ' '.join(facts)
        
        # Analyze termination cases
        termination_match = re.search(r"termination_type\(user, '([^']+)'\)", facts_str)
        if termination_match:
            termination_type = termination_match.group(1)
            
            if termination_type == 'wrongful_termination':
                analysis['wrongful_termination_case'] = True
                analysis['available_forums'] = ['conciliation', 'labor_court']
                analysis['recommended_actions'].extend([
                    'File complaint with Labor Commissioner',
                    'Gather employment documents',
                    'Calculate back wages and compensation'
                ])
                
                # Estimate compensation
                salary_match = re.search(r'monthly_salary\(user, (\d+)\)', facts_str)
                if salary_match:
                    monthly_salary = int(salary_match.group(1))
                    # Typically 6 months to 2 years compensation
                    analysis['compensation_estimate'] = monthly_salary * 12
        
        # Analyze harassment cases
        harassment_matches = re.findall(r"harassment_type\(user, '([^']+)'\)", facts_str)
        if harassment_matches:
            analysis['harassment_complaint_valid'] = True
            
            if 'sexual_harassment' in harassment_matches:
                analysis['required_procedures'] = ['complaint_to_icc', 'documentary_evidence']
                analysis['recommended_actions'].append('File complaint with Internal Complaints Committee')
                analysis['timeline_estimate'] = '90 days for ICC inquiry'
            
            analysis['available_forums'].append('internal_complaints_committee')
        
        # Analyze wage disputes
        wage_dispute_match = re.search(r"wage_dispute_type\(user, '([^']+)'\)", facts_str)
        if wage_dispute_match:
            analysis['wage_claim_valid'] = True
            analysis['available_forums'].append('labor_court')
            
            wage_dispute_type = wage_dispute_match.group(1)
            if wage_dispute_type == 'unpaid_wages':
                analysis['recommended_actions'].append('File complaint for recovery of wages')
                
                # Calculate claim amount
                unpaid_match = re.search(r'unpaid_amount\(user, (\d+)\)', facts_str)
                if unpaid_match:
                    unpaid_amount = int(unpaid_match.group(1))
                    # Add interest and compensation
                    analysis['compensation_estimate'] = unpaid_amount * 1.25  # 25% additional
        
        # Check minimum wage violations
        if 'minimum_wage_violation(user, true)' in facts_str:
            analysis['wage_claim_valid'] = True
            analysis['recommended_actions'].append('File complaint with Labor Inspector')
            analysis['applicable_acts'].append('Minimum Wages Act, 1948')
        
        # General applicable acts
        if analysis['wrongful_termination_case'] or 'employee(user)' in facts_str:
            analysis['applicable_acts'].append('Industrial Disputes Act, 1947')
        
        if 'sexual_harassment' in harassment_matches:
            analysis['applicable_acts'].append('Sexual Harassment of Women at Workplace Act, 2013')
        
        if any(harassment in harassment_matches for harassment in ['discrimination']):
            analysis['applicable_acts'].append('Equal Remuneration Act, 1976')
        
        # Timeline estimates
        if not analysis['timeline_estimate']:
            if analysis['available_forums']:
                if 'conciliation' in analysis['available_forums']:
                    analysis['timeline_estimate'] = '2 months conciliation + 6-24 months court proceedings'
                elif 'labor_court' in analysis['available_forums']:
                    analysis['timeline_estimate'] = '6 months to 2 years'
        
        return analysis
