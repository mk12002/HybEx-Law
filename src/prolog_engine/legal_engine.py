"""
Legal Aid Engine: Prolog integration for eligibility determination

This module provides the interface between the NLP pipeline and the Prolog
knowledge base for legal aid eligibility determination.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from pyswip import Prolog
    PYSWIP_AVAILABLE = True
except ImportError:
    PYSWIP_AVAILABLE = False
    logging.warning("PySwip not available. Using mock Prolog engine.")


class MockPrologEngine:
    """
    Mock Prolog engine for development and testing when PySwip is not available.
    """
    
    def __init__(self):
        self.facts = []
    
    def assertz(self, fact: str):
        """Mock assertz operation."""
        self.facts.append(fact)
    
    def retractall(self, pattern: str):
        """Mock retractall operation."""
        self.facts = [f for f in self.facts if not f.startswith(pattern.split('(')[0])]
    
    def query(self, query_str: str):
        """Mock query with basic rule simulation."""
        # Simple mock logic for demonstration
        if 'eligible(user)' in query_str:
            # Check for basic eligibility patterns
            has_low_income = any('income_monthly(user, 0)' in f or 
                               'income_monthly(user,' in f and int(f.split(',')[1].strip(' ).')) < 25000
                               for f in self.facts if 'income_monthly' in f)
            
            is_woman = any('is_woman(user, true)' in f for f in self.facts)
            is_sc_st = any('is_sc_st(user, true)' in f for f in self.facts)
            
            # Simple eligibility logic
            eligible = has_low_income or is_woman or is_sc_st
            
            if eligible:
                yield {'eligible': True}
            else:
                return iter([])
        else:
            return iter([])


class LegalAidEngine:
    """
    Interface to the Prolog-based legal aid eligibility system.
    
    This class manages the connection to the Prolog knowledge base and
    provides methods for checking eligibility based on extracted facts.
    """
    
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the legal aid engine.
        
        Args:
            knowledge_base_path: Path to the Prolog knowledge base file
        """
        self.logger = logging.getLogger(__name__)
        
        # Set default knowledge base path
        if knowledge_base_path is None:
            project_root = Path(__file__).parent.parent.parent
            knowledge_base_path = project_root / "knowledge_base" / "legal_aid_eligibility.pl"
        
        self.knowledge_base_path = Path(knowledge_base_path)
        
        # Initialize Prolog engine
        if PYSWIP_AVAILABLE:
            try:
                self.prolog = Prolog()
                self._load_knowledge_base()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Prolog: {e}. Using mock engine.")
                self.prolog = MockPrologEngine()
        else:
            self.prolog = MockPrologEngine()
    
    def _load_knowledge_base(self):
        """Load the Prolog knowledge base."""
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")
        
        try:
            # Convert Windows path to Prolog format
            kb_path_str = str(self.knowledge_base_path).replace('\\', '/')
            self.prolog.consult(kb_path_str)
            self.logger.info(f"Loaded knowledge base: {self.knowledge_base_path}")
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            raise
    
    def check_eligibility(self, facts: List[str]) -> Dict[str, Any]:
        """
        Check legal aid eligibility based on extracted facts.
        
        Args:
            facts: List of Prolog facts as strings
            
        Returns:
            Dictionary with eligibility decision and explanation
        """
        try:
            # Clear previous facts
            self._clear_applicant_facts()
            
            # Assert new facts
            self._assert_facts(facts)
            
            # Query eligibility
            is_eligible = self._query_eligibility()
            
            # Get explanation
            explanation = self._get_explanation(is_eligible)
            
            return {
                'eligible': is_eligible,
                'explanation': explanation,
                'facts_used': facts,
                'additional_info': self._get_additional_info()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking eligibility: {e}")
            return {
                'eligible': False,
                'explanation': f"Error in eligibility check: {str(e)}",
                'facts_used': facts,
                'error': True
            }
    
    def _clear_applicant_facts(self):
        """Clear all existing applicant facts."""
        predicates_to_clear = [
            'applicant(user)',
            'income_monthly(user, _)',
            'income_annual(user, _)',
            'location(user, _)',
            'case_type(user, _)',
            'is_woman(user, _)',
            'is_child(user, _)',
            'is_sc_st(user, _)',
            'is_industrial_workman(user, _)',
            'is_in_custody(user, _)',
            'is_disabled(user, _)',
            'is_disaster_victim(user, _)'
        ]
        
        for predicate in predicates_to_clear:
            try:
                self.prolog.retractall(predicate)
            except:
                # Continue if retract fails (fact might not exist)
                pass
    
    def _assert_facts(self, facts: List[str]):
        """
        Assert facts into the Prolog knowledge base.
        
        Args:
            facts: List of Prolog facts to assert
        """
        for fact in facts:
            # Clean the fact (remove trailing period for assertz)
            clean_fact = fact.rstrip('.')
            
            try:
                self.prolog.assertz(clean_fact)
                self.logger.debug(f"Asserted fact: {clean_fact}")
            except Exception as e:
                self.logger.warning(f"Failed to assert fact '{clean_fact}': {e}")
    
    def _query_eligibility(self) -> bool:
        """
        Query the Prolog system for eligibility.
        
        Returns:
            True if eligible, False otherwise
        """
        try:
            # Query for eligibility
            results = list(self.prolog.query("eligible(user)"))
            return len(results) > 0
        except Exception as e:
            self.logger.error(f"Error querying eligibility: {e}")
            return False
    
    def _get_explanation(self, is_eligible: bool) -> str:
        """
        Get explanation for the eligibility decision.
        
        Args:
            is_eligible: Whether the person is eligible
            
        Returns:
            Human-readable explanation
        """
        try:
            if is_eligible:
                # Query for explanation of eligibility
                explanation_query = 'explanation(user, eligible, Reason)'
                results = list(self.prolog.query(explanation_query))
                if results and 'Reason' in results[0]:
                    return results[0]['Reason']
                else:
                    return "Meets eligibility criteria for legal aid"
            else:
                # Query for explanation of ineligibility
                explanation_query = 'explanation(user, not_eligible, Reason)'
                results = list(self.prolog.query(explanation_query))
                if results and 'Reason' in results[0]:
                    return results[0]['Reason']
                else:
                    return "Does not meet eligibility criteria for legal aid"
        except Exception as e:
            self.logger.error(f"Error getting explanation: {e}")
            if is_eligible:
                return "Eligible for legal aid (explanation unavailable)"
            else:
                return "Not eligible for legal aid (explanation unavailable)"
    
    def _get_additional_info(self) -> Optional[str]:
        """
        Get additional information about the decision.
        
        Returns:
            Additional context or None
        """
        try:
            # Query for available information
            info_query = 'available_info(user, Info)'
            results = list(self.prolog.query(info_query))
            if results and 'Info' in results[0]:
                info_list = results[0]['Info']
                return f"Available information: {len(info_list)} facts processed"
        except:
            pass
        
        return None
    
    def test_knowledge_base(self) -> Dict[str, Any]:
        """
        Test the knowledge base with sample cases.
        
        Returns:
            Test results
        """
        test_results = {}
        
        # Test case 1: Low income person
        test_facts_1 = [
            'applicant(test1)',
            'income_monthly(test1, 15000)',
            'case_type(test1, "property_dispute")',
            'is_woman(test1, false)',
            'is_sc_st(test1, false)'
        ]
        
        try:
            self._clear_test_facts('test1')
            self._assert_facts(test_facts_1)
            eligible = list(self.prolog.query("eligible(test1)"))
            test_results['low_income_case'] = len(eligible) > 0
        except Exception as e:
            test_results['low_income_case'] = f"Error: {e}"
        
        # Test case 2: Woman applicant
        test_facts_2 = [
            'applicant(test2)',
            'income_monthly(test2, 50000)',
            'is_woman(test2, true)',
            'case_type(test2, "domestic_violence")'
        ]
        
        try:
            self._clear_test_facts('test2')
            self._assert_facts(test_facts_2)
            eligible = list(self.prolog.query("eligible(test2)"))
            test_results['woman_applicant_case'] = len(eligible) > 0
        except Exception as e:
            test_results['woman_applicant_case'] = f"Error: {e}"
        
        return test_results
    
    def _clear_test_facts(self, person: str):
        """Clear test facts for a specific person."""
        predicates = [
            f'applicant({person})',
            f'income_monthly({person}, _)',
            f'income_annual({person}, _)',
            f'case_type({person}, _)',
            f'is_woman({person}, _)',
            f'is_sc_st({person}, _)'
        ]
        
        for predicate in predicates:
            try:
                self.prolog.retractall(predicate)
            except:
                pass
