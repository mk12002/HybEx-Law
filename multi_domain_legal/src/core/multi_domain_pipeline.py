"""
Multi-Domain Legal Pipeline.

This module orchestrates the complete multi-domain legal analysis pipeline,
including domain classification, specialized processing, and unified reasoning.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from .domain_registry import LegalDomain, DOMAIN_REGISTRY
from .domain_classifier import DomainClassifier
from .text_preprocessor import TextPreprocessor

# Import domain processors
from ..domains.legal_aid.processor import LegalAidProcessor
from ..domains.family_law.processor import FamilyLawProcessor  
from ..domains.consumer_protection.processor import ConsumerProtectionProcessor
from ..domains.fundamental_rights.processor import FundamentalRightsProcessor
from ..domains.employment_law.processor import EmploymentLawProcessor

class MultiDomainLegalPipeline:
    """
    Complete multi-domain legal analysis pipeline.
    
    This pipeline handles legal queries across multiple domains:
    - Legal Aid and Access to Justice
    - Family Law and Personal Status
    - Consumer Protection and Rights
    - Fundamental Rights and Constitutional Law
    - Employment Law and Labor Rights
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize the multi-domain pipeline.
        
        Args:
            confidence_threshold: Minimum confidence for domain classification
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize core components
        self.preprocessor = TextPreprocessor()
        self.domain_classifier = DomainClassifier()
        
        # Initialize domain processors
        self.processors = {
            LegalDomain.LEGAL_AID: LegalAidProcessor(),
            LegalDomain.FAMILY_LAW: FamilyLawProcessor(),
            LegalDomain.CONSUMER_PROTECTION: ConsumerProtectionProcessor(),
            LegalDomain.FUNDAMENTAL_RIGHTS: FundamentalRightsProcessor(),
            LegalDomain.EMPLOYMENT_LAW: EmploymentLawProcessor()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def process_legal_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a legal query through the complete pipeline.
        
        Args:
            query: Legal query in natural language
            user_context: Additional user context (location, demographics, etc.)
            
        Returns:
            Complete analysis results including domain classification,
            fact extraction, legal analysis, and recommendations
        """
        try:
            # Step 1: Preprocess the query
            processed_text = self.preprocessor.preprocess_text(query)
            entities = self.preprocessor.extract_entities(query)
            
            # Step 2: Classify legal domains
            domain_predictions = self.domain_classifier.predict(query)
            
            # Filter domains by confidence threshold
            relevant_domains = [
                domain for domain, confidence in domain_predictions.items()
                if confidence >= self.confidence_threshold
            ]
            
            if not relevant_domains:
                # If no domains meet threshold, use highest confidence domain
                relevant_domains = [max(domain_predictions.items(), key=lambda x: x[1])[0]]
            
            # Step 3: Process query in each relevant domain
            domain_results = {}
            all_facts = []
            
            for domain in relevant_domains:
                if domain in self.processors:
                    processor = self.processors[domain]
                    
                    # Extract domain-specific facts
                    facts = processor.extract_facts(query)
                    all_facts.extend(facts)
                    
                    # Perform domain-specific analysis
                    analysis = processor.analyze_legal_position(facts)
                    
                    domain_results[domain.value] = {
                        'confidence': domain_predictions[domain],
                        'facts': facts,
                        'analysis': analysis,
                        'applicable_acts': self._get_domain_acts(domain)
                    }
            
            # Step 4: Generate unified analysis
            unified_analysis = self._generate_unified_analysis(domain_results, entities, user_context)
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(domain_results, unified_analysis)
            
            # Step 6: Prepare final response
            response = {
                'query': query,
                'processed_text': processed_text,
                'entities': entities,
                'domain_classification': domain_predictions,
                'relevant_domains': [domain.value for domain in relevant_domains],
                'domain_results': domain_results,
                'unified_analysis': unified_analysis,
                'recommendations': recommendations,
                'all_facts': all_facts,
                'confidence_threshold': self.confidence_threshold
            }
            
            self.logger.info(f"Successfully processed query with {len(relevant_domains)} relevant domains")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing legal query: {str(e)}")
            return {
                'error': str(e),
                'query': query,
                'status': 'failed'
            }
    
    def _get_domain_acts(self, domain: LegalDomain) -> List[str]:
        """Get list of acts for a domain"""
        domain_acts = DOMAIN_REGISTRY.get(domain, [])
        return [act.title for act in domain_acts]
    
    def _generate_unified_analysis(self, domain_results: Dict[str, Any], 
                                 entities: Dict[str, List[str]], 
                                 user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate unified analysis across all domains.
        
        Args:
            domain_results: Results from each domain processor
            entities: Extracted entities from query
            user_context: User context information
            
        Returns:
            Unified analysis across domains
        """
        unified = {
            'primary_domain': None,
            'cross_domain_issues': [],
            'legal_complexity': 'low',
            'urgency_level': 'normal',
            'estimated_cost': None,
            'estimated_timeline': None,
            'required_documents': [],
            'potential_outcomes': [],
            'risk_assessment': 'low'
        }
        
        if not domain_results:
            return unified
        
        # Identify primary domain (highest confidence)
        primary_domain = max(domain_results.items(), key=lambda x: x[1]['confidence'])
        unified['primary_domain'] = primary_domain[0]
        
        # Analyze cross-domain issues
        if len(domain_results) > 1:
            unified['cross_domain_issues'] = self._identify_cross_domain_issues(domain_results)
            unified['legal_complexity'] = 'high'
        
        # Aggregate urgency indicators
        urgency_indicators = []
        for domain, results in domain_results.items():
            analysis = results['analysis']
            
            # Check for urgent indicators
            if isinstance(analysis, dict):
                if analysis.get('urgent_matter', False):
                    urgency_indicators.append('urgent_legal_matter')
                if analysis.get('time_barred_risk', False):
                    urgency_indicators.append('limitation_period_risk')
                if 'harassment' in str(analysis).lower():
                    urgency_indicators.append('safety_concern')
        
        if urgency_indicators:
            unified['urgency_level'] = 'high'
        
        # Estimate overall timeline
        timelines = []
        for domain, results in domain_results.items():
            analysis = results['analysis']
            if isinstance(analysis, dict) and analysis.get('timeline_estimate'):
                timelines.append(analysis['timeline_estimate'])
        
        if timelines:
            unified['estimated_timeline'] = max(timelines)  # Take longest estimate
        
        # Aggregate required documents
        all_documents = set()
        for domain, results in domain_results.items():
            analysis = results['analysis']
            if isinstance(analysis, dict) and analysis.get('required_documents'):
                all_documents.update(analysis['required_documents'])
        
        unified['required_documents'] = list(all_documents)
        
        # Assess overall risk
        risk_factors = []
        for domain, results in domain_results.items():
            analysis = results['analysis']
            if isinstance(analysis, dict):
                if analysis.get('case_strength') == 'weak':
                    risk_factors.append('weak_case')
                if analysis.get('evidence_required') == 'substantial':
                    risk_factors.append('evidence_challenges')
                if analysis.get('legal_complexity') == 'high':
                    risk_factors.append('complex_legal_issues')
        
        if len(risk_factors) >= 2:
            unified['risk_assessment'] = 'high'
        elif len(risk_factors) >= 1:
            unified['risk_assessment'] = 'medium'
        
        return unified
    
    def _identify_cross_domain_issues(self, domain_results: Dict[str, Any]) -> List[str]:
        """Identify issues that span multiple legal domains"""
        cross_domain_issues = []
        domains = list(domain_results.keys())
        
        # Common cross-domain scenarios
        if 'family_law' in domains and 'employment_law' in domains:
            cross_domain_issues.append('Work-family conflict (harassment/discrimination affecting family)')
        
        if 'consumer_protection' in domains and 'fundamental_rights' in domains:
            cross_domain_issues.append('Consumer rights violation with constitutional implications')
        
        if 'legal_aid' in domains and any(other in domains for other in ['family_law', 'employment_law', 'consumer_protection']):
            cross_domain_issues.append('Financial constraints affecting access to justice')
        
        if 'employment_law' in domains and 'fundamental_rights' in domains:
            cross_domain_issues.append('Workplace discrimination with constitutional rights implications')
        
        return cross_domain_issues
    
    def _generate_recommendations(self, domain_results: Dict[str, Any], 
                                unified_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            domain_results: Results from domain processors
            unified_analysis: Unified analysis results
            
        Returns:
            Categorized recommendations
        """
        recommendations = {
            'immediate_actions': [],
            'legal_procedures': [],
            'documentation': [],
            'consultation_needed': [],
            'alternative_dispute_resolution': [],
            'preventive_measures': []
        }
        
        # Extract recommendations from each domain
        for domain, results in domain_results.items():
            analysis = results['analysis']
            
            if isinstance(analysis, dict):
                # Immediate actions
                if analysis.get('recommended_actions'):
                    recommendations['immediate_actions'].extend(analysis['recommended_actions'])
                
                # Legal procedures
                if analysis.get('available_forums'):
                    for forum in analysis['available_forums']:
                        recommendations['legal_procedures'].append(f"Consider filing in {forum}")
                
                # Documentation
                if analysis.get('required_documents'):
                    recommendations['documentation'].extend(analysis['required_documents'])
        
        # Add unified recommendations based on complexity
        if unified_analysis['legal_complexity'] == 'high':
            recommendations['consultation_needed'].append('Consult with specialized lawyer due to multi-domain complexity')
        
        if unified_analysis['urgency_level'] == 'high':
            recommendations['immediate_actions'].insert(0, 'Urgent action required - prioritize this matter')
        
        if len(unified_analysis['cross_domain_issues']) > 0:
            recommendations['consultation_needed'].append('Multi-domain legal expertise recommended')
        
        # Alternative dispute resolution suggestions
        if unified_analysis['risk_assessment'] in ['medium', 'high']:
            recommendations['alternative_dispute_resolution'].extend([
                'Consider mediation before litigation',
                'Explore settlement negotiations',
                'Evaluate cost-benefit of legal proceedings'
            ])
        
        # Remove duplicates
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))
        
        return recommendations
    
    def get_domain_summary(self) -> Dict[str, Any]:
        """
        Get summary of available domains and their capabilities.
        
        Returns:
            Dictionary with domain information
        """
        summary = {
            'total_domains': len(self.processors),
            'domains': {}
        }
        
        for domain, processor in self.processors.items():
            acts = self._get_domain_acts(domain)
            summary['domains'][domain.value] = {
                'description': domain.value.replace('_', ' ').title(),
                'applicable_acts': acts,
                'processor_type': type(processor).__name__
            }
        
        return summary
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for domain classification"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of pipeline components"""
        return {
            'preprocessor_ready': self.preprocessor is not None,
            'classifier_ready': self.domain_classifier is not None,
            'active_processors': list(self.processors.keys()),
            'confidence_threshold': self.confidence_threshold,
            'total_domains': len(self.processors)
        }
