# hybex_system/prolog_engine.py

import os
import tempfile
import subprocess
import logging
import importlib.util
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
import re
import uuid
from dataclasses import dataclass, asdict # Added asdict import

from .config import HybExConfig

# Setup logging
logger = logging.getLogger(__name__)
def prolog_escape(text: str) -> str:
    """
    Escape single quotes and backslashes for safe Prolog strings.
    """
    return text.replace("\\", "\\\\").replace("'", "''").replace("\n", " ")

@dataclass
class LegalReasoning:
    case_id: str
    eligible: bool
    confidence: float
    primary_reason: str
    detailed_reasoning: List  # more general, or List[Dict[str, Any]]
    applicable_rules: List[str]
    legal_citations: List[str]
    method: str

@dataclass
class PrologQuery:
    """Structure for Prolog queries"""
    query_text: str
    expected_variables: List[str]
    timeout: int = 60  # Increased from 30 to 60 seconds
    retry_count: int = 3

class PrologEngine:
    def _load_rules(self):
        """
        Loads all Prolog rules from the master knowledge base file.
        """
        kb_path = self.config.BASE_DIR / 'knowledge_base' / 'knowledge_base.pl'

        if not kb_path.exists():
            logger.error(f"Master knowledge base not found at: {kb_path}")
            self.rules_loaded = False
            return
        try:
            # Example for pyswip or similar: self.prolog.consult(str(kb_path))
            # If using subprocess, just store the path for consult in queries
            self.rules_loaded = True
            logger.info(f"✅ Successfully loaded master knowledge base from: {kb_path}")
        except Exception as e:
            logger.error(f"Failed to consult master knowledge base '{kb_path}': {e}", exc_info=True)
            self.rules_loaded = False
    """Advanced Prolog reasoning engine for comprehensive legal analysis"""
    
    def __init__(self, config: HybExConfig):
        self.config = config
        self.prolog_available = self._check_prolog_availability()
        self.rules_loaded = False
        self.temp_files = []
        self.session_id = str(uuid.uuid4())[:8]
        
        self.setup_logging()
        
        # Load comprehensive legal rules from the knowledge base or use fallback
        self.legal_rules = self._load_comprehensive_legal_rules()
        self.income_thresholds = self.config.ENTITY_CONFIG['income_thresholds']
        
        logger.info(f"Initialized Comprehensive PrologEngine (Session: {self.session_id}, Available: {self.prolog_available})")
        logger.info(f"Loaded {len(self.legal_rules)} rule categories with {sum(len(rules) for rules in self.legal_rules.values())} total rules")
        if any(self.legal_rules.get(k) for k in ['legal_aid_rules', 'family_law_rules', 'reasoning_rules']): # Added reasoning_rules
            self.rules_loaded = True
        logger.info(f"Comprehensive rules actually loaded: {self.rules_loaded}")

    def setup_logging(self):
        """Setup comprehensive Prolog engine logging"""
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'prolog_reasoning_{self.session_id}.log') for h in logger.handlers):
            log_file = self.config.get_log_path(f'prolog_reasoning_{self.session_id}')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')

            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                f'%(asctime)s - %(name)s - %(levelname)s - [Session:{self.session_id}] %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to PrologEngine logger.")
        
        logger.info("="*70)
        logger.info("Starting HybEx-Law Comprehensive Prolog Reasoning Engine")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Configuration: {self.config.get_legal_data_status()}")
        logger.info("="*70)
    
    def _check_prolog_availability(self) -> bool:
        """Check if SWI-Prolog is available with detailed diagnostics"""
        try:
            result = subprocess.run(['swipl', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_info = result.stdout.strip().split('\n')[0]
                logger.info(f"SWI-Prolog available: {version_info}")
                
                test_result = subprocess.run(['swipl', '-q', '-t', 'halt.'], 
                                           capture_output=True, text=True, timeout=5)
                if test_result.returncode == 0:
                    logger.info("SWI-Prolog basic functionality verified")
                    return True
                else:
                    logger.warning(f"⚠️ SWI-Prolog basic test failed: {test_result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ SWI-Prolog availability check timed out")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ SWI-Prolog process error: {e.stderr.strip()}")
        except FileNotFoundError:
            logger.warning("⚠️ SWI-Prolog not found in system PATH. Please install SWI-Prolog.")
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error checking SWI-Prolog: {e}")
        
        logger.warning("⚠️ SWI-Prolog not available. Using advanced fallback reasoning.")
        return False
    
    def _load_comprehensive_legal_rules(self) -> Dict[str, List[str]]:
        """Load comprehensive legal rules from multiple sources, prioritizing knowledge base."""
        
        comprehensive_rules_str = self._load_multi_domain_rules()
        
        if comprehensive_rules_str:
            logger.info("Parsing comprehensive multi-domain legal rules from knowledge base.")
            try:
                rule_sections = {}
                
                section_regex_map = {
                    'legal_aid_rules': r'% LEGAL AID DOMAIN RULES(.*?)(?:% FAMILY LAW DOMAIN RULES|% CONSUMER PROTECTION DOMAIN RULES|% EMPLOYMENT LAW DOMAIN RULES|% FUNDAMENTAL RIGHTS DOMAIN RULES|% REASONING RULES|\Z)',
                    'family_law_rules': r'% FAMILY LAW DOMAIN RULES(.*?)(?:% CONSUMER PROTECTION DOMAIN RULES|% EMPLOYMENT LAW DOMAIN RULES|% FUNDAMENTAL RIGHTS DOMAIN RULES|% REASONING RULES|\Z)',
                    'consumer_protection_rules': r'% CONSUMER PROTECTION DOMAIN RULES(.*?)(?:% EMPLOYMENT LAW DOMAIN RULES|% FUNDAMENTAL RIGHTS DOMAIN RULES|% REASONING RULES|\Z)',
                    'employment_law_rules': r'% EMPLOYMENT LAW DOMAIN RULES(.*?)(?:% FUNDAMENTAL RIGHTS DOMAIN RULES|% REASONING RULES|\Z)',
                    'fundamental_rights_rules': r'% FUNDAMENTAL RIGHTS DOMAIN RULES(.*?)(?:% REASONING RULES|\Z)',
                    'reasoning_rules': r'% REASONING RULES(.*?)(?:\Z)'
                }

                for key, pattern in section_regex_map.items():
                    match = re.search(pattern, comprehensive_rules_str, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content:
                            individual_rules = self._parse_prolog_rules_from_section(content)
                            rule_sections[key] = individual_rules
                            logger.debug(f"Extracted section: {key} with {len(individual_rules)} rules")
                
                rule_sections['eligibility_rules'] = self._extract_eligibility_rules(comprehensive_rules_str)
                # The reasoning rules are now extracted directly from the main rule string
                rule_sections['threshold_rules'] = self._generate_threshold_rules()
                rule_sections['meta_rules'] = self._generate_meta_rules()

                self.rules_loaded = True
                return rule_sections

            except Exception as e:
                logger.error(f"Error parsing comprehensive rules from knowledge base: {e}. Falling back to hardcoded rules.", exc_info=True)
                self.rules_loaded = False
                return self._get_enhanced_fallback_rules()
        else:
            logger.warning("No multi-domain rules found in knowledge base. Using enhanced fallback rules.")
            self.rules_loaded = False
            return self._get_enhanced_fallback_rules()
    
    def _detect_query_domain(self, extracted_entities: Dict[str, Any]) -> str:
        """Detect the legal domain based on case type and entities."""
        case_type = extracted_entities.get('case_type', '').lower()
        
        # Domain mapping based on case type
        domain_mappings = {
            'family': 'family_law',
            'divorce': 'family_law',
            'custody': 'family_law',
            'maintenance': 'family_law',
            'domestic_violence': 'family_law',
            
            'consumer': 'consumer_protection',
            'fraud': 'consumer_protection',
            'defective_goods': 'consumer_protection',
            'service_deficiency': 'consumer_protection',
            
            'employment': 'employment_law',
            'wrongful_termination': 'employment_law',
            'wage_dispute': 'employment_law',
            'harassment': 'employment_law',
            
            'criminal': 'criminal_law',
            'bail': 'criminal_law',
            'defense': 'criminal_law'
        }
        
        detected_domain = domain_mappings.get(case_type, 'general_legal_aid')
        logger.info(f"🎯 Detected domain '{detected_domain}' for case type '{case_type}'")
        return detected_domain

    def _load_domain_specific_rules(self, domain: str) -> Optional[str]:
        """Load only rules specific to the detected domain."""
        try:
            # For now, always use the full multi-domain rules to ensure all predicates are available
            logger.info(f"📋 Loading full rule set for domain '{domain}' to ensure all predicates are available")
            return self._load_multi_domain_rules()
            
        except Exception as e:
            logger.error(f"Error loading domain-specific rules for '{domain}': {e}")
            return self._load_multi_domain_rules()  # Fallback to full rule set

    def _load_multi_domain_rules(self) -> Optional[str]:
        """Load rules from the new modular knowledge base entry point (knowledge_base.pl)."""
        try:
            kb_path = Path(__file__).parent.parent / "knowledge_base" / "knowledge_base.pl"
            if kb_path.exists():
                logger.info(f"Using modular knowledge base entry point: {kb_path}")
                with open(kb_path, 'r', encoding='utf-8') as kb_file:
                    rules_str = kb_file.read()
                return rules_str
            else:
                logger.error("Critical error: knowledge_base.pl not found!")
        except Exception as e:
            logger.error(f"Error loading modular knowledge base: {e}", exc_info=True)
        logger.warning("No modular knowledge base found. Using enhanced fallback rules.")
        return self._get_enhanced_fallback_rules()

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a specific section from comprehensive rules text."""
        pattern = rf'% {section_name}.*?(?=% [A-Z_\\s]+(?:RULES)?|\\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return None

    def _extract_legal_aid_rules(self, comprehensive_rules: str) -> List[str]:
        """Extract legal aid domain rules. If not found, use basic fallback."""
        section = self._extract_section(comprehensive_rules, 'LEGAL AID DOMAIN RULES')
        return [section] if section else self._get_basic_legal_aid_rules()
    
    def _extract_family_law_rules(self, comprehensive_rules: str) -> List[str]:
        """Extract family law domain rules."""
        section = self._extract_section(comprehensive_rules, 'FAMILY LAW DOMAIN RULES')
        return [section] if section else []
    
    def _extract_consumer_protection_rules(self, comprehensive_rules: str) -> List[str]:
        """Extract consumer protection domain rules."""
        section = self._extract_section(comprehensive_rules, 'CONSUMER PROTECTION DOMAIN RULES')
        return [section] if section else []
    
    def _extract_employment_law_rules(self, comprehensive_rules: str) -> List[str]:
        """Extract employment law domain rules."""
        section = self._extract_section(comprehensive_rules, 'EMPLOYMENT LAW DOMAIN RULES')
        return [section] if section else []
    
    def _extract_fundamental_rights_rules(self, comprehensive_rules: str) -> List[str]:
        """Extract fundamental rights domain rules."""
        section = self._extract_section(comprehensive_rules, 'FUNDAMENTAL RIGHTS DOMAIN RULES')
        return [section] if section else []
    
    def _parse_prolog_rules_from_section(self, section_content: str) -> List[str]:
        """Parse individual Prolog rules from a section content."""
        if not section_content.strip():
            return []
            
        # Split by lines and process
        lines = section_content.split('\n')
        rules = []
        current_rule = []
        in_comment_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                if current_rule:
                    current_rule.append('')
                continue
                
            # Handle comment blocks
            if stripped.startswith('%'):
                if current_rule:
                    # End current rule if we hit a new comment block
                    rule_text = '\n'.join(current_rule).strip()
                    if rule_text and not rule_text.startswith('%'):
                        rules.append(rule_text)
                    current_rule = []
                in_comment_block = True
                continue
            else:
                in_comment_block = False
                
            # Add line to current rule
            current_rule.append(line)
            
            # Check if this line ends a rule (contains a period at the end of a meaningful statement)
            if (stripped.endswith('.') and 
                not stripped.endswith('..') and  # Not ellipsis
                len(stripped) > 1 and
                not stripped.startswith('%')):  # Not a comment
                
                # End current rule
                rule_text = '\n'.join(current_rule).strip()
                if rule_text and not rule_text.startswith('%'):
                    rules.append(rule_text)
                current_rule = []
        
        # Add any remaining rule
        if current_rule:
            rule_text = '\n'.join(current_rule).strip()
            if rule_text and not rule_text.startswith('%'):
                rules.append(rule_text)
        
        return rules
    
    def _extract_eligibility_rules(self, comprehensive_rules: str) -> List[str]:
        """Extract and enhance eligibility rules. Will contain general eligibility predicates."""
        extracted_rules = []
        legal_aid_section = self._extract_section(comprehensive_rules, 'LEGAL AID DOMAIN RULES')
        if legal_aid_section:
            predicate_patterns = [
                r'eligible_for_legal_aid\(.*?\)\s*:-.*?\.',
                r'income_eligible\(.*?\)\s*:-.*?\.',
                r'categorically_eligible\(.*?\)\s*:-.*?\.',
                r'vulnerable_group\(.*?\)\s*:-.*?\.',
                r'legal_aid_applicable\(.*?\)\s*:-.*?\.',
                r'covered_case_type\(.*?\)\.',
                r'excluded_case_type\(.*?\)\.',
                r'disqualified\(.*?\)\s*:-.*?\.',
                r'income_threshold\(.*?\)\.'
            ]
            for pattern in predicate_patterns:
                matches = re.findall(pattern, legal_aid_section, re.DOTALL)
                extracted_rules.extend(matches)

        enhanced_synthesized = [
            '''
            % Enhanced eligibility determination with confidence scoring
            eligible_with_confidence(Person, Eligible, Confidence) :-
                eligible_for_legal_aid(Person),
                Eligible = true,
                confidence_score(Person, Confidence).
            
            eligible_with_confidence(Person, Eligible, Confidence) :-
                \\+ eligible_for_legal_aid(Person),
                Eligible = false,
                Confidence = 0.9. % Default confidence for ineligibility

            confidence_score(Person, Confidence) :-
                findall(Factor, legal_aid_factor(Person, Factor), Factors),
                length(Factors, Count),
                (Count >= 3 -> Confidence = 0.95;
                 Count == 2 -> Confidence = 0.85;
                 Count == 1 -> Confidence = 0.75;
                 Confidence = 0.6).
            
            legal_aid_factor(Person, income) :-
                income_eligible(Person).
            
            legal_aid_factor(Person, category) :-
                categorically_eligible(Person).
            
            legal_aid_factor(Person, case_type) :-
                legal_aid_applicable(Person, _).

            legal_aid_factor(Person, vulnerability) :-
                vulnerable_group(Person, _).
            '''
        ]
        
        combined_rules_set = set(extracted_rules)
        combined_rules_set.update(enhanced_synthesized)
        
        return list(combined_rules_set)
    
    def _extract_reasoning_rules(self, comprehensive_rules: str) -> List[str]:
        """Generate comprehensive reasoning rules based on extracted predicates."""
        extracted_reasoning = []
        reasoning_section = self._extract_section(comprehensive_rules, 'REASONING RULES')
        if reasoning_section:
            matches = re.findall(r'\\w+_reasoning\\(.*?\)\s*:-.*?\.', reasoning_section, re.DOTALL)
            extracted_reasoning.extend(matches)

        synthesized_reasoning = [
            '''
            % Comprehensive legal reasoning predicates
            legal_reasoning(Person, CaseType, Reasoning) :-
                eligibility_reasoning(Person, EligibilityReason),
                case_type_reasoning(Person, CaseType, CaseReason),
                (EligibilityReason = '' -> Reasoning = CaseReason;
                 CaseReason = '' -> Reasoning = EligibilityReason;
                 format(atom(Reasoning), '~w; ~w', [EligibilityReason, CaseReason])).
            
            eligibility_reasoning(Person, Reason) :-
                income_eligible(Person),
                annual_income(Person, Income),
                applicant_social_category(Person, Category),
                income_threshold(Category, Threshold),
                format(atom(Reason), 'Income eligible: Annual income ₹~w is below threshold ₹~w for ~w category', [Income, Threshold, Category]).
            
            eligibility_reasoning(Person, Reason) :-
                categorically_eligible(Person),
                social_category(Person, Category),
                format(atom(Reason), 'Categorically eligible: Belongs to ~w category', [Category]).

            eligibility_reasoning(Person, 'Not eligible based on income or category.') :-
                \\+ income_eligible(Person),
                \\+ categorically_eligible(Person).
            
            case_type_reasoning(Person, CaseType, Reason) :-
                legal_aid_applicable(Person, CaseType),
                format(atom(Reason), 'Case type ~w is covered under legal aid provisions', [CaseType]).
            
            case_type_reasoning(Person, CaseType, Reason) :-
                excluded_case_type(CaseType),
                format(atom(Reason), 'Case type ~w is excluded from legal aid', [CaseType]).

            case_type_reasoning(Person, 'Case type eligibility not specified or unknown.') :-
                \\+ (legal_aid_applicable(Person, _); excluded_case_type(_)).
            
            generate_detailed_reasoning(Person, DetailedReason) :-
                findall(Reason, individual_reason(Person, Reason), Reasons),
                (Reasons = [] -> DetailedReason = 'No specific detailed reasons found';
                 atomic_list_concat(Reasons, '; ', DetailedReason)).
            
            individual_reason(Person, Reason) :-
                annual_income(Person, Income),
                applicant_social_category(Person, Category),
                income_threshold(Category, Threshold),
                Income =< Threshold,
                format(atom(Reason), 'Income ₹~w ≤ ₹~w threshold for ~w category', [Income, Threshold, Category]).
            
            individual_reason(Person, Reason) :-
                applicant_social_category(Person, Category),
                (Category = 'Scheduled Caste'; Category = 'Scheduled Tribe'; Category = 'Below Poverty Line'),
                format(atom(Reason), 'Automatic eligibility for ~w category', [Category]).
            
            individual_reason(Person, Reason) :-
                case_type(Person, CaseType),
                covered_case_type(CaseType),
                format(atom(Reason), '~w cases are covered under legal aid', [CaseType]).
            '''
        ]
        
        combined_reasoning_set = set(extracted_reasoning)
        combined_reasoning_set.update(synthesized_reasoning)
        return list(combined_reasoning_set)
    
    def _generate_threshold_rules(self) -> List[str]:
        """Generate dynamic threshold rules from configuration (HybExConfig)."""
        threshold_rules = [
            '% Dynamic income thresholds based on current configuration'
        ]
        
        for category, threshold in self.config.ENTITY_CONFIG['income_thresholds'].items():
            threshold_rules.append(f"income_threshold('{category}', {threshold}).")
        
        threshold_rules.append('''
        % Threshold validation and comparison
        income_within_threshold(Person, Category) :-
            annual_income(Person, Income),
            income_threshold(Category, Threshold),
            Income =< Threshold.
        
        income_ratio(Person, Category, Ratio) :-
            annual_income(Person, Income),
            income_threshold(Category, Threshold),
            (Threshold > 0 -> Ratio is Income / Threshold ; Ratio = 0.0).
        ''')
        
        return threshold_rules
    
    def _generate_meta_rules(self) -> List[str]:
        """Generate meta-rules for advanced reasoning and rule application."""
        return [
            '''
            % Meta-rules for rule application and conflict resolution
            applicable_rule(Person, RuleName, Domain) :-
                ( (RuleName = income_eligibility, income_eligible(Person), Domain = legal_aid);
                  (RuleName = categorical_eligibility, categorically_eligible(Person), Domain = legal_aid);
                  (RuleName = case_type_eligibility, legal_aid_applicable(Person, _), Domain = legal_aid)
                ).

            rule_priority(legal_aid, categorical_eligibility, 10).
            rule_priority(legal_aid, income_eligibility, 8).
            rule_priority(legal_aid, case_type_eligibility, 6).
            
            best_applicable_rule(Person, Domain, BestRule) :-
                findall(Priority-Rule, 
                       (applicable_rule(Person, Rule, Domain), 
                        rule_priority(Domain, Rule, Priority)), 
                       Rules),
                sort(Rules, SortedRules),
                (SortedRules = [] -> BestRule = 'no_applicable_rule';
                 last(SortedRules, _Priority-BestRule)).
            
            resolve_conflict(Person, Rules, ResolvedRule) :-
                member(categorical_eligibility, Rules),
                ResolvedRule = categorical_eligibility.
            
            resolve_conflict(Person, Rules, ResolvedRule) :-
                \\+ member(categorical_eligibility, Rules),
                member(income_eligibility, Rules),
                ResolvedRule = income_eligibility.

            resolve_conflict(Person, Rules, 'multiple_rules_apply') :-
                length(Rules, N), N > 1.
            '''
        ]
    
    def _get_enhanced_fallback_rules(self) -> Dict[str, List[str]]:
        """Enhanced fallback rules when comprehensive rules cannot be loaded."""
        logger.warning("Using enhanced fallback rules as multi_domain_rules.py could not be loaded or parsed correctly.")
        
        return {
            'legal_aid_rules': self._get_basic_legal_aid_rules(),
            'eligibility_rules': self._get_basic_eligibility_rules(),
            'reasoning_rules': self._get_basic_reasoning_rules(),
            'threshold_rules': self._generate_threshold_rules(),
            'meta_rules': self._generate_meta_rules(),
            'family_law_rules': [],
            'consumer_protection_rules': [],
            'employment_law_rules': [],
            'fundamental_rights_rules': []
        }
    
    def _get_basic_legal_aid_rules(self) -> List[str]:
        return [
            '''
            % Basic Legal Aid Eligibility Rules (fallback)
            eligible_for_legal_aid(Person) :-
                person(Person),
                annual_income(Person, Income),
                applicant_social_category(Person, Category),
                income_threshold(Category, Threshold),
                Income =< Threshold.
            
            eligible_for_legal_aid(Person) :-
                person(Person),
                applicant_social_category(Person, 'Scheduled Caste').
            eligible_for_legal_aid(Person) :-
                person(Person),
                applicant_social_category(Person, 'Scheduled Tribe').
            eligible_for_legal_aid(Person) :-
                person(Person),
                applicant_social_category(Person, 'Below Poverty Line').
            '''
        ]

    def _get_basic_eligibility_rules(self) -> List[str]:
        return [
            '''
            % Basic Eligibility Predicates for fallback
            income_eligible(Person) :-
                annual_income(Person, Income),
                applicant_social_category(Person, Category),
                income_threshold(Category, Threshold),
                Income =< Threshold.

            categorically_eligible(Person) :-
                applicant_social_category(Person, 'Scheduled Caste').
            categorically_eligible(Person) :-
                applicant_social_category(Person, 'Scheduled Tribe').
            categorically_eligible(Person) :-
                applicant_social_category(Person, 'Below Poverty Line').

            % These must also be present for confidence scoring, even in fallback
            eligible_with_confidence(Person, Eligible, Confidence) :-
                eligible_for_legal_aid(Person),
                Eligible = true,
                confidence_score(Person, Confidence).
            
            eligible_with_confidence(Person, Eligible, Confidence) :-
                \\+ eligible_for_legal_aid(Person),
                Eligible = false,
                Confidence = 0.9. % Default confidence for ineligibility

            confidence_score(Person, Confidence) :-
                findall(Factor, legal_aid_factor(Person, Factor), Factors),
                length(Factors, Count),
                (Count >= 3 -> Confidence = 0.95;
                 Count == 2 -> Confidence = 0.85;
                 Count == 1 -> Confidence = 0.75;
                 Confidence = 0.6).
            
            legal_aid_factor(Person, income) :-
                income_eligible(Person).
            
            legal_aid_factor(Person, category) :-
                categorically_eligible(Person).
            
            legal_aid_factor(Person, case_type) :-
                legal_aid_applicable(Person, _).

            legal_aid_factor(Person, vulnerability) :-
                vulnerable_group(Person, _).
            '''
        ]
    
    def _get_basic_reasoning_rules(self) -> List[str]:
        return [
            '''
            % Basic Reasoning Predicates for fallback
            primary_eligibility_reason(Person, Reason) :-
                categorically_eligible(Person),
                applicant_social_category(Person, Category),
                format(atom(Reason), 'Categorically eligible: Due to %w status.', [Category]).
            primary_eligibility_reason(Person, Reason) :-
                \\+ categorically_eligible(Person),
                income_eligible(Person),
                annual_income(Person, Income),
                applicant_social_category(Person, Category),
                income_threshold(Category, Threshold),
                format(atom(Reason), 'Income eligible: Annual income Rs. %w is below threshold Rs. %w for %w category.', [Income, Threshold, Category]).
            primary_eligibility_reason(Person, Reason) :-
                \\+ eligible_for_legal_aid(Person),
                Reason = 'Not eligible: Does not meet income or categorical criteria.'.
            
            generate_detailed_reasoning(Person, DetailedReason) :-
                findall(Reason, individual_reason(Person, Reason), Reasons),
                (Reasons = [] -> DetailedReason = 'No specific detailed reasons found';
                 atomic_list_concat(Reasons, '; ', DetailedReason)).
            
            individual_reason(Person, Reason) :-
                annual_income(Person, Income),
                applicant_social_category(Person, Category),
                income_threshold(Category, Threshold),
                Income =< Threshold,
                format(atom(Reason), 'Income condition met: Annual income (Rs. %w) is within the legal aid threshold (Rs. %w) for %w category.', [Income, Threshold, Category]).
            
            individual_reason(Person, Reason) :-
                applicant_social_category(Person, Category),
                (Category = 'Scheduled Caste'; Category = 'Scheduled Tribe'; Category = 'Below Poverty Line'),
                format(atom(Reason), 'Categorical eligibility: Applicant belongs to %w, which grants automatic legal aid.', [Category]).
            
            individual_reason(Person, Reason) :-
                case_type(Person, Type),
                (Type = 'Criminal' ; Type = 'Family_Law'), % Example basic covered case types
                format(atom(Reason), 'Case type covered: Legal aid is applicable for %w cases.', [Type]).
            '''
        ]

    def create_domain_specific_prolog_file_robust(self, facts: List[str], domain: str) -> str:
        """Create a robust Prolog file with MINIMAL rules for fast execution."""
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8')
        self.temp_files.append(temp_file.name)
        
        try:
            # Header
            temp_file.write(f'% HybEx-Law Minimal Legal Reasoning Session: {self.session_id}\n')
            temp_file.write(f'% Domain: {domain}\n')
            temp_file.write(f'% Generated: {datetime.now().isoformat()}\n\n')
            
            # Add essential discontiguous directives to fix warnings
            temp_file.write('% DISCONTIGUOUS DIRECTIVES\n')
            temp_file.write(':- discontiguous person/1.\n')
            temp_file.write(':- discontiguous applicable_rule/3.\n')
            temp_file.write(':- discontiguous eligible_for_legal_aid/1.\n')
            temp_file.write(':- discontiguous categorically_eligible/1.\n')
            temp_file.write(':- discontiguous income_eligible/1.\n')
            temp_file.write(':- discontiguous social_category/2.\n')
            temp_file.write(':- discontiguous annual_income/2.\n')
            temp_file.write(':- discontiguous income_threshold/2.\n\n')
            
            # 🚀 CRITICAL OPTIMIZATION: Use only ESSENTIAL rules for this case
            temp_file.write('% MINIMAL ESSENTIAL RULES\n')
            
            # Utility predicates (required)
            temp_file.write('member(X, [X|_]).\n')
            temp_file.write('member(X, [_|T]) :- member(X, T).\n\n')
            
            # Core eligibility rules (optimized with cuts)
            temp_file.write('% OPTIMIZED CORE ELIGIBILITY RULES\n')
            temp_file.write('eligible_for_legal_aid(Person) :- categorically_eligible(Person), !.\n')
            temp_file.write('eligible_for_legal_aid(Person) :- income_eligible(Person), !.\n\n')
            
            # Fast categorical eligibility (check first - most definitive)
            temp_file.write('categorically_eligible(Person) :- social_category(Person, sc), !.\n')
            temp_file.write('categorically_eligible(Person) :- social_category(Person, st), !.\n')
            temp_file.write('categorically_eligible(Person) :- social_category(Person, obc), !.\n\n')
            
            # Dynamic income thresholds from config
            temp_file.write('% DYNAMIC INCOME THRESHOLDS\n')
            for category, threshold in self.income_thresholds.items():
                # Handle combined 'sc_st' category from config
                if category == 'sc_st':
                    temp_file.write(f"income_threshold('sc', {threshold}).\n")
                    temp_file.write(f"income_threshold('st', {threshold}).\n")
                else:
                    temp_file.write(f"income_threshold('{category.lower()}', {threshold}).\n")
            temp_file.write('\n')

            # Income eligibility using dynamic thresholds
            temp_file.write('% INCOME ELIGIBILITY (dynamic thresholds)\n')
            temp_file.write('income_eligible(Person) :-\n')
            temp_file.write('    social_category(Person, Category),\n')
            temp_file.write('    income_threshold(Category, Threshold),\n')
            temp_file.write('    annual_income(Person, Income),\n')
            temp_file.write('    Income =< Threshold, !.\n')
            
            # Fallback for cases with no social category
            temp_file.write('income_eligible(Person) :-\n')
            temp_file.write('    \\+ social_category(Person, _),\n')
            temp_file.write("    income_threshold(general, Threshold),\n")
            temp_file.write('    annual_income(Person, Income),\n')
            temp_file.write('    Income =< Threshold, !.\n\n')
            
            # Simple reasoning predicates
            temp_file.write('% SIMPLE REASONING\n')
            temp_file.write("primary_eligibility_reason(Person, 'Eligible: SC/ST/OBC category') :- categorically_eligibile(Person), !.\n")
            temp_file.write("primary_eligibility_reason(Person, 'Eligible: Income below threshold') :- income_eligible(Person), !.\n")
            temp_file.write("primary_eligibility_reason(Person, 'Not eligible') :- \\+ eligible_for_legal_aid(Person), !.\n\n")
            
            temp_file.write("generate_detailed_reasoning(Person, 'Legal aid evaluation completed') :- person(Person), !.\n\n")
            
            # Add missing applicable_rule predicates to fix Unknown procedure error
            temp_file.write('% APPLICABLE RULES (to fix missing predicate error)\n')
            temp_file.write("applicable_rule(Person, 'categorical_eligibility', legal_aid) :- categorically_eligible(Person), !.\n")
            temp_file.write("applicable_rule(Person, 'income_eligibility', legal_aid) :- income_eligible(Person), !.\n")
            temp_file.write("applicable_rule(Person, 'not_eligible', legal_aid) :- \\+ eligible_for_legal_aid(Person), !.\n")
            temp_file.write("applicable_rule(_, _, _) :- fail.\n\n")
            
            # Write facts
            temp_file.write('% CASE FACTS\n')
            for fact in facts:
                clean_fact = fact.strip()
                if clean_fact and not clean_fact.startswith('%'):
                    if not clean_fact.endswith('.'):
                        temp_file.write(f"{clean_fact}.\n")
                    else:
                        temp_file.write(f"{clean_fact}\n")
            
            temp_file.close()
            
            # DEBUG: Show the minimal file
            logger.info("🔍 DEBUG: Minimal Prolog file content:")
            try:
                with open(temp_file.name, 'r', encoding='utf-8') as debug_file:
                    content = debug_file.read()
                    logger.info(f"Minimal file ({len(content)} chars):\n{content}")
            except Exception as debug_e:
                logger.error(f"Failed to read debug content: {debug_e}")
            
            logger.info(f"✅ Created MINIMAL Prolog file: {temp_file.name} with essential rules only")
            
            return temp_file.name
            
        except Exception as e:
            temp_file.close()
            logger.error(f"Error creating minimal Prolog file: {e}", exc_info=True)
            return None

    def create_domain_specific_prolog_file_modular(self, facts: List[str], domain: str) -> str:
        """🚀 MODULAR APPROACH: Create Prolog file by loading only domain-specific .pl files."""
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8')
        self.temp_files.append(temp_file.name)
        
        try:
            # Header
            temp_file.write(f'% HybEx-Law Modular Legal Reasoning Session: {self.session_id}\n')
            temp_file.write(f'% Domain: {domain}\n')
            temp_file.write(f'% Generated: {datetime.now().isoformat()}\n\n')
            
            # Essential discontiguous directives
            temp_file.write('% DISCONTIGUOUS DIRECTIVES\n')
            essential_directives = [
                ':- discontiguous person/1.',
                ':- discontiguous eligible_for_legal_aid/1.',
                ':- discontiguous categorically_eligible/1.',
                ':- discontiguous income_eligible/1.',
                ':- discontiguous social_category/2.',
                ':- discontiguous annual_income/2.',
                ':- discontiguous case_type/2.',
                ':- discontiguous generate_detailed_reasoning/2.',
                ':- discontiguous primary_eligibility_reason/2.',
                ':- discontiguous applicable_rule/3.',
                ':- discontiguous vulnerable_group/2.',
                ':- discontiguous gender/2.',
                ':- discontiguous age/2.'
            ]
            for directive in essential_directives:
                temp_file.write(directive + '\n')
            temp_file.write('\n')
            
            # 🚀 MODULAR LOADING: Load only required domain files
            domain_files = self._get_domain_files(domain)
            
            for domain_file in domain_files:
                domain_file_path = Path(__file__).parent.parent / "knowledge_base" / domain_file
                if domain_file_path.exists():
                    temp_file.write(f'% LOADING: {domain_file}\n')
                    with open(domain_file_path, 'r', encoding='utf-8') as df:
                        content = df.read()
                        temp_file.write(content)
                        temp_file.write('\n')
                    logger.info(f"✅ Loaded modular file: {domain_file}")
                else:
                    logger.warning(f"⚠️ Domain file not found: {domain_file}")
            
            # Write case facts
            temp_file.write('\n% CASE FACTS\n')
            for fact in facts:
                clean_fact = fact.strip()
                if clean_fact and not clean_fact.startswith('%'):
                    if not clean_fact.endswith('.'):
                        temp_file.write(f"{clean_fact}.\n")
                    else:
                        temp_file.write(f"{clean_fact}\n")
            
            temp_file.close()
            
            # DEBUG output
            logger.info("🔍 DEBUG: Modular Prolog file content:")
            try:
                with open(temp_file.name, 'r', encoding='utf-8') as debug_file:
                    content = debug_file.read()
                    logger.info(f"Modular file ({len(content)} chars):\n{content[:1000]}...")  # First 1000 chars only
            except Exception as debug_e:
                logger.error(f"Failed to read debug content: {debug_e}")
            
            logger.info(f"✅ Created MODULAR Prolog file: {temp_file.name} with {len(domain_files)} domain modules")
            
            return temp_file.name
            
        except Exception as e:
            temp_file.close()
            logger.error(f"Error creating modular Prolog file: {e}", exc_info=True)
            return None

    def _get_domain_files(self, domain: str) -> List[str]:
        """Get the required .pl files for a specific domain."""
        
        # Always include foundational rules first (contains essential predicates)
        base_files = []  # Let domain_mappings handle file loading completely
        
        # Define domain-to-file mappings (only load what's needed!)
        domain_mappings = {
            'legal_aid': ['foundational_rules_clean.pl', 'legal_aid_clean_v2.pl'],
            'family_law': ['foundational_rules_clean.pl', 'legal_aid_clean_v2.pl', 'family_law.pl', 'cross_domain_rules.pl'],
            'consumer_protection': ['foundational_rules_clean.pl', 'legal_aid_clean_v2.pl', 'consumer_protection.pl', 'cross_domain_rules.pl'],
            'employment_law': ['foundational_rules_clean.pl', 'legal_aid_clean_v2.pl', 'employment_law.pl', 'cross_domain_rules.pl'],
            'general_legal_aid': ['foundational_rules_clean.pl', 'legal_aid_clean_v2.pl']  # Use corrected versions
        }
        
        # Get domain-specific files and combine with base files
        domain_files = domain_mappings.get(domain, ['legal_aid.pl'])
        files = base_files + domain_files
        
        logger.info(f"📂 Selected files for domain '{domain}': {files}")
        return files

    def _is_relevant_domain(self, rule_category: str, domain: str) -> bool:
        """Check if a rule category is relevant to the specified domain for performance optimization."""
        
        # Define domain-to-category mappings for selective loading
        domain_mappings = {
            'legal_aid': ['legal_aid_rules', 'general_legal_rules', 'income_assessment', 'core_predicates'],
            'family_law': ['family_law_rules', 'legal_aid_rules', 'general_legal_rules', 'core_predicates'],
            'consumer_protection': ['consumer_protection_rules', 'legal_aid_rules', 'general_legal_rules', 'core_predicates'],
            'employment_law': ['employment_law_rules', 'legal_aid_rules', 'general_legal_rules', 'core_predicates'],
            'criminal_law': ['criminal_law_rules', 'legal_aid_rules', 'general_legal_rules', 'core_predicates'],
            'fundamental_rights': ['fundamental_rights_rules', 'legal_aid_rules', 'general_legal_rules', 'core_predicates'],
            'general_legal_aid': ['legal_aid_rules', 'general_legal_rules', 'income_assessment', 'core_predicates']
        }
        
        # Get relevant categories for this domain
        relevant_categories = domain_mappings.get(domain, [])
        
        # Always include core predicates and general rules
        if rule_category in ['core_predicates', 'general_legal_rules', 'legal_aid_rules']:
            return True
            
        # Check if this category is specifically relevant to the domain
        return rule_category in relevant_categories
    
    def _clean_prolog_rule(self, rule: str) -> str:
        """Clean and validate a Prolog rule for proper syntax."""
        if not rule or not rule.strip():
            return ""
        
        rule = rule.strip()
        
        # Skip comments and empty lines
        if rule.startswith('%') or not rule:
            return rule
        
        # Fix common syntax issues before processing
        rule = rule.replace('\\\\+', '\\+')
        rule = rule.replace('\\"', '"')  # Fix escaped quotes
        
        # 🔧 CRITICAL FIX: Split compound rules that got merged into one line
        # Look for patterns like: "rule1. comment rule2 :-" or "rule1.rule2 :-"
        if '. ' in rule and ':-' in rule:
            # Find the first complete rule ending with a period
            parts = rule.split('. ')
            if len(parts) > 1:
                first_rule = parts[0].strip() + '.'
                # Log that we're splitting and return only the first valid rule
                logger.debug(f"Split compound rule, using first part: {first_rule}")
                return first_rule
        
        # Handle rules that are concatenated without space: "rule1.rule2 :-"
        if '.' in rule and ':-' in rule and rule.count('.') > 1:
            dot_index = rule.find('.')
            if dot_index > 0:
                first_rule = rule[:dot_index + 1].strip()
                logger.debug(f"Extracted first rule from concatenated: {first_rule}")
                return first_rule
        
        # Remove extra whitespace and normalize
        rule = ' '.join(rule.split())
        
        # Ensure rule ends with period if it's a complete rule (not a comment)
        if rule and not rule.startswith('%'):
            if (':-' in rule or '(' in rule) and not rule.endswith('.'):
                rule += '.'
            # Handle facts (no :- operator)
            elif ':-' not in rule and '(' in rule and not rule.endswith('.'):
                rule += '.'
        
        # Validate basic Prolog syntax
        if rule and not rule.startswith('%'):
            # Check for unmatched parentheses
            if rule.count('(') != rule.count(')'):
                logger.warning(f"Unmatched parentheses in rule: {rule[:50]}...")
                return ""
            
            # Check for unmatched quotes
            if rule.count('"') % 2 != 0 and rule.count("'") % 2 != 0:
                logger.warning(f"Unmatched quotes in rule: {rule[:50]}...")
                return ""
        
        return rule

    def _split_compound_rules(self, rule_text: str) -> List[str]:
        """Split compound rules that are merged on one line into separate rules."""
        if not rule_text or not rule_text.strip():
            return []
        
        rule_text = rule_text.strip()
        
        # Skip comments
        if rule_text.startswith('%'):
            return [rule_text]
        
        # Look for patterns where multiple rules are on one line
        # Pattern 1: "rule1. % comment rule2 :-"
        # Pattern 2: "rule1. rule2 :-" 
        # Pattern 3: "rule1.rule2 :-"
        
        rules = []
        
        # First, try to split by ". " followed by a new rule (contains :- or starts with predicate)
        if '. ' in rule_text and rule_text.count('.') > 1:
            parts = rule_text.split('. ')
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                    
                # Add the period back except for the last part
                if i < len(parts) - 1:
                    if not part.endswith('.'):
                        part += '.'
                
                # Check if this looks like a new rule (has :- or is a fact)
                if (':-' in part or 
                    (part.endswith('.') and '(' in part and not part.startswith('%'))):
                    rules.append(part)
        else:
            # No splitting needed, return as single rule
            rules.append(rule_text)
        
        # Filter out empty rules and clean up
        cleaned_rules = []
        for rule in rules:
            rule = rule.strip()
            if rule and not rule.startswith('%'):
                # Ensure rule ends with period
                if not rule.endswith('.') and (':-' in rule or '(' in rule):
                    rule += '.'
                cleaned_rules.append(rule)
            elif rule.startswith('%'):
                cleaned_rules.append(rule)
        
        return cleaned_rules if cleaned_rules else [rule_text]

    def _parse_confidence_result(self, result: str) -> float:
        """Parse confidence score from Prolog result."""
        try:
            # Extract numeric value from result string
            import re
            confidence_match = re.search(r'Confidence\s*=\s*([0-9.]+)', result)
            if confidence_match:
                return float(confidence_match.group(1))
            else:
                return 0.75  # Default medium confidence
        except:
            return 0.75

    def _get_quick_reasoning(self, case_id: str, prolog_file: str) -> List[Dict[str, Any]]:
        """Get quick reasoning details with short timeout."""
        try:
            reasoning_query = f"primary_eligibility_reason('{case_id}', Reason)"
            # Short timeout for reasoning - this is optional
            result = self._execute_prolog_query_with_retry(reasoning_query, prolog_file, max_attempts=1, timeout=5)
            
            if result:
                return [{'type': 'primary_reason', 'content': result, 'confidence': 0.8}]
            else:
                return [{'type': 'basic_eligibility', 'content': 'Eligible based on legal aid criteria', 'confidence': 0.75}]
        except:
            return [{'type': 'fallback', 'content': 'Standard legal aid eligibility', 'confidence': 0.7}]

    
    def create_comprehensive_prolog_file(self, facts: List[str], rules: List[str] = None, 
                                       domains: Optional[List[str]] = None) -> str:
        """Create comprehensive Prolog file with domain-specific rules.
        If 'domains' are specified, only include rules relevant to those domains.
        Otherwise, include all loaded rules."""
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8')
        self.temp_files.append(temp_file.name)
        
        try:
            temp_file.write(f'% HybEx-Law Legal Reasoning Session: {self.session_id}\n')
            temp_file.write(f'% Generated: {datetime.now().isoformat()}\n\n')
            
            # Add discontiguous directives at the top to avoid warnings
            for directive in self._get_discontiguous_directives():
                temp_file.write(f"{directive}\n")
            temp_file.write('\n')

            # Write all rules first
            temp_file.write('% RULES\n')
            if rules:
                for rule in rules:
                    temp_file.write(rule.strip() + '\n\n')
            elif domains:
                for domain in domains:
                    domain_key = f'{domain.lower().replace(" ", "_")}_rules'
                    if domain_key in self.legal_rules:
                        temp_file.write(f'% {domain.upper()} DOMAIN RULES\n')
                        for rule in self.legal_rules[domain_key]:
                            temp_file.write(rule.strip() + '\n\n')
            else:
                for rule_category, rule_list in self.legal_rules.items():
                    temp_file.write(f'% {rule_category.upper()}\n')
                    for rule in rule_list:
                        temp_file.write(rule.strip() + '\n\n')

            # Write all facts at the end
            temp_file.write('\n% CASE FACTS\n')
            for fact in facts:
                if fact.strip():
                    if not fact.strip().endswith('.'):
                        temp_file.write(fact.strip() + '.\n')
                    else:
                        temp_file.write(fact.strip() + '\n')
            
            temp_file.close()
            logger.info(f"Created comprehensive Prolog file: {temp_file.name} with {len(facts)} facts.")
            
            return temp_file.name
            
        except Exception as e:
            temp_file.close()
            logger.error(f"Error creating comprehensive Prolog file: {e}", exc_info=True)
            return None

    def _execute_prolog_query(self, file_path: str, query: str, variables: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Execute a Prolog query using subprocess, changing the CWD to handle paths with spaces.
        This is the most robust method for Windows environments with spaces in user paths.
        """
        p = Path(file_path)
        prolog_dir = p.parent
        prolog_filename = p.name

        if variables:
            output_parts = [f"format('{v}:~q~n', [{v}])" for v in variables]
            full_query_text = f"({query}, {', '.join(output_parts)})"
        else:
            full_query_text = query

        # Use a list of arguments with shell=False for maximum safety and compatibility.
        # The filename is now relative because we set the current working directory (cwd).
        command_list = [
            'swipl',
            '-q',
            '-g', f"consult('{prolog_filename}')",
            '-g', full_query_text,
            '-g', 'halt'
        ]
        
        logger.info(f"Executing SWI-Prolog in '{prolog_dir}': {' '.join(command_list)}")
        
        try:
            result = subprocess.run(
                command_list,
                shell=False,  # shell=False is safer and recommended with a list of args
                capture_output=True,
                text=True,
                timeout=self.config.PROLOG_CONFIG.get('timeout', 120),
                encoding='utf-8',
                cwd=prolog_dir  # Set the working directory to the temp file's location
            )
            
            success = (result.returncode == 0)
            stderr = result.stderr.strip()
            
            if stderr:
                # Suppress non-fatal singleton variable warnings
                if 'Singleton variables' not in stderr:
                    logger.warning(f"Prolog warning/error: {stderr}")
                if 'ERROR:' in stderr.upper() or 'SYNTAX ERROR' in stderr.upper():
                    success = False
            
            results = []
            if success and variables and result.stdout:
                stdout = result.stdout.strip()
                try:
                    lines = stdout.split('\n')
                    res_dict = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            clean_value = value.strip().strip("'\"")
                            res_dict[key.strip()] = clean_value
                    if res_dict:
                        results = [res_dict]
                except Exception as e:
                    logger.warning(f"Failed to parse Prolog output: {stdout} | Error: {e}")
            
            return success, results
        
        except subprocess.TimeoutExpired:
            logger.error(f"Prolog query timed out after {self.config.PROLOG_CONFIG.get('timeout', 120)} seconds: {query}")
            return False, []
        except Exception as e:
            logger.error(f"Prolog execution failed: {e}", exc_info=True)
            return False, []

    def execute_prolog_query(self, prolog_file: str, query: PrologQuery, timeout_seconds: int = 60) -> Tuple[bool, List[Dict], str]:
        if not self.prolog_available:
            return self._advanced_fallback_reasoning(query.query_text)

        if not Path(prolog_file).exists():
            logger.error(f"Prolog file not found: {prolog_file}. Cannot execute query.")
            return False, [], "Prolog file not found."

        # Correctly format the path for Prolog, which may be on a different OS
        safe_prolog_path = Path(prolog_file).as_posix()
        for attempt in range(query.retry_count):
            try:
                # Use single quotes around the file path in the command to handle spaces and special characters
                # Add halt at the end to prevent SWI-Prolog from entering interactive mode
                prolog_query_cmd = f"['{safe_prolog_path}'], {query.query_text}, halt"
                # The -g flag requires a single goal, so we pass the whole thing as a string
                cmd = ['swipl', '-q', '-g', prolog_query_cmd, '--stack-limit=2g', '--traditional']
                
                logger.info(f"Executing SWI-Prolog command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False, encoding='utf-8')
                
                if result.returncode != 0:
                    logger.warning(f"SWI-Prolog exited with code {result.returncode} (attempt {attempt + 1}). Stderr: {result.stderr.strip()}")
                    if "Syntax error" in result.stderr or "ERROR" in result.stderr:
                        return False, [], f"Prolog syntax or fatal error: {result.stderr.strip()}"
                
                output = result.stdout.strip()
                
                # Enhanced success evaluation: consider exit code 0 as success even with empty output
                if result.returncode == 0:
                    # For boolean queries (like eligible_for_legal_aid), exit code 0 means TRUE
                    if not output.strip():
                        success = True  # Empty output with exit code 0 = success
                        output = "true"  # Set meaningful output
                    else:
                        success = self._evaluate_prolog_success(output)
                else:
                    success = False
                
                parsed_results = self._parse_prolog_output(output, query.expected_variables)
                
                if success:
                    logger.info(f"Prolog query successful (attempt {attempt + 1}): {query.query_text}")
                    return True, parsed_results, f"Prolog reasoning completed: {output}"
                else:
                    logger.warning(f"Prolog query returned false/no or empty (attempt {attempt + 1}): {query.query_text}. Output: {output}. Stderr: {result.stderr}")
                    if attempt == query.retry_count - 1:
                        return False, parsed_results, f"Prolog query failed after {query.retry_count} attempts: {result.stderr.strip() or output}"
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Prolog query timeout (attempt {attempt + 1}): {query.query_text}")
                if attempt == query.retry_count - 1:
                    return False, [], f"Query timeout after {query.retry_count} attempts"
            except Exception as e:
                logger.error(f"Error executing Prolog query (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt == query.retry_count - 1:
                    return self._advanced_fallback_reasoning(query.query_text)
        
        return self._advanced_fallback_reasoning(query.query_text)
    
    def _evaluate_prolog_success(self, output: str) -> bool:
        """Evaluate if Prolog query was successful based on output patterns."""
        output_lower = output.lower()
        
        if 'true.' in output_lower or 'yes.' in output_lower:
            return True
        
        if re.search(r'\\w+\\s*=\\s*\'?([^\',]+)\'?', output_lower):
            return True
        
        return False

    def _parse_prolog_output(self, output: str, expected_variables: List[str]) -> List[Dict]:
        """Parse Prolog output into structured results, expecting specific variables."""
        results = []
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            current_result = {}
            variable_bindings = re.findall(r'(\\w+)\\s*=\\s*\'?([^\',]+?)\'?[\\.,\\n]', line)
            
            if variable_bindings:
                for var, val in variable_bindings:
                    current_result[var.strip()] = val.strip().replace("'", "")
                results.append(current_result)
            elif line.lower() == 'true.':
                results.append({'result': True})
            elif line.lower() == 'false.':
                results.append({'result': False})

        if not results and output.strip():
            if self._evaluate_prolog_success(output):
                results.append({'result': True, 'raw_output': output})
            else:
                results.append({'raw_output': output})
        
        return results if results else []

    def _advanced_fallback_reasoning(self, query_text: str) -> Tuple[bool, List[Dict], str]:
        """Provides a simulated advanced reasoning if Prolog is unavailable."""
        logger.warning(f"SWI-Prolog not available. Performing advanced fallback reasoning for: {query_text}")
        
        # Simple rule-based parsing of the query to simulate eligibility
        query_lower = query_text.lower()
        
        eligible = False
        confidence = 0.5
        primary_reason = "Fallback: Unable to perform full symbolic reasoning."
        detailed_reasoning = []
        
        # Simulate entity extraction (basic)
        income_match = re.search(r'(\d+)\s*(lakh|lac|k|thousand)?', query_lower)
        income = int(income_match.group(1)) * 100000 if income_match and ('lakh' in income_match.group(2) or 'lac' in income_match.group(2)) else (int(income_match.group(1)) * 1000 if income_match and ('k' in income_match.group(2) or 'thousand' in income_match.group(2)) else (int(income_match.group(1)) if income_match else 0))
        
        category_match = re.search(r'(sc|st|scheduled caste|scheduled tribe|bpl|below poverty line)', query_lower)
        category = category_match.group(1) if category_match else 'general'

        # Use fallback thresholds (or hardcoded ones if no config is available)
        fallback_thresholds = self.config.ENTITY_CONFIG.get('income_thresholds', {'general': 300000, 'sc_st': 150000})
        
        threshold = fallback_thresholds.get(category.replace(" ", "_"), fallback_thresholds['general'])

        if 'sc' in category or 'st' in category or 'bpl' in category:
            eligible = True
            confidence = 0.9
            primary_reason = f"Fallback: Automatically eligible due to {category.upper()} status."
            detailed_reasoning.append({'type': 'category_rule', 'content': primary_reason, 'confidence': 0.9})
        elif income > 0 and income <= threshold:
            eligible = True
            confidence = 0.8
            primary_reason = f"Fallback: Eligible due to income (₹{income}) below threshold (₹{threshold})."
            detailed_reasoning.append({'type': 'income_rule', 'content': primary_reason, 'confidence': 0.8})
        else:
            eligible = False
            confidence = 0.4
            primary_reason = "Fallback: Not eligible based on basic income/category rules."
            detailed_reasoning.append({'type': 'no_rule_match', 'content': primary_reason, 'confidence': 0.4})

        return eligible, [{'Eligible': str(eligible), 'Confidence': str(confidence)}], primary_reason

    def comprehensive_legal_analysis(self, extracted_entities: Dict[str, Any], 
                                   domains: Optional[List[str]] = None) -> LegalReasoning:
        """Perform comprehensive legal analysis with domain-specific rule selection."""
        
        case_id = f"case_{self.session_id}_{datetime.now().strftime('%H%M%S')}"
        logger.info(f"🧠 Starting comprehensive legal analysis for {case_id}")
        
        # Detect the domain if not specified
        if not domains:
            detected_domain = self._detect_query_domain(extracted_entities)
            domains = [detected_domain]
        
        logger.info(f"🎯 Using domain-specific rules for: {domains}")
        
        facts = self._generate_comprehensive_facts(extracted_entities, case_id)
        
        # 🚀 USE MODULAR APPROACH: Create Prolog file with only domain-specific rules
        prolog_file = self.create_domain_specific_prolog_file_modular(facts, domains[0])
        
        # Fallback to robust approach if modular fails
        if not prolog_file:
            logger.warning("🔄 Modular approach failed, falling back to robust approach")
            prolog_file = self.create_domain_specific_prolog_file_robust(facts, domains[0])
        
        if not prolog_file:
            logger.error(f"Failed to create domain-specific Prolog file for {case_id}. Using fallback analysis.")
            eligible, parsed_results, reason = self._advanced_fallback_reasoning(extracted_entities.get('query', ''))
            
            # Construct the LegalReasoning object correctly using the fallback results
            return LegalReasoning(
                case_id=case_id,
                eligible=eligible,
                confidence=float(parsed_results[0].get('Confidence', 0.5)),
                primary_reason=reason,
                detailed_reasoning=[],
                applicable_rules=[],
                legal_citations=[],
                method='advanced_fallback'
            )
        
        try:
            eligibility_result = self._analyze_eligibility(prolog_file, case_id)
            reasoning_result = self._analyze_reasoning(prolog_file, case_id)
            rule_analysis = self._analyze_applicable_rules(prolog_file, case_id)
            
            # Dynamic confidence based on primary reason
            primary_reason = reasoning_result['primary_reason']
            if primary_reason:
                if 'categorical' in primary_reason.lower():
                    confidence = 0.95
                elif 'vulnerable' in primary_reason.lower():
                    confidence = 0.90
                elif 'income' in primary_reason.lower():
                    confidence = 0.85
                else:
                    confidence = 0.80  # Fallback for not eligible or unknown
            else:
                confidence = 0.80 if self.prolog_available else 0.50
            
            legal_reasoning = LegalReasoning(
                case_id=case_id,
                eligible=eligibility_result['eligible'],
                confidence=confidence,  # Use dynamic confidence instead of eligibility_result['confidence']
                primary_reason=reasoning_result['primary_reason'],
                detailed_reasoning=reasoning_result['detailed_reasoning'],
                applicable_rules=rule_analysis['applicable_rules'],
                legal_citations=rule_analysis['legal_citations'],
                method='prolog' if self.prolog_available else 'advanced_fallback'
            )
            
            logger.info(f"Comprehensive legal analysis complete for {case_id}: Eligible={legal_reasoning.eligible}, Confidence={legal_reasoning.confidence:.2f}")
            return legal_reasoning
            
        except Exception as e:
            logger.error(f"Error in comprehensive legal analysis for {case_id}: {e}", exc_info=True)
            eligible, parsed_results, reason = self._advanced_fallback_reasoning(extracted_entities.get('query', ''))
            return LegalReasoning(
                case_id=case_id,
                eligible=eligible,
                confidence=float(parsed_results[0].get('Confidence', 0.5)),
                primary_reason=reason,
                detailed_reasoning=[],
                applicable_rules=[],
                legal_citations=[],
                method='advanced_fallback_error'
            )
        
        finally:
            self._cleanup_temp_files()

    def _analyze_eligibility(self, prolog_file: str, case_id: str) -> Dict[str, Any]:
        """Analyze eligibility with ultra-fast optimized strategy."""
        
        # 🚀 STEP 1: Ultra-fast basic eligibility check (3s timeout)
        logger.info(f"🎯 Ultra-fast eligibility check for {case_id}")
        basic_query = PrologQuery(
            query_text=f"eligible_for_legal_aid('{case_id}')",
            expected_variables=[],
            timeout=3,
            retry_count=1
        )
        
        success, results, explanation = self.execute_prolog_query(prolog_file, basic_query, timeout_seconds=3)
        
        if success:
            logger.info(f"✅ FAST SUCCESS: Basic eligibility confirmed for {case_id}")
            return {
                'eligible': True,
                'confidence': 0.85,  # High confidence for fast success
                'explanation': f"Fast Prolog eligibility confirmed: {explanation}"
            }
        else:
            logger.info(f"❌ FAST RESULT: Not eligible - {explanation}")
            return {
                'eligible': False,
                'confidence': 0.9,   # High confidence for definitive no
                'explanation': f"Fast Prolog check failed: {explanation}"
            }
    
    def _analyze_reasoning(self, prolog_file: str, case_id: str) -> Dict[str, Any]:
        """Analyze detailed reasoning for the decision using Prolog."""
        
        # Get primary reason using new JSON method
        success_primary, results_primary = self._execute_prolog_query(
            prolog_file, 
            f"primary_eligibility_reason('{case_id}', Reason)",
            ['Reason']
        )
        
        primary_reason = "System analysis"
        if success_primary and results_primary:
            for res_dict in results_primary:
                if 'Reason' in res_dict:
                    primary_reason = res_dict['Reason'].strip("'\"")  # Strip any outer quotes
                    break
        
        # Get detailed reasoning using new JSON method
        success_detailed, results_detailed = self._execute_prolog_query(
            prolog_file,
            f"generate_detailed_reasoning('{case_id}', DetailedReason)",
            ['DetailedReason']
        )
        
        detailed_reasoning_list = []
        if success_detailed and results_detailed:
            for res_dict in results_detailed:
                if 'DetailedReason' in res_dict:
                    detailed_reasoning = res_dict['DetailedReason'].strip("'\"")  # Strip any outer quotes
                    detailed_reasoning_list.append({
                        'type': 'prolog_reasoning',
                        'content': detailed_reasoning,
                        'confidence': 0.95
                    })
                    break
        
        return {
            'primary_reason': primary_reason,
            'detailed_reasoning': detailed_reasoning_list
        }

    def _analyze_applicable_rules(self, prolog_file: str, case_id: str) -> Dict[str, Any]:
        """Analyze applicable rules and legal citations using Prolog."""
        
        # Use new JSON method for rules
        success, results = self._execute_prolog_query(
            prolog_file,
            f"findall(Rule, applicable_rule('{case_id}', Rule, legal_aid), Rules)",
            ['Rules']
        )
        
        applicable_rules = []
        if success and results:
            for res_dict in results:
                if 'Rules' in res_dict:
                    rules_str = res_dict['Rules'].strip('[]')  # Strip list brackets
                    applicable_rules = [r.strip().strip("'\"") for r in rules_str.split(',') if r.strip()] if rules_str else []
                    break

        legal_citations = [
            "Legal Services Authorities Act, 1987 - Section 12",
            "Legal Services Authorities Act, 1987 - Chapter III"
        ]
        
        return {
            'applicable_rules': applicable_rules,
            'legal_citations': legal_citations
        }

    # In hybex_system/prolog_engine.py
# REPLACE the entire _generate_comprehensive_facts method with this final version.

    def _generate_comprehensive_facts(self, entities: Dict[str, Any], case_id: str) -> List[str]:
        """
        Generate comprehensive Prolog facts from extracted and pre-normalized entities.
        This version trusts that the DataProcessor has already cleaned the data.
        """
        facts = [f"person('{case_id}')."]
        
        # General & Demographic facts
        if 'income' in entities:
            facts.append(f"annual_income('{case_id}', {entities['income']}).")
        if 'social_category' in entities:
            # No normalization needed here, as data_processor now provides the correct short-form
            facts.append(f"social_category('{case_id}', '{entities['social_category']}').")
        if 'age' in entities:
            facts.append(f"age('{case_id}', {entities['age']}).")
        if 'gender' in entities:
            facts.append(f"gender('{case_id}', '{entities['gender']}').")
        
        # Employment Law Facts
        if 'employment_duration' in entities:
            facts.append(f"employment_duration('{case_id}', {entities['employment_duration']}).")
        if 'daily_wage' in entities:
            facts.append(f"daily_wage('{case_id}', {entities['daily_wage']}).")
            # Derive hourly wage for overtime calculations
            facts.append(f"hourly_wage('{case_id}', {int(entities['daily_wage'] / 8)}).")
        if 'notice_period_given' in entities:
            facts.append(f"notice_period_given('{case_id}', {entities['notice_period_given']}).")
        if entities.get('disciplinary_hearing_conducted', True): # Default to True unless explicitly False
            facts.append(f"disciplinary_hearing_conducted('{case_id}').")

        # Family Law Facts
        if entities.get('is_married'):
            facts.append(f"married('{case_id}', 'spouse_of_{case_id}').")
        if entities.get('has_children', 0) > 0:
             facts.append(f"parent('{case_id}', 'child_of_{case_id}').")

        # Consumer Protection Facts
        if 'goods_value' in entities:
            facts.append(f"goods_value('{case_id}', {entities['goods_value']}).")
            facts.append(f"transaction_amount('{case_id}', {entities['goods_value']}).")
        
        if 'incident_date' in entities:
            try:
                y, m, d = map(int, entities['incident_date'].split('-'))
                facts.append(f"incident_date('{case_id}', general_complaint, date({y}, {m}, {d})).")
                today = datetime.now()
                facts.append(f"complaint_date('{case_id}', general_complaint, date({today.year}, {today.month}, {today.day})).")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse incident_date: {entities['incident_date']}")

        # Add income threshold facts (essential for legal_aid_clean_v2.pl)
        facts.extend(self._generate_income_threshold_facts())
        facts.extend(self._generate_derived_facts(entities, case_id))
        
        logger.info(f"Generated {len(facts)} comprehensive Prolog facts for {case_id}")
        logger.debug(f"Facts generated: {facts}")
        return facts

    def _generate_income_threshold_facts(self) -> List[str]:
        """Generate income threshold facts needed by legal_aid_clean_v2.pl"""
        threshold_facts = [
            "income_threshold('sc', 800000).",
            "income_threshold('st', 800000).", 
            "income_threshold('obc', 600000).",
            "income_threshold('bpl', 0).",
            "income_threshold('ews', 800000).",
            "income_threshold('general', 500000)."
        ]
        return threshold_facts
    def _get_discontiguous_directives(self) -> List[str]:
        """Generate discontiguous directives for all known multi-clause predicates."""
        predicates = [
            'person/1', 'eligible_for_legal_aid/1', 'income_eligible/1', 'categorically_eligible/1',
            'eligible_with_confidence/3', 'primary_eligibility_reason/2', 'generate_detailed_reasoning/2',
            'legal_aid_applicable/2', 'covered_case_type/1', 'excluded_case_type/1',
            'valid_marriage/3', 'marriage_conditions_met/3', 'age_requirement_met/2',
            'divorce_grounds_exist/3', 'valid_divorce_ground/2', 'maintenance_eligible/1',
            'child_custody_preference/2', 'consumer_forum_jurisdiction/2',
            'valid_consumer_complaint/2', 'consumer_issue/1', 'complaint_within_time_limit/2',
            'consumer_compensation/3', 'defective_goods_compensation/2',
            'fundamental_right_violated/2', 'constitutional_right/1',
            'right_violation_occurred/2', 'prohibited_discrimination_ground/1',
            'rti_applicable/2', 'exempt_information/1', 'pil_standing/2',
            'public_interest_issue/1', 'wrongful_termination/1', 'improper_procedure/1',
            'sufficient_notice_period/2', 'retrenchment_compensation/2',
            'valid_harassment_complaint/1', 'harassment_remedy_available/2',
            'harassment_remedy/1', 'minimum_wage_violation/1', 'minimum_wage_rate/2',
            'overtime_payment_due/2', 'overtime_hours/2',
            'legal_aid_employment_case/1', 'legal_aid_family_case/1', 'legal_aid_consumer_case/1',
            'constitutional_employment_remedy/1', 'employment_pil_standing/2',
            'employment_related_issue/1', 'days_between/3', 'appeal_court/2',
            'legal_costs_estimate/3', 'base_court_fee/2', 'lawyer_fee_estimate/2',
            'required_documents/2', 'case_timeline_estimate/2',
            'income_threshold/2', 'income_within_threshold/2', 'income_ratio/3',
            'applicable_rule/3', 'rule_priority/3', 'best_applicable_rule/3',
            'resolve_conflict/3', 'confidence_score/2',
            'legal_aid_factor/2', 'individual_reason/2',
            'annual_income/2', 'monthly_income/2', 'applicant_social_category/2',
            'social_category/2', 'case_type/2', 'applicant_location/2',
            'vulnerable_group/2'
        ]
        return [f":- discontiguous {p}." for p in predicates]
    def _generate_derived_facts(self, entities: Dict[str, Any], case_id: str) -> List[str]:
        """Generate additional derived facts based on extracted entities."""
        derived_facts = []
        
        # Example: Assume anyone below a very low threshold is "vulnerable"
        # This is a simplification; actual vulnerability rules would be complex
        # You can add more sophisticated derived facts here.
        if 'income' in entities and entities['income'] is not None:
            general_threshold = self.config.ENTITY_CONFIG['income_thresholds'].get('general', 300000)
            if entities['income'] < general_threshold / 2: # Very low income implies vulnerability
                derived_facts.append(f"vulnerable_group('{case_id}', 'low_income').")

        if 'social_category' in entities:
            category = entities['social_category'].lower()
            if 'women' in category: # If your entity extractor identifies 'women' as a social category
                derived_facts.append(f"vulnerable_group('{case_id}', 'women').")
            if 'child' in category or 'minor' in category:
                derived_facts.append(f"vulnerable_group('{case_id}', 'children').")
            if 'disabled' in category:
                derived_facts.append(f"vulnerable_group('{case_id}', 'disabled').")

        return derived_facts

    def _cleanup_temp_files(self, file_paths: List[str] = None):
        """Clean up temporary Prolog files idempotently."""
        files_to_clean = file_paths if file_paths else self.temp_files
        
        # Create a copy to avoid modification issues during iteration
        for temp_file_path in list(files_to_clean):
            try:
                os.remove(temp_file_path)
                # We can keep the info log, or remove it for a quieter run
                # logger.info(f"Cleaned up temp file: {temp_file_path}")
            except FileNotFoundError:
                # This is expected if the file was already cleaned up by a 'finally' block.
                # We can safely ignore this error.
                pass 
            except OSError as e:
                # Log other, unexpected OS errors.
                logger.warning(f"Error cleaning up temp file {temp_file_path}: {e}")

        # After attempting cleanup, remove the files from the master list
        if file_paths:
            # If a specific list was provided, remove only those items from the master list
            self.temp_files = [f for f in self.temp_files if f not in file_paths]
        else:
            # If no list was provided, we were cleaning the whole master list, so clear it.
            self.temp_files.clear()

    def get_comprehensive_rule_summary(self) -> Dict[str, Any]:
        """Returns a summary of the loaded comprehensive rules."""
        summary = {
            'rules_loaded_from_kb': self.rules_loaded,
            'total_rules': sum(len(rules) for rules in self.legal_rules.values()),
            'rule_counts': {k: len(v) for k, v in self.legal_rules.items()},
            'prolog_engine_available': self.prolog_available
        }
        return summary

    def batch_legal_analysis(self, cases: List[Dict[str, Any]]) -> List[LegalReasoning]:
        """Perform batch legal analysis for multiple cases."""
        logger.info(f"Batch analyzing {len(cases)} cases with Prolog engine.")
        
        results = []
        all_facts_combined = []
        case_ids_in_batch = []

        # Generate all facts first and map case_id to original sample_id
        for i, case_entities in enumerate(cases):
            # Use original sample_id if available, otherwise generate a new one
            # This 'sample_id' should map to the 'case_id' used in Prolog facts
            # Assuming 'sample_id' is already in `case_entities` if coming from `DataProcessor`
            case_id = case_entities.get('sample_id', f"batch_case_{self.session_id}_{i}")
            case_ids_in_batch.append(case_id)
            
            # Generate facts for each case using the *original* entities
            # _generate_comprehensive_facts expects extracted_entities and case_id
            current_case_facts = self._generate_comprehensive_facts(case_entities.get('extracted_entities', {}), case_id)
            all_facts_combined.extend(current_case_facts)
            
        # Create a single Prolog file with all rules and all facts for efficiency
        prolog_file = self.create_comprehensive_prolog_file(all_facts_combined)

        if not prolog_file:
            logger.error("Failed to create combined Prolog file for batch processing. Falling back for all cases.")
            for case_entities in cases:
                eligible, parsed_res, reason = self._advanced_fallback_reasoning(case_entities.get('query', ''))
                results.append(LegalReasoning(
                    case_id=case_entities.get('sample_id', 'N/A'),
                    eligible=eligible,
                    confidence=float(parsed_res[0].get('Confidence', 0.5)),
                    primary_reason=reason,
                    detailed_reasoning=[],
                    applicable_rules=[],
                    legal_citations=[],
                    method='advanced_fallback'
                ))
            return results

        try:
            for case_entities in cases:
                case_id = case_entities.get('sample_id', 'N/A') # Re-use or retrieve the case_id
                
                # Query each case_id for its final eligibility and reasoning directly from the combined file.
                eligibility_query = PrologQuery(
                    query_text=f"eligible_with_confidence('{case_id}', Eligible, Confidence)",
                    expected_variables=['Eligible', 'Confidence']
                )
                eligible, eligibility_results, _ = self.execute_prolog_query(prolog_file, eligibility_query)

                reasoning_query = PrologQuery(
                    query_text=f"generate_detailed_reasoning('{case_id}', DetailedReason)",
                    expected_variables=['DetailedReason']
                )
                success_detailed, results_detailed, _ = self.execute_prolog_query(prolog_file, reasoning_query)
                
                primary_reason_query = PrologQuery(
                    query_text=f"primary_eligibility_reason('{case_id}', Reason)",
                    expected_variables=['Reason']
                )
                success_primary, results_primary, _ = self.execute_prolog_query(prolog_file, primary_reason_query)

                primary_reason = "Determined by Prolog"
                detailed_reasoning = []
                if success_detailed and results_detailed:
                    for res_dict in results_detailed:
                        if 'DetailedReason' in res_dict:
                            primary_reason = res_dict['DetailedReason'].replace("'", "")
                            detailed_reasoning.append({
                                'type': 'prolog_reasoning',
                                'content': primary_reason,
                                'confidence': 0.95
                            })
                            break
                elif success_primary and results_primary: # Fallback to primary reason if detailed not found
                     for res_dict in results_primary:
                        if 'Reason' in res_dict:
                            primary_reason = res_dict['Reason'].replace("'", "")
                            detailed_reasoning.append({
                                'type': 'prolog_reasoning',
                                'content': primary_reason,
                                'confidence': 0.9 # Slightly lower confidence for just primary
                            })
                            break

                # For batch, we're not re-extracting rules/citations, assuming they apply generally
                # Or, you'd need to modify _analyze_applicable_rules to work on the combined file
                # A simpler way for batch is to provide a generic placeholder or pre-compute
                
                # Extract eligible and confidence from eligibility_results
                current_eligible = False
                current_confidence = 0.5
                if eligible and eligibility_results:
                    for res_dict in eligibility_results:
                        if 'Eligible' in res_dict:
                            current_eligible = (res_dict['Eligible'].lower() == 'true')
                        if 'Confidence' in res_dict:
                            try:
                                current_confidence = float(res_dict['Confidence'])
                            except ValueError:
                                pass # Keep default or previously parsed

                results.append(LegalReasoning(
                    case_id=case_id,
                    eligible=current_eligible,
                    confidence=current_confidence,
                    primary_reason=primary_reason,
                    detailed_reasoning=detailed_reasoning,
                    applicable_rules=["Rules applied from comprehensive knowledge base"], # Placeholder for batch
                    legal_citations=["Legal Services Authorities Act, 1987"], # Placeholder for batch
                    method='prolog' if self.prolog_available else 'advanced_fallback'
                ))
        
        finally:
            self._cleanup_temp_files()

        logger.info(f"Batch analysis complete: {len(results)} results")
        return results

    def save_reasoning_results(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """Save reasoning results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prolog_reasoning_results_{self.session_id}_{timestamp}.json"
        
        results_path = self.config.RESULTS_DIR / "prolog_reasoning" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        eligible_count = sum(1 for r in results if r.get('eligible', False))
        total_count = len(results)
        
        output_data = {
            'summary': {
                'total_cases': total_count,
                'eligible_cases': eligible_count,
                'eligibility_rate': eligible_count / total_count if total_count > 0 else 0,
                'average_confidence': sum(r.get('confidence', 0) for r in results) / len(results) if results else 0,
                'prolog_engine_used': self.prolog_available,
                'generation_timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            },
            'results': results
        }
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved reasoning results to {results_path}")
            logger.info(f"  - Total cases: {total_count}")
            logger.info(f"  - Eligible cases: {eligible_count}")
            logger.info(f"  - Eligibility rate: {eligible_count/total_count*100:.1f}%")
            
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Failed to save reasoning results: {e}", exc_info=True)
            raise

    def get_comprehensive_rule_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of loaded rules"""
        if not self.legal_rules:
            return {
                'total_rules': 0,
                'rule_counts': {},
                'status': 'no_rules_loaded'
            }
        
        rule_counts = {}
        total_rules = 0
        
        for category, rules in self.legal_rules.items():
            if rules:
                count = len(rules)
                rule_counts[category] = count
                total_rules += count
        
        return {
            'total_rules': total_rules,
            'rule_counts': rule_counts,
            'rules_loaded_from_kb': self.rules_loaded,
            'prolog_available': self.prolog_available,
            'status': 'loaded' if self.rules_loaded else 'fallback'
        }

    def cleanup(self):
        """Clean up Prolog engine resources"""
        logger.info(f"Cleaning up PrologEngine session {self.session_id}")
        self._cleanup_temp_files()
        logger.info("PrologEngine cleanup completed")

    def __del__(self):
        self._cleanup_temp_files()

    def determine_eligibility_for_generation(self, facts: List[str]) -> Tuple[bool, str]:
        """
        A streamlined eligibility check for dataset generation.
        """
        if not self.prolog_available:
            # Simple fallback for generation if Prolog isn't there.
            income_fact = next((f for f in facts if 'annual_income' in f), None)
            if income_fact:
                try:
                    income = int(re.search(r'\d+', income_fact).group())
                    return income <= 500000, "Fallback eligibility based on income <= 500000"
                except (ValueError, AttributeError):
                    return False, "Fallback eligibility failed due to income parse error"
            return False, "Fallback eligibility: no income data"

        # Use the simpler, single-file creation method
        prolog_file = self.create_prolog_file_for_generation(facts)
        if not prolog_file:
            return False, "Failed to create the Prolog query file."

        try:
            query = "eligible_for_legal_aid(case_gen)."
            # Use the robust _execute_prolog_query method
            success, _ = self._execute_prolog_query(prolog_file, query, [])
            
            eligible = success
            reason = "Eligible based on Prolog rules." if eligible else "Not eligible based on Prolog rules."
            return eligible, reason

        except Exception as e:
            logger.error(f"Prolog query for generation failed: {e}", exc_info=True)
            return False, "Prolog query execution error"
            
        finally:
            # Ensure the single temp file is cleaned up
            self._cleanup_temp_files([prolog_file])

    def create_prolog_file_for_generation(self, facts: List[str]) -> Optional[str]:
        """Creates a temporary Prolog file including the master KB and case-specific facts."""
        try:
            # Get the absolute, sanitized path to your master knowledge base
            kb_path = self.config.BASE_DIR / 'knowledge_base' / 'knowledge_base.pl'
            if not kb_path.exists():
                logger.error(f"Master knowledge base not found at: {kb_path}")
                return None
            # Ensure the path is in a format Prolog understands (forward slashes)
            kb_path_str = str(kb_path.resolve()).replace('\\', '/')

            # This tells Prolog: "First, load all our rules from the main knowledge base..."
            prolog_content = f":- include('{kb_path_str}').\n\n"
            # "...Then, read all the specific facts for this case."
            prolog_content += "% Case-specific facts for generation\n"
            prolog_content += "\n".join(facts)

            # Now, write this combined prolog_content to your temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8')
            self.temp_files.append(temp_file.name) # Track for cleanup
            
            temp_file.write(prolog_content)
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to create prolog file for generation: {e}", exc_info=True)
            return None