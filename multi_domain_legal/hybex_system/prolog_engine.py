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

@dataclass
class LegalReasoning:
    """Structure for legal reasoning results"""
    case_id: str
    eligible: bool
    confidence: float
    primary_reason: str
    detailed_reasoning: List[Dict[str, Any]]
    applicable_rules: List[str]
    legal_citations: List[str]
    method: str  # 'prolog', 'fallback', 'hybrid'

@dataclass
class PrologQuery:
    """Structure for Prolog queries"""
    query_text: str
    expected_variables: List[str]
    timeout: int = 30
    retry_count: int = 3

class PrologEngine:
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
        # Ensure income_thresholds are taken from the config, which should be updated by scraper
        self.income_thresholds = self.config.ENTITY_CONFIG['income_thresholds']
        
        logger.info(f"Initialized Comprehensive PrologEngine (Session: {self.session_id}, Available: {self.prolog_available})")
        logger.info(f"Loaded {len(self.legal_rules)} rule categories with {sum(len(rules) for rules in self.legal_rules.values())} total rules")
        if any(self.legal_rules.get(k) for k in ['legal_aid_rules', 'family_law_rules']):
            self.rules_loaded = True
        logger.info(f"Comprehensive rules actually loaded: {self.rules_loaded}")

    def setup_logging(self):
        """Setup comprehensive Prolog engine logging"""
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'prolog_reasoning_{self.session_id}.log') for h in logger.handlers):
            log_file = self.config.get_log_path(f'prolog_reasoning_{self.session_id}')
            file_handler = logging.FileHandler(log_file)
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
                    'legal_aid_rules': r'% LEGAL AID DOMAIN RULES(.*?)(?:% FAMILY LAW DOMAIN RULES|% CONSUMER PROTECTION DOMAIN RULES|% EMPLOYMENT LAW DOMAIN RULES|% FUNDAMENTAL RIGHTS DOMAIN RULES|\Z)',
                    'family_law_rules': r'% FAMILY LAW DOMAIN RULES(.*?)(?:% CONSUMER PROTECTION DOMAIN RULES|% EMPLOYMENT LAW DOMAIN RULES|% FUNDAMENTAL RIGHTS DOMAIN RULES|\Z)',
                    'consumer_protection_rules': r'% CONSUMER PROTECTION DOMAIN RULES(.*?)(?:% EMPLOYMENT LAW DOMAIN RULES|% FUNDAMENTAL RIGHTS DOMAIN RULES|\Z)',
                    'employment_law_rules': r'% EMPLOYMENT LAW DOMAIN RULES(.*?)(?:% FUNDAMENTAL RIGHTS DOMAIN RULES|\Z)',
                    'fundamental_rights_rules': r'% FUNDAMENTAL RIGHTS DOMAIN RULES(.*?)(?:\Z)',
                }

                for key, pattern in section_regex_map.items():
                    match = re.search(pattern, comprehensive_rules_str, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content:
                            # Parse the section content into individual rules
                            individual_rules = self._parse_prolog_rules_from_section(content)
                            rule_sections[key] = individual_rules
                            logger.debug(f"Extracted section: {key} with {len(individual_rules)} rules")
                
                rule_sections['eligibility_rules'] = self._extract_eligibility_rules(comprehensive_rules_str)
                rule_sections['reasoning_rules'] = self._extract_reasoning_rules(comprehensive_rules_str)
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
    
    def _load_multi_domain_rules(self) -> Optional[str]:
        """Load multi-domain rules from knowledge base (multi_domain_rules.py)."""
        try:
            rules_file_path = Path(__file__).parent.parent / "knowledge_base" / "multi_domain_rules.py"
            
            if rules_file_path.exists():
                logger.info(f"Attempting to load rules from: {rules_file_path}")
                spec = importlib.util.spec_from_file_location("multi_domain_rules", rules_file_path)
                rules_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rules_module)
                
                if hasattr(rules_module, 'MULTI_DOMAIN_LEGAL_RULES'):
                    logger.info("Successfully loaded MULTI_DOMAIN_LEGAL_RULES from knowledge base.")
                    return rules_module.MULTI_DOMAIN_LEGAL_RULES
                else:
                    logger.warning(f"File {rules_file_path} found, but 'MULTI_DOMAIN_LEGAL_RULES' variable not found within it.")
                    
        except Exception as e:
            logger.error(f"Error loading multi-domain rules from {rules_file_path}: {e}", exc_info=True)
        
        return None

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
            
            if rules:
                temp_file.write('% CUSTOM RULES\n')
                for rule in rules:
                    temp_file.write(rule.strip() + '\n\n')
            elif domains:
                for domain in domains:
                    domain_key = f'{domain.lower().replace(" ", "_")}_rules'
                    if domain_key in self.legal_rules:
                        temp_file.write(f'% {domain.upper()} DOMAIN RULES\n')
                        for rule in self.legal_rules[domain_key]:
                            temp_file.write(rule.strip() + '\n\n')
                        logger.debug(f"Added rules for domain: {domain_key}")
                    else:
                        logger.warning(f"No rules found for specified domain: {domain}")
            else:
                for rule_category, rule_list in self.legal_rules.items():
                    temp_file.write(f'% {rule_category.upper()}\n')
                    for rule in rule_list:
                        temp_file.write(rule.strip() + '\n\n')
            
            temp_file.write('% CASE FACTS\n')
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

    def execute_prolog_query(self, prolog_file: str, query: PrologQuery) -> Tuple[bool, List[Dict], str]:
        """Execute Prolog query with enhanced error handling and retries."""
        if not self.prolog_available:
            return self._advanced_fallback_reasoning(query.query_text)
        
        if not Path(prolog_file).exists():
            logger.error(f"Prolog file not found: {prolog_file}. Cannot execute query.")
            return False, [], "Prolog file not found."

        for attempt in range(query.retry_count):
            try:
                prolog_query_cmd = f"['{prolog_file}'], {query.query_text}."
                
                cmd = ['swipl', '-q', '-g', prolog_query_cmd, '--stack-limit=2g', '--traditional']
                
                logger.debug(f"Executing SWI-Prolog command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=query.timeout, check=False)
                
                if result.returncode != 0:
                    logger.warning(f"SWI-Prolog exited with code {result.returncode} (attempt {attempt + 1}). Stderr: {result.stderr.strip()}")
                    if "Syntax error" in result.stderr or "ERROR" in result.stderr:
                        return False, [], f"Prolog syntax or fatal error: {result.stderr.strip()}"
                
                output = result.stdout.strip()
                success = self._evaluate_prolog_success(output)
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
        """Perform comprehensive legal analysis with detailed reasoning."""
        
        case_id = f"case_{self.session_id}_{datetime.now().strftime('%H%M%S')}"
        logger.info(f"🧠 Starting comprehensive legal analysis for {case_id}")
        
        facts = self._generate_comprehensive_facts(extracted_entities, case_id)
        
        prolog_file = self.create_comprehensive_prolog_file(facts, domains=domains)
        
        if not prolog_file:
            logger.error(f"Failed to create Prolog file for {case_id}. Using fallback analysis.")
            eligible, parsed_results, reason = self._advanced_fallback_reasoning(extracted_entities.get('query', ''))
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
            
            legal_reasoning = LegalReasoning(
                case_id=case_id,
                eligible=eligibility_result['eligible'],
                confidence=eligibility_result['confidence'],
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
        """Analyze eligibility with confidence scoring using Prolog."""
        eligibility_query = PrologQuery(
            query_text=f"eligible_with_confidence('{case_id}', Eligible, Confidence)",
            expected_variables=['Eligible', 'Confidence']
        )
        
        success, results, explanation = self.execute_prolog_query(prolog_file, eligibility_query)
        
        if success and results:
            for res_dict in results:
                if 'Eligible' in res_dict and 'Confidence' in res_dict:
                    eligible_str = res_dict['Eligible']
                    confidence_str = res_dict['Confidence']
                    
                    try:
                        eligible_bool = (eligible_str.lower() == 'true')
                        confidence_val = float(confidence_str)
                        return {
                            'eligible': eligible_bool,
                            'confidence': confidence_val,
                            'explanation': explanation
                        }
                    except ValueError:
                        logger.warning(f"Could not parse Eligible/Confidence from Prolog output: {res_dict}. Falling back.")

        logger.warning(f"Failed to get confident eligibility from Prolog. Trying basic 'eligible_for_legal_aid'.")
        basic_eligibility_query = PrologQuery(
            query_text=f"eligible_for_legal_aid('{case_id}')",
            expected_variables=[]
        )
        basic_success, _, basic_explanation = self.execute_prolog_query(prolog_file, basic_eligibility_query)
        
        return {
            'eligible': basic_success,
            'confidence': 0.7 if basic_success else 0.3,
            'explanation': basic_explanation if basic_success else "Basic eligibility check failed."
        }
    
    def _analyze_reasoning(self, prolog_file: str, case_id: str) -> Dict[str, Any]:
        """Analyze detailed reasoning for the decision using Prolog."""
        primary_reason_query = PrologQuery(
            query_text=f"primary_eligibility_reason('{case_id}', Reason)",
            expected_variables=['Reason']
        )
        
        success_primary, results_primary, _ = self.execute_prolog_query(prolog_file, primary_reason_query)
        primary_reason = "System analysis"
        if success_primary and results_primary:
            for res_dict in results_primary:
                if 'Reason' in res_dict:
                    primary_reason = res_dict['Reason'].replace("'", "")
                    break
        
        detailed_reasoning_query = PrologQuery(
            query_text=f"generate_detailed_reasoning('{case_id}', DetailedReason)",
            expected_variables=['DetailedReason']
        )
        
        success_detailed, results_detailed, _ = self.execute_prolog_query(prolog_file, detailed_reasoning_query)
        detailed_reasoning_list = []
        if success_detailed and results_detailed:
            for res_dict in results_detailed:
                if 'DetailedReason' in res_dict:
                    detailed_reasoning_list.append({
                        'type': 'prolog_reasoning',
                        'content': res_dict['DetailedReason'].replace("'", ""),
                        'confidence': 0.95
                    })
        
        return {
            'primary_reason': primary_reason,
            'detailed_reasoning': detailed_reasoning_list
        }

    def _analyze_applicable_rules(self, prolog_file: str, case_id: str) -> Dict[str, Any]:
        """Analyze applicable rules and legal citations using Prolog."""
        rules_query = PrologQuery(
            query_text=f"findall(Rule, applicable_rule('{case_id}', Rule, legal_aid), Rules)",
            expected_variables=['Rules']
        )
        
        success, results, _ = self.execute_prolog_query(prolog_file, rules_query)
        applicable_rules = []
        if success and results:
            for res_dict in results:
                if 'Rules' in res_dict:
                    raw_rules = res_dict['Rules'].strip("[]")
                    if raw_rules:
                        applicable_rules = [r.strip().replace("'", "") for r in raw_rules.split(',')]
                        applicable_rules = [r for r in applicable_rules if r]

        legal_citations = [
            "Legal Services Authorities Act, 1987 - Section 12",
            "Legal Services Authorities Act, 1987 - Chapter III"
        ]
        
        return {
            'applicable_rules': applicable_rules,
            'legal_citations': legal_citations
        }

    def _generate_comprehensive_facts(self, entities: Dict[str, Any], case_id: str) -> List[str]:
        """Generate comprehensive Prolog facts from extracted entities and config."""
        facts = [f"person('{case_id}')"]
        
        if 'income' in entities and entities['income'] is not None:
            income_value = entities['income']
            facts.append(f"annual_income('{case_id}', {income_value})")
            if income_value is not None and isinstance(income_value, (int, float)) and income_value > 0:
                facts.append(f"monthly_income('{case_id}', {int(income_value / 12)})")
        
        if 'social_category' in entities and entities['social_category']:
            category = entities['social_category'].replace(" ", "_").lower()
            facts.append(f"applicant_social_category('{case_id}', '{category}')")
            facts.append(f"social_category('{case_id}', {category})")
        
        if 'case_type' in entities and entities['case_type']:
            case_type = entities['case_type'].replace(" ", "_").lower()
            facts.append(f"case_type('{case_id}', '{case_type}')")
        
        if 'location' in entities and entities['location']:
            location = entities['location'].replace(" ", "_").lower()
            facts.append(f"applicant_location('{case_id}', '{location}')")
        
        facts.extend(self._generate_derived_facts(entities, case_id))
        
        logger.info(f"Generated {len(facts)} comprehensive Prolog facts for {case_id}")
        return facts

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

    def _cleanup_temp_files(self):
        """Clean up temporary Prolog files."""
        for f_path in self.temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    logger.debug(f"Cleaned up temporary Prolog file: {f_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {f_path}: {e}")
        self.temp_files = []

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

    def cleanup(self):
        """Clean up Prolog engine resources"""
        logger.info(f"Cleaning up PrologEngine session {self.session_id}")
        self._cleanup_temp_files()
        logger.info("PrologEngine cleanup completed")

    def __del__(self):
        self._cleanup_temp_files()