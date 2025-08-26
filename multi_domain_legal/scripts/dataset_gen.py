"""
Comprehensive and Logically Consistent Legal Data Generation for HybEx-Law System.

This script generates high-quality, balanced, and varied training data for the
hybrid neural-symbolic legal AI system. It is a complete rewrite focusing on
logical consistency by using the Prolog engine to establish ground truth.

Core Principles:
1.  **Prolog-Driven Ground Truth**: It first generates structured facts, then uses
    the `PrologEngine` to determine the correct `expected_eligibility`.
2.  **Fact-to-Text Generation**: The natural language `query` is generated *from*
    the facts, ensuring the text, facts, and label are always aligned.
3.  **Balanced & Diverse**: Aims for a 50/50 eligible/ineligible split per domain
    and uses a wide variety of templates, including noise and edge cases.
4.  **Stratified Splits**: Automatically generates train (70%), val (15%), and
    test (15%) JSON files.
"""

import json
import random
import logging
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
from sklearn.model_selection import train_test_split
from faker import Faker

# --- Path Setup ---
# Add the project root to the Python path to allow importing from hybex_system
# Assumes the script is run from `multi_domain_legal/` or the repo root.
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from hybex_system.prolog_engine import PrologEngine
    from hybex_system.config import HybExConfig
except ImportError as e:
    print(f"Error: Could not import from hybex_system. Make sure you are running this script from the 'multi_domain_legal' directory or the project root.")
    print(f"System Path: {sys.path}")
    print(f"Project Root: {project_root}")
    raise e


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Data Structures ---
class LegalDomain(Enum):
    """Legal domains covered by the system."""
    LEGAL_AID = "legal_aid"
    FAMILY_LAW = "family_law"
    CONSUMER_PROTECTION = "consumer_protection"
    EMPLOYMENT_LAW = "employment_law"
    FUNDAMENTAL_RIGHTS = "fundamental_rights"

@dataclass
class LegalTrainingExample:
    """Structured training example for the legal AI system."""
    query: str
    domains: List[str]
    extracted_facts: List[str]  # Human-readable facts
    prolog_facts: List[str]     # Prolog-formatted facts
    expected_eligibility: bool
    legal_reasoning: str
    user_demographics: Dict[str, Any]
    case_complexity: str

class ComprehensiveLegalDataGenerator:
    """
    Generates diverse, balanced, and logically consistent legal training data
    by integrating directly with the PrologEngine for ground truth.
    """
    def __init__(self):
        logger.info("Initializing ComprehensiveLegalDataGenerator...")
        self.config = HybExConfig()
        try:
            self.prolog_engine = PrologEngine(self.config)
            self.prolog_engine._load_rules()
            if not self.prolog_engine.rules_loaded:
                raise RuntimeError("Prolog knowledge base could not be loaded. Aborting generation.")
            logger.info("PrologEngine initialized and knowledge base loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PrologEngine: {e}", exc_info=True)
            self.prolog_engine = None
        
        self.faker = Faker('en_IN')
        self.templates = self._initialize_templates()
        self.demographics = self._initialize_demographics()
        self.noise_patterns = self._initialize_noise_patterns()
        self.negative_snippets = self._initialize_negative_snippets()
        logger.info("Generator initialized successfully.")

    def _initialize_templates(self) -> Dict[str, List[Dict]]:
        """Initializes a wide variety of templates for generating legal queries."""
        # Note: The 'eligibility_logic' lambdas are removed. Prolog is the source of truth.
        # Templates now focus on generating text from variables.
        return {
            LegalDomain.LEGAL_AID.value: [
                {
                    "template_id": "legal_aid_general_001",
                    "template": "{greeting}, I am a {social_category_desc} earning Rs {income} {income_frequency}. My case is about {case_description}. {question}",
                    "variables": {
                        "greeting": lambda: random.choice(["Sir", "Madam", "Hello", "Namaste"]),
                        "social_category_desc": lambda: random.choice(["a poor woman", "a daily wage worker", "a senior citizen", "a disabled person", "a member of the SC community", "a member of the ST community"]),
                        "income": lambda: random.randint(50000, 500000),
                        "income_frequency": lambda: random.choice(["annually", "per year"]),
                        "case_description": lambda: random.choice(["a land dispute", "a family matter", "a police complaint"]),
                        "question": lambda: random.choice(["Am I eligible for free legal help?", "Can I get a government lawyer?"])
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}')."
                    ]
                },
                {
                    "template_id": "legal_aid_vulnerable_002",
                    "template": "I am a {vulnerable_group_desc}. {case_description}. Do I get free legal aid?",
                    "variables": {
                        "vulnerable_group_desc": lambda: random.choice(["woman who is a victim of domestic violence", "person in custody", "disabled person", "senior citizen over 60"]),
                        "case_description": lambda: random.choice(["I need to file for divorce", "I was arrested yesterday", "My landlord is evicting me"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"vulnerable_group(case_gen, '{self._map_desc_to_prolog_vulnerability(v['vulnerable_group_desc'])}')."
                    ]
                },
                {
                    "template_id": "legal_aid_industrial_worker_003",
                    "template": "I work as an {occupation} in a factory in {location}. My annual income is {income}. Can I avail free legal services?",
                    "variables": {
                        "occupation": lambda: "industrial workman",
                        "location": lambda: self.faker.city(),
                        "income": lambda: random.randint(150000, 700000),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"occupation(case_gen, industrial_workman).",
                        f"annual_income(case_gen, {v['income']})."
                    ]
                },
                {
                    "template_id": "legal_aid_disaster_victim_004",
                    "template": "My home was destroyed in the recent {disaster}. I have lost everything. I need legal help to get compensation. Am I eligible for aid?",
                    "variables": {
                        "disaster": lambda: random.choice(["flood", "earthquake", "cyclone"]),
                        "income": lambda: random.randint(20000, 100000), # Usually low income
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"vulnerable_group(case_gen, victim_of_natural_disaster).",
                        f"annual_income(case_gen, {v['income']})."
                    ]
                }
            ],
            LegalDomain.FAMILY_LAW.value: [
                {
                    "template_id": "family_law_custody_001",
                    "template": "My husband and I are separating. We were married for {duration} years. He was {behavior}. I am {social_category_desc} and want custody of my child. My income is {income} per year. Can I get legal aid?",
                    "variables": {
                        "duration": lambda: random.randint(1, 20),
                        "behavior": lambda: random.choice(["violent", "an alcoholic", "unemployed"]),
                        "income": lambda: random.randint(80000, 400000),
                        "social_category_desc": lambda: random.choice(["a woman from the general category", "a woman from the SC category", "a woman from the ST category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, family_law).",
                        f"has_grounds(case_gen, domestic_violence).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}')."
                    ]
                },
                {
                    "template_id": "family_law_maintenance_002",
                    "template": "I need help getting child maintenance from my ex-husband. He earns around {father_income} per month but is not paying for our {num_children} child's expenses. My own annual income is {mother_income} and I am from the {social_category_desc}. Can I get a free lawyer for this?",
                    "variables": {
                        "father_income": lambda: random.randint(25000, 150000),
                        "num_children": lambda: random.randint(1, 3),
                        "mother_income": lambda: random.randint(50000, 400000),
                        "social_category_desc": lambda: random.choice(["general category", "SC category", "ST category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, family_law).",
                        f"has_grounds(case_gen, child_maintenance).",
                        f"annual_income(case_gen, {v['mother_income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}').",
                        f"respondent_income(case_gen, {v['father_income'] * 12}).",
                        f"number_of_children(case_gen, {v['num_children']})."
                    ]
                },
                {
                    "template_id": "family_law_divorce_mutual_003",
                    "template": "My wife and I have decided to get a divorce mutually. We have no children. My income is {income} and I am {social_category_desc}. Do we still need a lawyer and can we get legal aid?",
                    "variables": {
                        "income": lambda: random.randint(200000, 800000),
                        "social_category_desc": lambda: random.choice(["general category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, family_law).",
                        f"has_grounds(case_gen, mutual_consent_divorce).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}')."
                    ]
                }
            ],
            LegalDomain.CONSUMER_PROTECTION.value: [
                {
                    "template_id": "consumer_protection_defective_product_001",
                    "template": "I bought a {product} for Rs {amount} from {seller}. It stopped working after {time_since}. I am from the {social_category_desc} and my annual income is {income}. Can I file a consumer case and get free legal help?",
                    "variables": {
                        "product": lambda: random.choice(["TV", "fridge", "washing machine", "mobile phone"]),
                        "amount": lambda: random.randint(10000, 50000),
                        "seller": lambda: random.choice(["a local shop", "an online store"]),
                        "time_since": lambda: random.choice(["2 months", "6 months"]),
                        "income": lambda: random.randint(90000, 450000),
                        "social_category_desc": lambda: random.choice(["general category", "SC category", "ST category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, consumer_protection).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}').",
                        f"claim_value(case_gen, {v['amount']}).",
                        f"defect_manifestation_period(case_gen, {int(re.search('[0-9]+', v['time_since']).group())})."
                    ]
                },
                {
                    "template_id": "consumer_protection_service_deficiency_002",
                    "template": "I booked a holiday package to {destination} for Rs {amount} which promised a 5-star hotel, but they gave us a 2-star hotel with no facilities. My income is {income} and I am from the {social_category_desc}. Can I get free legal assistance?",
                    "variables": {
                        "destination": lambda: random.choice(["Goa", "Kerala", "Shimla", "Rajasthan"]),
                        "amount": lambda: random.randint(20000, 100000),
                        "income": lambda: random.randint(90000, 450000),
                        "social_category_desc": lambda: random.choice(["general category", "SC category", "ST category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, consumer_protection).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}').",
                        f"claim_value(case_gen, {v['amount']}).",
                        f"has_grounds(case_gen, deficiency_in_service)."
                    ]
                },
                {
                    "template_id": "consumer_protection_medical_negligence_003",
                    "template": "My relative was admitted to {hospital_type} for a minor surgery and died due to negligence. The hospital is demanding a huge bill. My income is {income}. Can I file a case and get legal aid?",
                    "variables": {
                        "hospital_type": lambda: random.choice(["a private hospital", "a government hospital"]),
                        "income": lambda: random.randint(100000, 500000),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, consumer_protection).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"has_grounds(case_gen, medical_negligence)."
                    ]
                }
            ],
            LegalDomain.EMPLOYMENT_LAW.value: [
                {
                    "template_id": "employment_law_termination_001",
                    "template": "I worked at a company for {duration} years. They fired me without giving a proper notice period. My annual salary was {salary} and I belong to the {social_category_desc}. Am I eligible for legal aid to sue them?",
                    "variables": {
                        "duration": lambda: random.randint(1, 10),
                        "salary": lambda: random.randint(100000, 600000),
                        "social_category_desc": lambda: random.choice(["general category", "SC category", "ST category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, employment_law).",
                        f"annual_income(case_gen, {v['salary']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}').",
                        f"employment_duration(case_gen, {v['duration']}).",
                        f"has_grounds(case_gen, wrongful_termination)."
                    ]
                },
                {
                    "template_id": "employment_law_harassment_002",
                    "template": "I am facing severe harassment at my workplace from my manager. I have complained to HR but no action has been taken. My salary is {salary} per year and I am from the {social_category_desc}. Can I get legal aid to file a case?",
                    "variables": {
                        "salary": lambda: random.randint(100000, 600000),
                        "social_category_desc": lambda: random.choice(["general category", "SC category", "ST category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, employment_law).",
                        f"annual_income(case_gen, {v['salary']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}').",
                        f"has_grounds(case_gen, workplace_harassment)."
                    ]
                },
                {
                    "template_id": "employment_law_unpaid_salary_003",
                    "template": "My previous employer has not paid my salary for the last {months} months. The total amount is {amount}. My current financial situation is not good. Can I get free legal help to recover my dues?",
                    "variables": {
                        "months": lambda: random.randint(2, 6),
                        "amount": lambda: random.randint(50000, 200000),
                        "income": lambda: random.randint(80000, 300000),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, employment_law).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"has_grounds(case_gen, unpaid_salary).",
                        f"claim_value(case_gen, {v['amount']})."
                    ]
                }
            ],
            LegalDomain.FUNDAMENTAL_RIGHTS.value: [
                 {
                    "template_id": "fundamental_rights_discrimination_001",
                    "template": "I was prevented from entering a public temple because of my caste. Is this a violation of my fundamental rights? Can I get a free lawyer to file a case?",
                    "variables": {
                         "income": lambda: random.randint(100000, 600000),
                         "caste": lambda: random.choice(["a scheduled caste", "a scheduled tribe"])
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, fundamental_rights).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['caste'])}').",
                        f"has_grounds(case_gen, discrimination)."
                    ]
                },
                {
                    "template_id": "fundamental_rights_assembly_002",
                    "template": "The police are not allowing us to hold a peaceful protest in a public ground. We are protesting against {issue}. Is this a violation of our rights? We are from the {social_category_desc} and our members are low-income. Can we get a free lawyer?",
                    "variables": {
                        "issue": lambda: random.choice(["new government policies", "local corruption", "environmental damage"]),
                        "income": lambda: random.randint(100000, 600000),
                        "social_category_desc": lambda: random.choice(["general category", "SC category", "ST category", "OBC category"]),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, fundamental_rights).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"social_category(case_gen, '{self._map_desc_to_prolog_category(v['social_category_desc'])}').",
                        f"has_grounds(case_gen, right_to_assemble)."
                    ]
                },
                {
                    "template_id": "fundamental_rights_speech_003",
                    "template": "I wrote a blog post criticizing a government policy and now I am facing a police inquiry. Do I have the right to express my opinion? Can I get legal aid to defend myself?",
                    "variables": {
                        "income": lambda: random.randint(120000, 550000),
                    },
                    "fact_generator": lambda v: [
                        f"person(case_gen).",
                        f"case_type(case_gen, fundamental_rights).",
                        f"annual_income(case_gen, {v['income']}).",
                        f"has_grounds(case_gen, freedom_of_speech)."
                    ]
                }
            ]
        }

    def _map_desc_to_prolog_category(self, desc: str) -> str:
        if "sc" in desc or "scheduled caste" in desc: return "sc"
        if "st" in desc or "scheduled tribe" in desc: return "st"
        if "obc" in desc: return "obc"
        return "general"

    def _map_desc_to_prolog_vulnerability(self, desc: str) -> str:
        if "violence" in desc: return "woman_victim_of_violence"
        if "custody" in desc: return "person_in_custody"
        if "disabled" in desc: return "disabled"
        if "senior" in desc: return "senior_citizen"
        if "disaster" in desc: return "victim_of_natural_disaster"
        return "none"

    def _initialize_demographics(self) -> List[Dict]:
        """Generates a pool of diverse user demographic profiles."""
        return [
            {"name": self.faker.name(), "location": self.faker.city_name(), "age": random.randint(18, 80), "gender": random.choice(["male", "female", "other"])}
            for _ in range(200)
        ]

    def _initialize_noise_patterns(self) -> List[callable]:
        """Functions to add realistic noise to queries."""
        def add_typos(text):
            if random.random() < 0.15:
                i = random.randint(0, len(text) - 1)
                if text[i].isalpha():
                    return text[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[i+1:]
            return text
        def use_slang(text):
            if random.random() < 0.1:
                return text.replace("Am I eligible", "Meko milega kya free lawyer")
            return text
        return [add_typos, use_slang]

    def _initialize_negative_snippets(self) -> List[str]:
        """Initializes a pool of irrelevant or misleading information snippets."""
        return [
            "I have been feeling very stressed lately.",
            "My son is doing well in school.",
            "The weather has been very hot.",
            "I am thinking of buying a new phone.",
            "My neighbor is very noisy.",
            "I like to watch cricket.",
            "I had a fight with my friend yesterday."
        ]

    def _apply_noise(self, query: str, complexity: str) -> str:
        """Applies random noise and negative facts based on complexity."""
        # Apply basic noise like typos or slang
        if self.noise_patterns and random.random() < 0.2:
            noise_fn = random.choice(self.noise_patterns)
            query = noise_fn(query)

        # Add negative snippets for medium and high complexity
        if complexity in ["medium", "high"] and random.random() < 0.5:
            snippet = random.choice(self.negative_snippets)
            query = f"{query} {snippet}"
        
        return query

    # FIX 1: Correct the fact generation to be lowercase.
    def _demographics_to_prolog_facts(self, demographics: Dict[str, Any]) -> List[str]:
        """Converts a demographics dictionary to a list of Prolog facts."""
        facts = []
        
        # --- START OF CORRECTION 1 ---
        # Convert category to lowercase to perfectly match the Prolog rules.
        if 'category' in demographics:
            facts.append(f"social_category(case_gen, '{demographics['category'].lower()}').")
        # --- END OF CORRECTION 1 ---
            
        if 'age' in demographics:
            facts.append(f"age(case_gen, {demographics['age']}).")
        if 'gender' in demographics:
            facts.append(f"gender(case_gen, '{demographics['gender'].lower()}').")
        if 'income' in demographics:
            facts.append(f"annual_income(case_gen, {demographics['income']}).")
        if demographics.get('vulnerable'): # Use .get() for safety
            facts.append(f"is_vulnerable(case_gen, true).")
        if 'occupation' in demographics:
            facts.append(f"occupation(case_gen, '{demographics['occupation'].lower().replace(' ', '_')}').")

        return facts

    # FIX 2: Correct the goal-directed demographics generation for precision.
    def _generate_demographics(self, domain: LegalDomain, target_eligibility: bool) -> Dict[str, Any]:
        """
        Generates realistic user demographics, precisely biased to meet the target_eligibility.
        """
        # Start with a default INELIGIBLE case

        # Ensure the generated occupation is not one that grants special status.
        occupation = self.faker.job().lower()
        while occupation == "industrial worker":
            occupation = self.faker.job().lower()

        demographics = {
            "age": self.faker.random_int(min=25, max=55),
            "gender": "male",
            "category": "General",
            "location": self.faker.city(),
            "income": self.faker.random_int(min=1000000, max=5000000), # High income
            "vulnerable": False,
            "occupation": occupation
        }

        if target_eligibility:
            # If we WANT an eligible case, precisely engineer the facts to meet one of the rules.
            eligibility_method = random.choice(['category', 'vulnerable_age', 'income'])

            if eligibility_method == 'category':
                # This is a guaranteed pass for the 'categorically_eligible' rule in Prolog.
                demographics['category'] = random.choice(['SC', 'ST', 'BPL', 'OBC'])

            elif eligibility_method == 'vulnerable_age':
                # This is a guaranteed pass for the 'vulnerable_group' rule.
                demographics['age'] = self.faker.random_int(min=61, max=90)
                
            else: # income-based
                # --- START OF CORRECTION 2 ---
                # Precisely choose a category and generate an income that fits its specific threshold.
                cat = random.choice(['General', 'OBC', 'SC', 'ST'])
                demographics['category'] = cat
                
                # Get the SPECIFIC threshold for the chosen category from the config.
                # Fallback to 'general' if the key doesn't exist.
                threshold_key = cat.lower() if cat in ['SC', 'ST', 'OBC'] else 'general'
                income_threshold = self.config.ENTITY_CONFIG['income_thresholds'].get(threshold_key, 500000)
                
                # Generate an income safely below that specific threshold.
                demographics['income'] = self.faker.random_int(min=10000, max=income_threshold - 1)
                # --- END OF CORRECTION 2 ---
                
        # Finally, ensure the 'vulnerable' flag is consistent with the generated age.
        if demographics['age'] >= 60:
            demographics['vulnerable'] = True
            
        return demographics

    def _generate_single_example(self, domain: str, template: Dict, target_eligibility: bool) -> Optional[LegalTrainingExample]:
        """
        Generates a single, logically consistent training example using a goal-directed approach.
        """
        for _ in range(10): # Should be much faster now
            # 1. Generate demographics biased towards the target eligibility.
            demographics = self._generate_demographics(LegalDomain(domain), target_eligibility)

            # 2. Generate text variables from the template's lambdas.
            vars = {k: v_func() for k, v_func in template["variables"].items()}

            # 3. Overwrite the generated vars with our definite demographics.
            # This ensures the generated text will be consistent with the facts.
            vars.update(demographics)
            
            # Map the Prolog-style category from demographics back to a human-readable description for the text.
            if 'social_category_desc' in vars:
                cat_map = {
                    "SC": "a member of the SC community", "ST": "a member of the ST community",
                    "OBC": "a member of the OBC community", "General": "a general category person",
                    "BPL": "a person below the poverty line"
                }
                vars['social_category_desc'] = cat_map.get(demographics['category'], "a general category person")

            # 4. Generate Prolog facts.
            # First, get the demographic facts from our corrected function.
            prolog_facts = self._demographics_to_prolog_facts(demographics)
            # Then, add the other non-demographic facts from the template's generator.
            other_facts = template["fact_generator"](vars)
            
            # Combine them, avoiding duplicates.
            for fact in other_facts:
                # Add fact if it's not a demographic one already handled by _demographics_to_prolog_facts
                if not any(p_fact.split('(')[0] == fact.split('(')[0] for p_fact in prolog_facts):
                    prolog_facts.append(fact)


            # 5. Get ground truth from Prolog Engine.
            is_eligible, reason = self.prolog_engine.determine_eligibility_for_generation(prolog_facts)

            # 6. If we matched the target, create and return the example.
            if is_eligible == target_eligibility:
                complexity = random.choice(["low", "medium", "high"])
                query = template["template"].format(**vars)
                query = self._apply_noise(query, complexity)

                human_facts = [f.replace("case_gen", "user").replace("_", " ").replace(".", "") for f in prolog_facts]
                
                return LegalTrainingExample(
                    query=query,
                    domains=[domain],
                    extracted_facts=human_facts,
                    prolog_facts=prolog_facts,
                    expected_eligibility=is_eligible,
                    legal_reasoning=reason,
                    user_demographics=demographics, # Use the generated demographics
                    case_complexity=complexity
                )
        
        logger.warning(f"Failed to generate a sample for domain={domain} with target_eligibility={target_eligibility} after 10 attempts.")
        return None

    def generate_dataset_for_domain(self, domain: str, count: int) -> List[LegalTrainingExample]:
        """Generates a balanced dataset for a specific domain."""
        samples = []
        
        eligible_needed = count // 2
        ineligible_needed = count - eligible_needed

        # --- Loop for ELIGIBLE samples ---
        logger.info(f"Generating {eligible_needed} eligible samples for {domain}...")
        generated_count = 0
        while generated_count < eligible_needed:
            # For eligible cases, all templates are generally fine.
            template = random.choice(self.templates[domain])
            example = self._generate_single_example(domain, template, target_eligibility=True)
            if example:
                samples.append(example)
                generated_count += 1

        # --- Loop for INELIGIBLE samples ---
        logger.info(f"Generating {ineligible_needed} ineligible samples for {domain}...")
        ineligible_count = 0
        while ineligible_count < ineligible_needed:
            # --- Start of new code ---
            # Filter templates to avoid logical contradictions
            compatible_templates = self.templates[domain]
            if domain == 'legal_aid':
                # When creating an INELIGIBLE legal aid case, exclude templates that hardcode automatic eligibility.
                ineligible_templates = {
                    'legal_aid_vulnerable_002',      # Excludes cases based on general vulnerability
                    'legal_aid_disaster_victim_004', # Excludes cases for disaster victims (vulnerable)
                    'legal_aid_industrial_worker_003'# Excludes cases for industrial workers (special category)
                }
                compatible_templates = [
                    t for t in compatible_templates
                    if t.get('template_id') not in ineligible_templates
                ]

            # Ensure there are still templates to choose from after filtering
            if not compatible_templates:
                logger.warning(f"No compatible templates found for domain={domain}, eligibility=False. Skipping sample generation.")
                break # Exit the while loop

            template = random.choice(compatible_templates)
            # --- End of new code ---
            
            example = self._generate_single_example(domain, template, target_eligibility=False)
            if example:
                samples.append(example)
                ineligible_count += 1
        
        return samples

    def generate_comprehensive_dataset(self, total_samples: int = 15000) -> Tuple[List, List, List]:
        """
        Generates the full dataset across all domains and splits it.
        """
        dataset = []
        
        samples_per_domain = total_samples // len(self.templates)

        for domain, templates in self.templates.items():
            logger.info(f"--- Starting domain: {domain} ---")
            domain_samples = self.generate_dataset_for_domain(domain, samples_per_domain)
            dataset.extend(domain_samples)
            logger.info(f"--- Finished domain: {domain}, Total samples: {len(dataset)} ---")

        random.shuffle(dataset)

        # Convert to dicts for JSON serialization
        dataset_dicts = [asdict(s) for s in dataset]

        # Stratified split
        try:
            # Ensure there's more than one class to stratify on
            labels = [d['expected_eligibility'] for d in dataset_dicts]
            if len(set(labels)) < 2:
                logger.warning("Cannot perform stratified split: only one class present in the dataset. Using regular split.")
                raise ValueError("Only one class present.")

            train_data, temp_data = train_test_split(
                dataset_dicts,
                test_size=0.30,
                random_state=42,
                stratify=labels
            )
            val_data, test_data = train_test_split(
                temp_data,
                test_size=0.50,
                random_state=42,
                stratify=[d['expected_eligibility'] for d in temp_data]
            )
            return train_data, val_data, test_data
        except ValueError as e:
            logger.error(f"Could not perform stratified split, likely due to insufficient samples of one class in a domain: {e}")
            # Fallback to non-stratified split
            train_size = int(0.7 * len(dataset_dicts))
            val_size = int(0.15 * len(dataset_dicts))
            train_data = dataset_dicts[:train_size]
            val_data = dataset_dicts[train_size:train_size + val_size]
            test_data = dataset_dicts[train_size + val_size:]
            return train_data, val_data, test_data


    def save_datasets(self, train: List, val: List, test: List, output_dir: Path):
        """Saves the datasets to JSON files."""
        output_dir.mkdir(exist_ok=True)
        
        # Clean up old files first
        for old_file in ["train_samples.json", "val_samples.json", "test_samples.json"]:
            if (output_dir / old_file).exists():
                os.remove(output_dir / old_file)
                logger.info(f"Removed old dataset file: {output_dir / old_file}")

        with open(output_dir / "train_samples.json", 'w', encoding='utf-8') as f:
            json.dump(train, f, indent=2, ensure_ascii=False)
        with open(output_dir / "val_samples.json", 'w', encoding='utf-8') as f:
            json.dump(val, f, indent=2, ensure_ascii=False)
        with open(output_dir / "test_samples.json", 'w', encoding='utf-8') as f:
            json.dump(test, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved new datasets to {output_dir}")


def main():
    """Main function to run the data generation process."""
    logger.info("--- Starting Comprehensive Legal Data Generation ---")
    
    # The output directory is `multi_domain_legal/data/`
    output_data_dir = project_root / "data"
    
    generator = ComprehensiveLegalDataGenerator()
    
    total_samples_to_generate = 15000 
    logger.info(f"Targeting a total of {total_samples_to_generate} samples.")

    train, val, test = generator.generate_comprehensive_dataset(total_samples=total_samples_to_generate)
    
    if not train or not val or not test:
        logger.error("Dataset generation failed to produce one or more data splits. Aborting.")
        return

    generator.save_datasets(train, val, test, output_data_dir)

    logger.info("--- Data Generation Complete ---")
    total_generated = len(train) + len(val) + len(test)
    logger.info(f"Generated: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    logger.info(f"Total samples generated: {total_generated}")
    print(f"\n✅ New, logically consistent datasets saved to '{output_data_dir.relative_to(Path.cwd())}' folder.")
    print("   You can now proceed with model training.")

if __name__ == "__main__":
    main()
