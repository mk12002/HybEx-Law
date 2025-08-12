"""
Comprehensive Legal Data Generation for HybEx-Law System - Improved Version.

This enhanced script generates higher-quality, more diverse training data for the hybrid neural-symbolic
legal AI system. Improvements include:
- Increased template diversity with more realistic, varied scenarios in Indian English (formal/informal, regional variations).
- Better balance: 50% eligible, 50% ineligible per domain.
- Added noise: Typos, slang, incomplete sentences for robustness.
- Multi-domain integration: More cross-domain samples with logical connections.
- Edge cases: Explicitly generate ambiguous, borderline, and negative examples.
- Larger scale: 30,000 total samples for better generalization.
- Stratified splits: Generate train (70%), val (15%), test (15%) JSON files directly.
- Domain-specific enhancements: More nuanced facts tied to Indian laws (e.g., specific acts, thresholds).
"""

import json
import random
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDomain(Enum):
    """Legal domains covered by the system"""
    LEGAL_AID = "legal_aid"
    FAMILY_LAW = "family_law"
    CONSUMER_PROTECTION = "consumer_protection"
    FUNDAMENTAL_RIGHTS = "fundamental_rights"
    EMPLOYMENT_LAW = "employment_law"

@dataclass
class LegalTrainingExample:
    """Structured training example for legal AI system"""
    query: str
    domains: List[str]
    extracted_facts: List[str]
    expected_eligibility: bool
    legal_reasoning: str
    confidence_factors: Dict[str, float]
    user_demographics: Dict[str, Any]
    case_complexity: str
    priority_level: str

class EnhancedLegalDataGenerator:
    """
    Improved generator for diverse, balanced legal training data.
    Focuses on realism, diversity, and balance for better model training.
    """
    
    def __init__(self):
        self.legal_templates = self._initialize_enhanced_templates()
        self.demographic_profiles = self._initialize_diverse_demographics()
        self.case_variations = self._initialize_case_variations()
        self.noise_patterns = self._initialize_noise_patterns()  # New: For robustness

    def _initialize_enhanced_templates(self) -> Dict[str, List[Dict]]:
        """Initialize diverse legal query templates with improved realism and variations"""
        return {
            "legal_aid": [
                # Income-based: Balanced eligible/ineligible, varied language
                {
                    "template": "{greeting}, I am a {social_category} {occupation} earning Rs {income} {income_frequency}. {case_description}. {question} I stay in {location}.",
                    "variables": {
                        "greeting": ["Sir", "Madam", "Respected sir/madam", "Hello ji", "Namaste"],
                        "social_category": ["poor woman", "dalit woman", "tribal woman", "disabled person", "widow", "divorced woman", "female laborer", "SC man", "ST farmer", "OBC artisan", "general category senior", "EWS youth"],
                        "occupation": ["daily wage worker", "farmer", "housewife", "street vendor", "construction laborer", "domestic helper", "rickshaw driver", "small shop owner"],
                        "income": [str(random.randint(0, 1000000)) for _ in range(50)],  # Wide range for balance
                        "income_frequency": ["monthly", "per year", "annually", "per month", "daily but total yearly"],
                        "location": ["village in UP", "slum in Mumbai", "remote area in Jharkhand", "tehsil in Rajasthan", "district in Bihar", "city in Delhi", "tribal area in Odisha"],
                        "case_description": [
                            "Husband abandoned me and kids no support",
                            "Landlord evicting without notice wrongfully",
                            "Police beat me no reason at all",
                            "Doctor did negligence in treatment now demanding extra money",
                            "Builder took advance left house half-built",
                            "Sarpanch demanding bribe for scheme",
                            "Neighbor grabbed my land with fake papers",
                            "Boss fired me when pregnant",
                            "Got fake product online no refund",  # Cross-domain hint
                            "Job lost due to caste discrimination",  # Cross-domain
                            "Child custody battle with ex-husband",
                            "Consumer court case for defective fridge",
                            "Fundamental right violated by police custody"
                        ],
                        "question": ["Am I eligible for free legal help?", "Can I get legal aid?", "How to get pro bono lawyer?", "Is free lawyer possible for me?", "Legal services free for poor like me?"]
                    },
                    "eligibility_logic": lambda vars: int(vars['income']) <= 300000 if 'sc' in vars['social_category'].lower() or 'st' in vars['social_category'].lower() else int(vars['income']) <= 100000,  # Balanced thresholds
                    "expected_domains": ["legal_aid"],
                    "complexity": random.choice(["low", "medium"])
                },
                # Vulnerable group focused
                {
                    "template": "I am {vulnerable_group} from {location}. {case_description}. {eligibility_question}",
                    "variables": {
                        "vulnerable_group": ["senior citizen 70 years", "disabled with 50% handicap", "minor child 15 years", "woman victim of violence", "transgender person", "HIV patient", "mental health patient", "person in custody", "victim of a natural disaster"],
                        "location": ["rural area", "urban slum", "hill station", "coastal village", "a shelter home", "jail"],
                        "case_description": ["Facing property dispute with relatives", "Denied pension by government", "School harassment case", "Domestic abuse by in-laws", "Discrimination at workplace", "Medical negligence in hospital", "Need to file for bail", "Lost all documents in flood"],
                        "eligibility_question": ["Do I qualify for legal aid?", "Free legal support available?", "How to apply for legal services?"]
                    },
                    "eligibility_logic": lambda vars: True if any(kw in vars['vulnerable_group'] for kw in ["senior", "disabled", "minor", "woman victim", "custody", "disaster"]) else random.choice([True, False]),  # Bias toward eligible but balanced
                    "expected_domains": ["legal_aid"],
                    "complexity": "medium"
                },
                # Case-specific eligibility
                {
                    "template": "My case is about {case_type_specific}. I am from {social_category} community. Can I get a free government lawyer?",
                    "variables": {
                        "case_type_specific": ["defending a criminal case", "a land dispute", "filing for a consumer complaint", "a matter of untouchability", "a bail application"],
                        "social_category": ["SC", "ST", "OBC", "a minority", "general"],
                    },
                    "eligibility_logic": lambda vars: True if any(kw in vars['social_category'] for kw in ["SC", "ST"]) or "untouchability" in vars['case_type_specific'] else False,
                    "expected_domains": ["legal_aid", "fundamental_rights"],
                    "complexity": "medium"
                },
                # Informal/confused query
                {
                    "template": "help please... {problem}. money is very less, maybe {income} per month. {location}. what to do?",
                    "variables": {
                        "problem": ["my landlord is a big problem", "cheated by online seller", "police not listening to my complaint", "family matter is very tense"],
                        "income": [str(random.randint(2000, 15000))],
                        "location": ["a small village", "the city outskirts", "near the railway lines"],
                    },
                    "eligibility_logic": lambda vars: int(vars['income']) * 12 <= 100000,
                    "expected_domains": ["legal_aid"],
                    "complexity": "low"
                },
                # High income but vulnerable category
                {
                    "template": "I am a {vulnerable_group} and my annual income is Rs {income}. I need legal help for {case_description}. Am I excluded due to my income?",
                    "variables": {
                        "vulnerable_group": ["woman facing domestic violence", "Scheduled Tribe member", "person with a disability"],
                        "income": [str(random.randint(500000, 1500000))],
                        "case_description": ["a divorce case", "a land rights issue against a corporation", "a workplace discrimination suit"],
                    },
                    "eligibility_logic": lambda vars: True,  # Vulnerable categories are often eligible regardless of income
                    "expected_domains": ["legal_aid", "family_law", "employment_law"],
                    "complexity": "high"
                }
            ],
            "family_law": [
                # Divorce/maintenance: Balanced outcomes
                {
                    "template": "{informal_start} I am {gender} {marital_status}, husband/wife {behavior}. {income_info}. {child_info}. {question}",
                    "variables": {
                        "informal_start": ["Bhai sahab", "Didi", "Please help", "My problem is"],
                        "gender": ["woman", "man", "wife", "husband"],
                        "marital_status": ["married for 5 years", "in a live-in relationship for 3 years", "separated for 1 year", "divorced but alimony pending"],
                        "behavior": ["beats me daily", "left home 2 years ago", "has an affair", "doesn't give money for the house", "is mentally cruel", "threatens to kill"],
                        "income_info": ["I earn Rs {income} but he earns more", "No job, dependent on him/her", "Both working but he/she hides income"],
                        "child_info": ["We have {num_children} kids", "One daughter 10 years old", "No children"],
                        "question": ["Can I get divorce and maintenance?", "Is legal aid available for family court?", "How to file a custody case for free?"]
                    },
                    "eligibility_logic": lambda vars: True if 'woman' in vars['gender'] or int(vars.get('income', 99999)) < 20000 or 'beats' in vars['behavior'] else False,
                    "expected_domains": ["family_law", "legal_aid"],
                    "complexity": random.choice(["medium", "high"])
                },
                # Child Custody
                {
                    "template": "My spouse and I are separating. I want {custody_type} of my {child_details}. My spouse is {spouse_condition}. Am I likely to get custody? Is there free legal help for this?",
                    "variables": {
                        "custody_type": ["sole custody", "full custody", "primary custody"],
                        "child_details": ["son who is 7 years old", "daughter aged 12", "two children, 5 and 8"],
                        "spouse_condition": ["an alcoholic", "violent", "unemployed", "a good parent but lives in another city", "not interested in the child"],
                        "income": [str(random.randint(10000, 500000))],
                    },
                    "eligibility_logic": lambda vars: True if any(kw in vars['spouse_condition'] for kw in ["alcoholic", "violent"]) else False,
                    "expected_domains": ["family_law", "legal_aid"],
                    "complexity": "high"
                },
                # Adoption
                {
                    "template": "My partner and I want to adopt a child. We are {couple_status} and have a combined income of Rs {income} annually. What is the legal procedure and can we get assistance?",
                    "variables": {
                        "couple_status": ["a married couple", "in a live-in relationship", "a single woman", "a single man"],
                        "income": [str(random.randint(400000, 2000000))],
                    },
                    "eligibility_logic": lambda vars: int(vars['income']) < 800000, # Adoption procedures themselves don't have eligibility, this is for legal aid
                    "expected_domains": ["family_law", "legal_aid"],
                    "complexity": "medium"
                },
                # Domestic Violence
                {
                    "template": "I am a victim of domestic violence. My {aggressor} is {abusive_action}. I live in {location} and I am scared. How can I get a protection order immediately with a free lawyer?",
                    "variables": {
                        "aggressor": ["husband", "in-laws", "live-in partner", "son"],
                        "abusive_action": ["physically abusive", "emotionally and verbally abusive", "controlling all my finances", "constantly threatening me"],
                        "location": ["a joint family", "our own flat", "a rented house"],
                    },
                    "eligibility_logic": lambda vars: True, # Victims of domestic violence are a priority category
                    "expected_domains": ["family_law", "legal_aid"],
                    "complexity": "high"
                }
            ],
            "consumer_protection": [
                # Complaints: Balanced valid/invalid
                {
                    "template": "Bought {product} for Rs {amount} from {seller}. {issue}. {time_since}. {question}",
                    "variables": {
                        "product": ["mobile phone", "refrigerator", "car", "online course", "medicines", "groceries", "a laptop"],
                        "amount": [str(random.randint(500, 1000000))],
                        "seller": ["Amazon", "a local shop", "Flipkart", "the official dealer", "a pharmacy", "an online education portal"],
                        "issue": ["it was defective from day 1", "it's not as advertised", "it was an expired product", "they never delivered it", "they overcharged me", "it was a fake item"],
                        "time_since": ["1 month ago", "2 years back", "just yesterday", "6 months ago"],
                        "question": ["Can I file a consumer case? Is legal aid possible?", "Am I eligible for a free lawyer in the consumer forum?", "How do I get compensation without paying high fees?"]
                    },
                    "eligibility_logic": lambda vars: True if int(vars['amount']) < 2000000 and 'defective' in vars['issue'] and int(re.search(r'\d+', vars['time_since'])[0]) <= 24 else False,  # 2-year limit
                    "expected_domains": ["consumer_protection", "legal_aid"],
                    "complexity": "low"
                },
                # Deficiency in Service
                {
                    "template": "I have a problem with {service_provider} regarding {service_issue}. I have complained many times but there is no resolution. Can I take them to consumer court? My income is low.",
                    "variables": {
                        "service_provider": ["my bank", "Airtel", "the electricity board", "an insurance company", "a hospital"],
                        "service_issue": ["unfair bank charges", "poor network and false billing", "frequent power cuts", "rejection of a valid insurance claim", "medical negligence during treatment"],
                    },
                    "eligibility_logic": lambda vars: True if any(kw in vars['service_issue'] for kw in ["unfair", "false billing", "rejection"]) else False,
                    "expected_domains": ["consumer_protection", "legal_aid"],
                    "complexity": "medium"
                },
                # Unfair Trade Practice
                {
                    "template": "I saw an advertisement for {product} that claimed {false_claim}. After buying it, I found out it was a lie. This feels like an unfair trade practice. What are my rights?",
                    "variables": {
                        "product": ["a face cream", "a health supplement", "a coaching class", "an investment scheme"],
                        "false_claim": ["'guaranteed results in 7 days'", "'100% risk-free'", "'government approved' but it was not", "'buy one get one free' but the price was doubled"],
                    },
                    "eligibility_logic": lambda vars: True,
                    "expected_domains": ["consumer_protection"],
                    "complexity": "medium"
                }
            ],
            "fundamental_rights": [
                # Rights violations: Balanced
                {
                    "template": "Police {action} without reason. I am {category}. {details}. {question}",
                    "variables": {
                        "action": ["arrested me illegally", "beat me in custody", "denied my right to vote", "discriminated against me based on my caste"],
                        "category": ["a Muslim", "a Christian", "a Dalit", "a woman", "a journalist", "a student activist"],
                        "details": ["No FIR was shown", "I was tortured for a confession", "My speech was censored online", "My property was seized unfairly"],
                        "question": ["Is this a violation of Article {article}? Can I get legal aid for a PIL?", "Can I get a free lawyer for a human rights case?"]
                    },
                    "eligibility_logic": lambda vars: True if any(kw in vars['action'] + vars['details'] for kw in ["arrested illegally", "tortured", "censored"]) else False,
                    "expected_domains": ["fundamental_rights", "legal_aid"],
                    "complexity": "high"
                },
                # Right to Equality
                {
                    "template": "I was denied entry into {place} because of my {reason_for_denial}. Is this not a violation of my right to equality under the Constitution?",
                    "variables": {
                        "place": ["a public temple", "a restaurant", "a housing society"],
                        "reason_for_denial": ["caste", "religion", "gender", "sexual orientation"],
                    },
                    "eligibility_logic": lambda vars: True,
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "high"
                },
                # Freedom of Speech
                {
                    "template": "I wrote a blog post criticizing a government policy and now I am facing {consequence}. Is my freedom of speech being violated?",
                    "variables": {
                        "consequence": ["a police inquiry", "threats from political workers", "a suspension from my university"],
                    },
                    "eligibility_logic": lambda vars: True if "threats" in vars['consequence'] else random.choice([True, False]),
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "high"
                }
            ],
            "employment_law": [
                # Labor issues: Balanced
                {
                    "template": "I was working as a {job} for {duration}. My boss {issue}. My salary is Rs {salary}. {question}",
                    "variables": {
                        "job": ["factory worker", "IT employee", "domestic helper", "delivery driver", "teacher in a private school"],
                        "duration": ["6 months", "2 years", "5 years"],
                        "issue": ["fired me without any notice", "did not pay my last two months' salary", "is sexually harassing me at work", "makes me work overtime with no extra pay", "is discriminating against me"],
                        "salary": [str(random.randint(5000, 50000))],
                        "question": ["Am I eligible for legal aid at the labor court?", "Can I get a free lawyer for my wage case?", "How do I file a case under the Minimum Wages Act?"]
                    },
                    "eligibility_logic": lambda vars: True if int(vars['salary']) < 25000 or 'harassing' in vars['issue'] or (int(re.search(r'\d+', vars['duration'])[0]) > 1 and 'fired' in vars['issue']) else False,
                    "expected_domains": ["employment_law", "legal_aid"],
                    "complexity": "medium"
                },
                # Workplace Safety
                {
                    "template": "The conditions at my workplace, a {workplace_type}, are very unsafe. {safety_issue}. I have complained to the manager but nothing has changed. What action can I take?",
                    "variables": {
                        "workplace_type": ["construction site", "chemical factory", "warehouse"],
                        "safety_issue": ["There is no safety equipment like helmets or gloves", "The machinery is old and faulty", "There is a constant risk of fire"],
                    },
                    "eligibility_logic": lambda vars: True,
                    "expected_domains": ["employment_law"],
                    "complexity": "high"
                },
                # Breach of Contract
                {
                    "template": "I signed an employment contract that promised {promised_term}, but my employer is not honoring it. They are {breach_action}. Can I sue for breach of contract?",
                    "variables": {
                        "promised_term": ["a specific salary", "a work-from-home option", "a promotion after one year"],
                        "breach_action": ["paying me less than agreed", "forcing me to come to the office every day", "denying my promotion without reason"],
                    },
                    "eligibility_logic": lambda vars: random.choice([True, False]),
                    "expected_domains": ["employment_law"],
                    "complexity": "medium"
                },
                # PF/Gratuity Issues
                {
                    "template": "I have resigned from my job after working for {years_worked} years, but the company is delaying my {payment_type}. How can I claim it?",
                    "variables": {
                        "years_worked": ["6", "10", "15"],
                        "payment_type": ["Provident Fund (PF) withdrawal", "gratuity payment"],
                    },
                    "eligibility_logic": lambda vars: int(vars['years_worked']) >= 5,
                    "expected_domains": ["employment_law"],
                    "complexity": "medium"
                }
            ]
        }
    
    # def _initialize_enhanced_templates(self) -> Dict[str, List[Dict]]:
    #     """Initialize diverse legal query templates with improved realism and variations"""
    #     return {
    #         "legal_aid": [
    #             # Income-based: Balanced eligible/ineligible, varied language
    #             {
    #                 "template": "{greeting}, I am a {social_category} {occupation} earning Rs {income} {income_frequency}. {case_description}. {question} I stay in {location}.",
    #                 "variables": {
    #                     "greeting": ["Sir", "Madam", "Respected sir/madam", "Hello ji", "Namaste"],
    #                     "social_category": ["poor woman", "dalit woman", "tribal woman", "disabled person", "widow", "divorced woman", "female laborer", "SC man", "ST farmer", "OBC artisan", "general category senior", "EWS youth"],
    #                     "occupation": ["daily wage worker", "farmer", "housewife", "street vendor", "construction laborer", "domestic helper", "rickshaw driver", "small shop owner"],
    #                     "income": [str(random.randint(0, 1000000)) for _ in range(50)],  # Wide range for balance
    #                     "income_frequency": ["monthly", "per year", "annually", "per month", "daily but total yearly"],
    #                     "location": ["village in UP", "slum in Mumbai", "remote area in Jharkhand", "tehsil in Rajasthan", "district in Bihar", "city in Delhi", "tribal area in Odisha"],
    #                     "case_description": [
    #                         "Husband abandoned me and kids no support", 
    #                         "Landlord evicting without notice wrongfully",
    #                         "Police beat me no reason at all",
    #                         "Doctor did negligence in treatment now demanding extra money",
    #                         "Builder took advance left house half-built",
    #                         "Sarpanch demanding bribe for scheme",
    #                         "Neighbor grabbed my land with fake papers",
    #                         "Boss fired me when pregnant",
    #                         "Got fake product online no refund",  # Cross-domain hint
    #                         "Job lost due to caste discrimination",  # Cross-domain
    #                         "Child custody battle with ex-husband",
    #                         "Consumer court case for defective fridge",
    #                         "Fundamental right violated by police custody"
    #                     ],
    #                     "question": ["Am I eligible for free legal help?", "Can I get legal aid?", "How to get pro bono lawyer?", "Is free lawyer possible for me?", "Legal services free for poor like me?"]
    #                 },
    #                 "eligibility_logic": lambda vars: int(vars['income']) <= 300000 if 'sc' in vars['social_category'].lower() or 'st' in vars['social_category'].lower() else int(vars['income']) <= 100000,  # Balanced thresholds
    #                 "expected_domains": ["legal_aid"],
    #                 "complexity": random.choice(["low", "medium"])
    #             },
    #             # Vulnerable group focused
    #             {
    #                 "template": "I am {vulnerable_group} from {location}. {case_description}. {eligibility_question}",
    #                 "variables": {
    #                     "vulnerable_group": ["senior citizen 70 years", "disabled with 50% handicap", "minor child 15 years", "woman victim of violence", "transgender person", "HIV patient", "mental health patient"],
    #                     "location": ["rural area", "urban slum", "hill station", "coastal village"],
    #                     "case_description": ["Facing property dispute with relatives", "Denied pension by government", "School harassment case", "Domestic abuse by in-laws", "Discrimination at workplace", "Medical negligence in hospital"],
    #                     "eligibility_question": ["Do I qualify for legal aid?", "Free legal support available?", "How to apply for legal services?"]
    #                 },
    #                 "eligibility_logic": lambda vars: True if any(kw in vars['vulnerable_group'] for kw in ["senior", "disabled", "minor", "woman victim"]) else random.choice([True, False]),  # Bias toward eligible but balanced
    #                 "expected_domains": ["legal_aid"],
    #                 "complexity": "medium"
    #             },
    #             # More templates for diversity...
    #             # Add 3-5 more per domain with similar structure
    #         ],
    #         "family_law": [
    #             # Divorce/maintenance: Balanced outcomes
    #             {
    #                 "template": "{informal_start} I am {gender} {marital_status}, husband/wife {behavior}. {income_info}. {child_info}. {question}",
    #                 "variables": {
    #                     "informal_start": ["Bhai sahab", "Didi", "Please help", "My problem is"],
    #                     "gender": ["woman", "man", "wife", "husband"],
    #                     "marital_status": ["married since 5 years", "separated", "divorced but alimony pending"],
    #                     "behavior": ["beats me daily", "left home 2 years ago", "has affair", "doesn't give money for house", "good but no support", "threatens to kill"],
    #                     "income_info": ["I earn Rs {income} but he earns more", "No job, dependent on him", "Both working but he hides income"],
    #                     "child_info": ["We have {num_children} kids", "One daughter 10 years old", "No children"],
    #                     "question": ["Can I get divorce and maintenance?", "Eligible for legal aid in family court?", "How to file custody case free?"]
    #                 },
    #                 "eligibility_logic": lambda vars: True if 'woman' in vars['gender'] or int(vars.get('income', 0)) < 20000 or 'beats' in vars['behavior'] else False,
    #                 "expected_domains": ["family_law"],
    #                 "complexity": random.choice(["medium", "high"])
    #             },
    #             # Add more for custody, adoption, etc.
    #         ],
    #         "consumer_protection": [
    #             # Complaints: Balanced valid/invalid
    #             {
    #                 "template": "Bought {product} for Rs {amount} from {seller}. {issue}. {time_since}. {question}",
    #                 "variables": {
    #                     "product": ["mobile phone", "fridge", "car", "online course", "medicine", "grocery"],
    #                     "amount": [str(random.randint(500, 1000000))],
    #                     "seller": ["Amazon", "local shop", "Flipkart", "dealer", "pharmacy"],
    #                     "issue": ["defective from day 1", "not as advertised", "expired product", "no delivery", "overcharged", "fake item"],
    #                     "time_since": ["1 month ago", "2 years back", "just yesterday", "6 months"],
    #                     "question": ["Can I file consumer case? Legal aid possible?", "Eligible for free lawyer in consumer forum?", "How to get compensation without paying fees?"]
    #                 },
    #                 "eligibility_logic": lambda vars: True if int(vars['amount']) < 2000000 and 'defective' in vars['issue'] and int(re.search(r'\d+', vars['time_since'])[0]) <= 24 else False,  # 2-year limit
    #                 "expected_domains": ["consumer_protection"],
    #                 "complexity": "low"
    #             },
    #             # Add more for services, unfair practices.
    #         ],
    #         "fundamental_rights": [
    #             # Rights violations: Balanced
    #             {
    #                 "template": "Police {action} without reason. I am {category}. {details}. {question}",
    #                 "variables": {
    #                     "action": ["arrested me illegally", "beat me in custody", "denied my vote", "discriminated based on caste"],
    #                     "category": ["Muslim", "Christian", "Dalit", "woman", "journalist"],
    #                     "details": ["No FIR shown", "Tortured for confession", "Speech censored", "Property seized unfairly"],
    #                     "question": ["Is this violation of Article {article}? Legal aid for PIL?", "Can I get free lawyer for human rights case?"]
    #                 },
    #                 "eligibility_logic": lambda vars: True if any(kw in vars['action'] + vars['details'] for kw in ["arrested illegally", "tortured", "censored"]) else False,
    #                 "expected_domains": ["fundamental_rights"],
    #                 "complexity": "high"
    #             },
    #             # Add more for equality, freedom, etc.
    #         ],
    #         "employment_law": [
    #             # Labor issues: Balanced
    #             {
    #                 "template": "Working as {job} for {duration}. Boss {issue}. Salary Rs {salary}. {question}",
    #                 "variables": {
    #                     "job": ["factory worker", "IT employee", "maid", "driver", "teacher"],
    #                     "duration": ["6 months", "2 years", "5 years"],
    #                     "issue": ["fired without notice", "no PF deduction", "harassment at work", "overtime no pay", "discrimination"],
    #                     "salary": [str(random.randint(5000, 50000))],
    #                     "question": ["Eligible for labor court legal aid?", "Can I get free lawyer for wage case?", "How to file under Minimum Wages Act?"]
    #                 },
    #                 "eligibility_logic": lambda vars: True if int(vars['salary']) < 25000 or 'harassment' in vars['issue'] or int(re.search(r'\d+', vars['duration'])[0]) > 1 else False,
    #                 "expected_domains": ["employment_law"],
    #                 "complexity": "medium"
    #             },
    #             # Add more for contracts, safety.
    #         }
    #     }
    
    def _initialize_diverse_demographics(self) -> List[Dict]:
        """Diverse user profiles for realism"""
        return [
            {"age": random.randint(18, 80), "gender": random.choice(["male", "female", "other"]), "category": random.choice(["SC", "ST", "OBC", "General", "EWS"]),
             "location": random.choice(["Delhi", "Mumbai", "Rural UP", "Tribal MP", "Urban Karnataka"]), "income": random.randint(0, 1000000),
             "vulnerable": random.choice([True, False])}
            for _ in range(1000)  # Pre-generate for variety
        ]
    
    def _initialize_case_variations(self) -> Dict:
        """Variations for complexity and priority"""
        return {
            "complexity": ["low", "medium", "high", "very high"],
            "priority": ["normal", "urgent", "high", "critical"],
            "confidence_factors": lambda: {f"factor_{i}": round(random.uniform(0.5, 1.0), 2) for i in range(3)}
        }
    
    def _initialize_noise_patterns(self) -> List[callable]:
        """Functions to add realistic noise (typos, slang)"""
        def add_typos(text): return re.sub(r'([a-zA-Z])', lambda m: random.choice([m.group(0), m.group(0).upper() if random.random() < 0.05 else m.group(0)]), text)
        def add_slang(text): return text.replace("I am", random.choice(["Me is", "I m", "Im"])) if random.random() < 0.3 else text
        def incomplete_sentence(text): return text[:-random.randint(5,20)] + "..." if random.random() < 0.2 else text
        return [add_typos, add_slang, incomplete_sentence]
    
    def _apply_noise(self, query: str) -> str:
        """Apply random noise for robustness"""
        for noise_fn in random.sample(self.noise_patterns, k=random.randint(0, 2)):
            query = noise_fn(query)
        return query
    
    def _generate_single_example(self, domain: LegalDomain, template: Dict, eligible: bool = None) -> LegalTrainingExample:
        """Generate one example with optional forced eligibility"""
        vars = {k: random.choice(v) for k, v in template["variables"].items()}
        query = template["template"].format(**vars)
        query = self._apply_noise(query)
        
        # Force eligibility if specified
        eligibility = template["eligibility_logic"](vars) if eligible is None else eligible
        
        demo = random.choice(self.demographic_profiles)
        return LegalTrainingExample(
            query=query,
            domains=template["expected_domains"],
            extracted_facts=[f"{k}: {v}" for k, v in vars.items()],
            expected_eligibility=eligibility,
            legal_reasoning=f"Reasoning for {domain.value}: {random.choice(['Eligible under Section X', 'Not eligible due to Y'])}",
            confidence_factors=self.case_variations["confidence_factors"](),
            user_demographics=demo,
            case_complexity=random.choice(self.case_variations["complexity"]),
            priority_level=random.choice(self.case_variations["priority"])
        )
    
    def _generate_enhanced_domain_samples(self, domain: LegalDomain, count: int) -> List[LegalTrainingExample]:
        """Generate balanced samples per domain (50% eligible)"""
        templates = self.legal_templates[domain.value]
        samples = []
        for _ in range(count // 2):  # Eligible
            template = random.choice(templates)
            samples.append(self._generate_single_example(domain, template, eligible=True))
        for _ in range(count // 2):  # Ineligible
            template = random.choice(templates)
            samples.append(self._generate_single_example(domain, template, eligible=False))
        return samples
    
    def _generate_enhanced_cross_domain_samples(self, count: int) -> List[LegalTrainingExample]:
        """Generate realistic multi-domain samples"""
        domain_pairs = list(itertools.combinations(LegalDomain, 2))
        samples = []
        for _ in range(count):
            pair = random.choice(domain_pairs)
            templates = [random.choice(self.legal_templates[d.value]) for d in pair]
            combined_template = {**templates[0], **templates[1]}  # Merge for hybrid query
            combined_template["expected_domains"] = [d.value for d in pair]
            eligible = random.choice([True, False])
            sample = self._generate_single_example(pair[0], combined_template, eligible)
            sample.domains = combined_template["expected_domains"]
            sample.legal_reasoning += f" Cross-domain: {pair[1].value}"
            samples.append(sample)
        return samples
    
    def _generate_edge_case_samples(self, count: int) -> List[LegalTrainingExample]:
        """Generate borderline/ambiguous cases"""
        samples = []
        for _ in range(count):
            domain = random.choice(list(LegalDomain))
            template = random.choice(self.legal_templates[domain.value])
            # Modify vars for edge: e.g., income exactly at threshold
            vars = {k: random.choice(v) for k, v in template["variables"].items()}
            if 'income' in vars:
                vars['income'] = str(random.choice([100000, 300000, 500000, 800000]))  # Threshold edges
            query = template["template"].format(**vars)
            query = self._apply_noise(query + " But borderline case.")  # Add ambiguity
            eligible = random.random() < 0.5  # Random for edges
            sample = self._generate_single_example(domain, template, eligible)
            sample.case_complexity = "very high"
            samples.append(sample)
        return samples
    
    def generate_comprehensive_dataset(self, total_samples: int = 30000) -> Tuple[List, List, List]:
        """Generate and split dataset into train/val/test"""
        dataset = []
        
        # Domain distribution (balanced)
        domain_distribution = {d: int(total_samples * 0.18) for d in LegalDomain}  # ~5400 per domain
        domain_distribution[LegalDomain.LEGAL_AID] += total_samples - sum(domain_distribution.values())  # Remainder to core
        
        for domain, count in domain_distribution.items():
            logger.info(f"Generating {count} samples for {domain.value}")
            domain_samples = self._generate_enhanced_domain_samples(domain, count)
            dataset.extend(domain_samples)
        
        # Cross-domain (20%)
        cross_count = int(total_samples * 0.20)
        logger.info(f"Generating {cross_count} cross-domain samples")
        dataset.extend(self._generate_enhanced_cross_domain_samples(cross_count))
        
        # Edge cases (10%)
        edge_count = int(total_samples * 0.10)
        logger.info(f"Generating {edge_count} edge case samples")
        dataset.extend(self._generate_edge_case_samples(edge_count))
        
        random.shuffle(dataset)
        
        # Stratified split by eligibility
        train, temp = train_test_split(dataset, test_size=0.3, stratify=[s.expected_eligibility for s in dataset])
        val, test = train_test_split(temp, test_size=0.5, stratify=[s.expected_eligibility for s in temp])
        
        return [asdict(s) for s in train], [asdict(s) for s in val], [asdict(s) for s in test]
    
    def save_datasets(self, train: List, val: List, test: List):
        """Save splits to JSON"""
        Path("data").mkdir(exist_ok=True)
        with open("data/train_samples.json", 'w', encoding='utf-8') as f:
            json.dump(train, f, indent=2)
        with open("data/val_samples.json", 'w', encoding='utf-8') as f:
            json.dump(val, f, indent=2)
        with open("data/test_samples.json", 'w', encoding='utf-8') as f:
            json.dump(test, f, indent=2)

def main():
    logger.info("Starting enhanced legal data generation")
    generator = EnhancedLegalDataGenerator()
    train, val, test = generator.generate_comprehensive_dataset(total_samples=30000)
    generator.save_datasets(train, val, test)
    
    logger.info(f"Generated: Train {len(train)}, Val {len(val)}, Test {len(test)}")
    print(f"✅ Datasets saved to data/ folder")
    print(f"📊 Total samples: {len(train) + len(val) + len(test):,}")
    print("Use these for training: system.preprocess_data('data/') then system.train_complete_system('data/')")

if __name__ == "__main__":
    main()