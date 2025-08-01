"""
English-only Legal Data Generation Script
Generates realistic Indian legal scenarios in English for training hybrid neural-symbolic system
"""

import json
import random
import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDomain(Enum):
    LEGAL_AID = "legal_aid"
    FAMILY_LAW = "family_law"
    CONSUMER_PROTECTION = "consumer_protection"
    EMPLOYMENT_LAW = "employment_law"
    FUNDAMENTAL_RIGHTS = "fundamental_rights"

@dataclass
class LegalTrainingExample:
    query: str
    domains: List[str]
    extracted_facts: List[str]
    expected_eligibility: bool
    legal_reasoning: str
    confidence_factors: Dict[str, float]
    user_demographics: Dict[str, Any]
    case_complexity: str
    priority_level: str

class EnglishLegalDataGenerator:
    """Generate comprehensive legal training data in English with Indian context"""
    
    def __init__(self):
        self.legal_templates = self._initialize_english_templates()
        self.demographic_profiles = self._initialize_demographic_profiles()
        
    def _initialize_english_templates(self) -> Dict[str, List[Dict]]:
        """Initialize English-only legal templates with Indian legal context"""
        return {
            "legal_aid": [
                {
                    "template": "I am a {social_status} earning Rs {income} monthly. {case_situation}. Can I get free legal aid under Legal Services Authorities Act?",
                    "variables": {
                        "social_status": ["poor woman", "widow", "single mother", "disabled person", "SC person", "ST person", "transgender person"],
                        "income": ["0", "5000", "8000", "12000", "15000", "18000", "20000", "25000"],
                        "case_situation": [
                            "My husband left me and children without support",
                            "Landlord is evicting me without proper notice",
                            "Police harassed me without reason",
                            "Hospital did wrong treatment and demanding money",
                            "Contractor took advance and left work incomplete",
                            "Government officials demanding bribes",
                            "Company fired me during pregnancy"
                        ]
                    },
                    "complexity": "medium"
                },
                {
                    "template": "Sir, I am a {profession} from {community} community earning {income} rupees monthly. {problem}. Am I eligible for legal aid?",
                    "variables": {
                        "profession": ["daily wage worker", "domestic worker", "security guard", "rickshaw driver", "construction worker"],
                        "community": ["SC", "ST", "OBC", "minority"],
                        "income": ["5000", "8000", "10000", "12000", "15000", "18000"],
                        "problem": [
                            "Boss has not paid salary for 3 months",
                            "Had accident and insurance not giving claim",
                            "Police trapped me in false case",
                            "Property dealer did fraud with my money",
                            "Hospital negligence caused more problems"
                        ]
                    },
                    "complexity": "medium"
                }
            ],
            
            "family_law": [
                {
                    "template": "I got married {years} ago. {marital_problem}. What should I do for {remedy}? {additional_info}",
                    "variables": {
                        "years": ["2 years", "5 years", "8 years", "12 years", "15 years"],
                        "marital_problem": [
                            "Husband does physical violence and demands dowry",
                            "Husband has extramarital affair and doesn't come home",
                            "In-laws torture me mentally and don't give food",
                            "Husband threw me out without any fault",
                            "Husband drinks and only beats me",
                            "Husband married second wife without divorcing me"
                        ],
                        "remedy": ["divorce", "maintenance", "protection order", "custody"],
                        "additional_info": [
                            "I have 2 small children and am housewife",
                            "I work but salary is low",
                            "Police complaint filed but no action taken",
                            "Parents supporting but have financial problems"
                        ]
                    },
                    "complexity": "high"
                },
                {
                    "template": "I am Muslim woman married for {duration}. {muslim_issue}. {rights_query} under Muslim Personal Law?",
                    "variables": {
                        "duration": ["3 years", "7 years", "12 years"],
                        "muslim_issue": [
                            "Husband gave triple talaq over phone and threw me out",
                            "Husband married second wife without my consent",
                            "I want khula because husband is abusive",
                            "After husband's death, his family not giving inheritance"
                        ],
                        "rights_query": [
                            "What are my maintenance rights",
                            "Can I get custody of children",
                            "Is this triple talaq valid after 2019 law",
                            "What is my property share"
                        ]
                    },
                    "complexity": "very_high"
                }
            ],
            
            "consumer_protection": [
                {
                    "template": "I bought {product} from {platform} for Rs {amount}. {problem}. {query} under Consumer Protection Act?",
                    "variables": {
                        "product": ["mobile phone", "laptop", "washing machine", "TV", "furniture"],
                        "platform": ["Amazon", "Flipkart", "local store", "online platform"],
                        "amount": ["15000", "25000", "50000", "75000", "100000"],
                        "problem": [
                            "Completely different product delivered",
                            "Product came damaged and no replacement",
                            "Fake product sent at original price",
                            "No delivery but payment deducted",
                            "Product faulty but no service"
                        ],
                        "query": [
                            "Can I file complaint in consumer court",
                            "How to get refund with compensation",
                            "Can I approach district forum",
                            "Do I need lawyer or can file myself"
                        ]
                    },
                    "complexity": "medium"
                },
                {
                    "template": "I booked {property} from builder paying Rs {amount} advance. {rera_issue}. {remedy} under RERA?",
                    "variables": {
                        "property": ["2 BHK flat", "3 BHK apartment", "villa", "plot"],
                        "amount": ["5 lakh", "10 lakh", "25 lakh", "50 lakh"],
                        "rera_issue": [
                            "Project delayed by 3 years without intimation",
                            "Construction quality very poor with defects",
                            "Builder demanding extra charges not in agreement",
                            "Project stopped and builder not responding"
                        ],
                        "remedy": [
                            "Can I get full refund with interest",
                            "How to file RERA complaint",
                            "Can I claim delay compensation",
                            "How to recover invested amount"
                        ]
                    },
                    "complexity": "very_high"
                }
            ],
            
            "employment_law": [
                {
                    "template": "I am {employee_type} working for {duration}. {employment_issue}. {legal_query} under labor laws?",
                    "variables": {
                        "employee_type": ["software engineer", "factory worker", "office employee", "contract worker"],
                        "duration": ["2 years", "5 years", "8 years", "10 years"],
                        "employment_issue": [
                            "Company terminated me during maternity leave",
                            "Boss is sexually harassing me",
                            "Salary not paid for 3 months",
                            "Forced to work overtime without pay",
                            "Terminated without notice period",
                            "Facing caste discrimination at workplace"
                        ],
                        "legal_query": [
                            "Can I file wrongful termination case",
                            "How to claim unpaid salary",
                            "What are my maternity rights",
                            "Can I file sexual harassment complaint",
                            "How to prove workplace discrimination"
                        ]
                    },
                    "complexity": "high"
                }
            ],
            
            "fundamental_rights": [
                {
                    "template": "I faced {rights_violation} by {authority}. {incident_details}. {remedy_sought} under Constitution?",
                    "variables": {
                        "rights_violation": ["caste discrimination", "police harassment", "denial of services", "freedom restriction"],
                        "authority": ["police", "government office", "public institution", "local administration"],
                        "incident_details": [
                            "They refused service because of my caste",
                            "Police detained without warrant or reason",
                            "Government office demanding bribe for legal work",
                            "Denied admission despite meeting criteria",
                            "Beaten by police during peaceful protest"
                        ],
                        "remedy_sought": [
                            "Can I file case under Article 14",
                            "How to complain about police excess",
                            "Can I file RTI for transparency",
                            "What remedy under SC/ST Act",
                            "How to approach Human Rights Commission"
                        ]
                    },
                    "complexity": "very_high"
                }
            ]
        }
    
    def _initialize_demographic_profiles(self) -> List[Dict]:
        """Initialize diverse demographic profiles"""
        return [
            {"age": 25, "gender": "female", "location": "urban", "education": "graduate", "income_level": "low"},
            {"age": 35, "gender": "male", "location": "rural", "education": "high_school", "income_level": "very_low"},
            {"age": 45, "gender": "female", "location": "semi_urban", "education": "primary", "income_level": "medium"},
            {"age": 55, "gender": "male", "location": "urban", "education": "graduate", "income_level": "high"},
            {"age": 30, "gender": "transgender", "location": "urban", "education": "intermediate", "income_level": "low"},
            {"age": 40, "gender": "female", "location": "rural", "education": "illiterate", "income_level": "very_low"},
        ]
    
    def generate_dataset(self, total_samples: int = 5000) -> List[LegalTrainingExample]:
        """Generate comprehensive training dataset"""
        logger.info(f"Generating English dataset with {total_samples} samples")
        
        dataset = []
        
        # Domain distribution
        domain_distribution = {
            LegalDomain.LEGAL_AID: int(total_samples * 0.30),
            LegalDomain.FAMILY_LAW: int(total_samples * 0.25),
            LegalDomain.CONSUMER_PROTECTION: int(total_samples * 0.20),
            LegalDomain.EMPLOYMENT_LAW: int(total_samples * 0.15),
            LegalDomain.FUNDAMENTAL_RIGHTS: int(total_samples * 0.10)
        }
        
        # Generate single-domain samples
        for domain, count in domain_distribution.items():
            domain_samples = self._generate_domain_samples(domain, count)
            dataset.extend(domain_samples)
        
        # Generate cross-domain samples (remaining samples)
        remaining = total_samples - len(dataset)
        if remaining > 0:
            cross_domain_samples = self._generate_cross_domain_samples(remaining)
            dataset.extend(cross_domain_samples)
        
        random.shuffle(dataset)
        logger.info(f"Dataset generation complete: {len(dataset)} samples")
        return dataset
    
    def _generate_domain_samples(self, domain: LegalDomain, count: int) -> List[LegalTrainingExample]:
        """Generate samples for specific domain"""
        samples = []
        templates = self.legal_templates.get(domain.value, [])
        
        for i in range(count):
            template_data = random.choice(templates)
            sample = self._create_sample_from_template(template_data, [domain.value])
            samples.append(sample)
        
        return samples
    
    def _generate_cross_domain_samples(self, count: int) -> List[LegalTrainingExample]:
        """Generate cross-domain samples with realistic Indian scenarios"""
        cross_domain_scenarios = [
            {
                "query": "I am poor woman earning Rs {income} monthly. My husband left me during pregnancy and company fired me. I need legal aid for divorce, maintenance and wrongful termination case. Can I get free legal help for all three cases?",
                "domains": ["legal_aid", "family_law", "employment_law"],
                "variables": {"income": ["8000", "12000", "15000", "18000"]},
                "complexity": "very_high"
            },
            {
                "query": "I am SC community person who booked flat for Rs {amount}. Builder delayed project and when I complained, he used caste slurs. Police also not helping due to discrimination. Can I get legal aid for RERA case and caste discrimination both?",
                "domains": ["legal_aid", "consumer_protection", "fundamental_rights"],
                "variables": {"amount": ["15 lakh", "25 lakh", "35 lakh"]},
                "complexity": "very_high"
            },
            {
                "query": "During my maternity leave, company terminated me saying cost-cutting. My husband filed divorce claiming I cannot earn now. The flat we bought has defects and builder not responding. I need to fight wrongful termination, contest divorce and get flat issues resolved. Can I get legal aid despite previous salary since now unemployed?",
                "domains": ["employment_law", "family_law", "consumer_protection", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            }
        ]
        
        samples = []
        for i in range(count):
            scenario = random.choice(cross_domain_scenarios)
            # Fill variables if any
            query = scenario["query"]
            for var_name, options in scenario.get("variables", {}).items():
                if f"{{{var_name}}}" in query:
                    query = query.replace(f"{{{var_name}}}", random.choice(options))
            
            # Create sample
            sample = LegalTrainingExample(
                query=query,
                domains=scenario["domains"],
                extracted_facts=self._extract_facts_from_query(query, scenario["domains"]),
                expected_eligibility=self._determine_eligibility(query, scenario["domains"]),
                legal_reasoning=self._generate_legal_reasoning(scenario["domains"], True),
                confidence_factors=self._calculate_confidence_factors(),
                user_demographics=random.choice(self.demographic_profiles),
                case_complexity=scenario.get("complexity", "high"),
                priority_level=self._determine_priority(query)
            )
            samples.append(sample)
        
        return samples
    
    def _create_sample_from_template(self, template_data: Dict, domains: List[str]) -> LegalTrainingExample:
        """Create training sample from template"""
        template = template_data["template"]
        variables = template_data.get("variables", {})
        
        # Fill template with random variables
        filled_query = template
        for var_name, options in variables.items():
            if f"{{{var_name}}}" in filled_query:
                filled_query = filled_query.replace(f"{{{var_name}}}", random.choice(options))
        
        # Extract facts and determine eligibility
        extracted_facts = self._extract_facts_from_query(filled_query, domains)
        eligibility = self._determine_eligibility(filled_query, domains)
        
        return LegalTrainingExample(
            query=filled_query,
            domains=domains,
            extracted_facts=extracted_facts,
            expected_eligibility=eligibility,
            legal_reasoning=self._generate_legal_reasoning(domains, eligibility),
            confidence_factors=self._calculate_confidence_factors(),
            user_demographics=random.choice(self.demographic_profiles),
            case_complexity=template_data.get("complexity", "medium"),
            priority_level=self._determine_priority(filled_query)
        )
    
    def _extract_facts_from_query(self, query: str, domains: List[str]) -> List[str]:
        """Extract structured facts from query"""
        facts = ["applicant(user)."]
        query_lower = query.lower()
        
        # Income extraction
        income_patterns = [
            r'(\d+)\s*rupees?\s*(?:monthly|per month)',
            r'earning\s*(?:rs\.?)?\s*(\d+)',
            r'income\s*(?:rs\.?)?\s*(\d+)'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, query_lower)
            if match:
                income = int(match.group(1))
                facts.append(f"income_monthly(user, {income}).")
                if income <= 25000:
                    facts.append("income_category(user, 'low').")
                break
        
        # Social categories
        social_categories = {
            "woman": "is_woman(user, true).",
            "sc": "social_category(user, 'sc').",
            "st": "social_category(user, 'st').",
            "disabled": "is_disabled(user, true).",
            "widow": "is_woman(user, true).",
            "single mother": "is_woman(user, true).",
            "transgender": "is_transgender(user, true)."
        }
        
        for category, fact in social_categories.items():
            if category in query_lower:
                facts.append(fact)
        
        # Domain-specific facts
        if "legal_aid" in domains:
            if any(word in query_lower for word in ["poor", "cannot afford", "free legal"]):
                facts.append("seeks_legal_aid(user, true).")
        
        if "family_law" in domains:
            if any(word in query_lower for word in ["husband", "wife", "married"]):
                facts.append("marital_status(user, married).")
            if any(word in query_lower for word in ["divorce", "separation"]):
                facts.append("seeks_divorce(user, true).")
        
        return facts
    
    def _determine_eligibility(self, query: str, domains: List[str]) -> bool:
        """Determine legal aid eligibility"""
        if "legal_aid" in domains:
            query_lower = query.lower()
            # Income-based eligibility
            income_match = re.search(r'(\d+)\s*rupees?\s*monthly', query_lower)
            if income_match and int(income_match.group(1)) <= 25000:
                return True
            # Categorical eligibility
            if any(word in query_lower for word in ["sc", "st", "woman", "disabled", "widow"]):
                return True
        return True  # Other domains generally eligible
    
    def _generate_legal_reasoning(self, domains: List[str], eligibility: bool) -> str:
        """Generate legal reasoning"""
        if "legal_aid" in domains and eligibility:
            return "Eligible for legal aid under LSA Act due to income criteria or categorical eligibility"
        return f"Valid legal matter under {', '.join(domains)} with grounds for legal action"
    
    def _calculate_confidence_factors(self) -> Dict[str, float]:
        """Calculate confidence factors"""
        return {
            "query_clarity": random.uniform(0.8, 0.95),
            "fact_extraction_confidence": random.uniform(0.85, 0.98),
            "legal_rule_certainty": random.uniform(0.9, 1.0)
        }
    
    def _determine_priority(self, query: str) -> str:
        """Determine case priority"""
        urgent_words = ["violence", "harassment", "emergency", "threat"]
        if any(word in query.lower() for word in urgent_words):
            return "urgent"
        return random.choice(["normal", "high"])
    
    def save_dataset(self, dataset: List[LegalTrainingExample], filepath: str):
        """Save dataset to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        dataset_json = {
            "metadata": {
                "total_samples": len(dataset),
                "domains": [domain.value for domain in LegalDomain],
                "language": "english",
                "generation_date": "2025-01-08",
                "version": "1.0"
            },
            "samples": [asdict(sample) for sample in dataset]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"English dataset saved to {filepath}")

def main():
    """Generate English legal training dataset"""
    generator = EnglishLegalDataGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(total_samples=5000)
    
    # Save dataset
    output_path = "../data/training/english_legal_dataset.json"
    generator.save_dataset(dataset, output_path)
    
    # Print statistics
    domain_counts = {}
    for sample in dataset:
        for domain in sample.domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Domain distribution: {domain_counts}")
    
    # Show sample
    print(f"\nSample query: {dataset[0].query}")
    print(f"Domains: {dataset[0].domains}")
    print(f"Facts: {dataset[0].extracted_facts[:3]}...")

if __name__ == "__main__":
    main()
