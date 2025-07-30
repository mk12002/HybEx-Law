"""
Sample dataset for HybEx-Law system development and testing.

This module contains example queries and their expected extractions
for training and evaluating the NLP pipeline.
"""

from typing import List, Dict, Any

# Sample legal aid queries with annotations
SAMPLE_QUERIES = [
    {
        "id": 1,
        "query": "I lost my factory job last month and have no income right now. My landlord is trying to evict me from my house in Delhi. I am from the general category. Can I get a free government lawyer?",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 0)",
            "case_type(user, \"property_dispute\")",
            "is_woman(user, false)",
            "is_sc_st(user, false)",
            "is_industrial_workman(user, true)",
            "location(user, \"delhi\")"
        ],
        "expected_eligible": True,
        "reason": "Zero income and industrial worker status"
    },
    {
        "id": 2,
        "query": "I am a woman and my husband is demanding dowry. He has been beating me. I earn 30000 rupees per month. Please help me.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 30000)",
            "case_type(user, \"domestic_violence\")",
            "is_woman(user, true)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": True,
        "reason": "Woman applicant - categorical eligibility"
    },
    {
        "id": 3,
        "query": "I am a businessman and someone wrote false things about me in the newspaper. This has damaged my reputation. I want to file a defamation case. My annual income is 10 lakhs.",
        "expected_facts": [
            "applicant(user)",
            "income_annual(user, 1000000)",
            "case_type(user, \"defamation\")",
            "is_woman(user, false)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": False,
        "reason": "Defamation case excluded from legal aid"
    },
    {
        "id": 4,
        "query": "I am from scheduled caste community. I work in a shop and earn 15000 per month. My neighbor has occupied part of my land illegally.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 15000)",
            "case_type(user, \"property_dispute\")",
            "is_woman(user, false)",
            "is_sc_st(user, true)"
        ],
        "expected_eligible": True,
        "reason": "SC/ST categorical eligibility"
    },
    {
        "id": 5,
        "query": "I am 16 years old. My father died in an accident and the insurance company is not paying compensation. We have no income now.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 0)",
            "case_type(user, \"accident_compensation\")",
            "is_child(user, true)",
            "is_woman(user, false)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": True,
        "reason": "Minor - categorical eligibility"
    },
    {
        "id": 6,
        "query": "I run a small business. My partner is cheating me out of profits. I make about 50000 per month from the business.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 50000)",
            "case_type(user, \"business_dispute\")",
            "is_woman(user, false)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": False,
        "reason": "Business dispute excluded and income exceeds limit"
    },
    {
        "id": 7,
        "query": "I am unemployed for 6 months. My former employer has not paid my salary. I am a woman with two children.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 0)",
            "case_type(user, \"labor_dispute\")",
            "is_woman(user, true)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": True,
        "reason": "Woman applicant and zero income"
    },
    {
        "id": 8,
        "query": "I was arrested last week and I am in police custody. My family has no money for a lawyer. I am innocent.",
        "expected_facts": [
            "applicant(user)",
            "case_type(user, \"criminal_matter\")",
            "is_in_custody(user, true)",
            "is_woman(user, false)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": True,
        "reason": "Person in custody - categorical eligibility"
    },
    {
        "id": 9,
        "query": "I am disabled and use a wheelchair. The government office is not accessible. I want to file a case for my rights. I get disability pension of 2000 per month.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 2000)",
            "is_disabled(user, true)",
            "is_woman(user, false)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": True,
        "reason": "Disabled person - categorical eligibility"
    },
    {
        "id": 10,
        "query": "My house was destroyed in the recent floods. I need compensation from the government. I used to earn 20000 per month before the disaster.",
        "expected_facts": [
            "applicant(user)",
            "income_monthly(user, 20000)",
            "is_disaster_victim(user, true)",
            "case_type(user, \"accident_compensation\")",
            "is_woman(user, false)",
            "is_sc_st(user, false)"
        ],
        "expected_eligible": True,
        "reason": "Disaster victim - categorical eligibility"
    }
]

# Entity presence annotations for training Stage 1 classifier
ENTITY_ANNOTATIONS = [
    {
        "query": "I lost my factory job last month and have no income right now. My landlord is trying to evict me from my house in Delhi.",
        "entities": ["income", "case_type", "social_category"]
    },
    {
        "query": "I am a woman and my husband is demanding dowry. He has been beating me. I earn 30000 rupees per month.",
        "entities": ["income", "case_type", "social_category"]
    },
    {
        "query": "I am a businessman and someone wrote false things about me in the newspaper.",
        "entities": ["case_type"]
    },
    {
        "query": "I am from scheduled caste community. I work in a shop and earn 15000 per month.",
        "entities": ["income", "social_category", "case_type"]
    },
    {
        "query": "I am 16 years old. My father died in an accident and the insurance company is not paying compensation.",
        "entities": ["social_category", "case_type", "income"]
    },
    {
        "query": "I run a small business. My partner is cheating me out of profits.",
        "entities": ["case_type", "income"]
    },
    {
        "query": "I am unemployed for 6 months. My former employer has not paid my salary. I am a woman with two children.",
        "entities": ["income", "case_type", "social_category"]
    },
    {
        "query": "I was arrested last week and I am in police custody. My family has no money for a lawyer.",
        "entities": ["social_category", "case_type", "income"]
    },
    {
        "query": "I am disabled and use a wheelchair. The government office is not accessible.",
        "entities": ["social_category", "case_type", "income"]
    },
    {
        "query": "My house was destroyed in the recent floods. I need compensation from the government.",
        "entities": ["social_category", "case_type", "income"]
    }
]

# Case type classification training data
CASE_TYPE_ANNOTATIONS = [
    {"text": "landlord evicting me from house", "label": "property_dispute"},
    {"text": "husband demanding dowry beating me", "label": "family_matter"},
    {"text": "false things written in newspaper reputation damaged", "label": "defamation"},
    {"text": "neighbor occupied my land illegally", "label": "property_dispute"},
    {"text": "insurance company not paying accident compensation", "label": "accident_compensation"},
    {"text": "business partner cheating profits", "label": "business_dispute"},
    {"text": "employer not paid salary", "label": "labor_dispute"},
    {"text": "arrested police custody need lawyer", "label": "criminal_matter"},
    {"text": "government office not accessible disabled rights", "label": "consumer_dispute"},
    {"text": "house destroyed floods government compensation", "label": "accident_compensation"},
    {"text": "domestic violence husband wife", "label": "family_matter"},
    {"text": "factory worker fired unfairly", "label": "labor_dispute"},
    {"text": "defective product warranty refund", "label": "consumer_dispute"},
    {"text": "child custody divorce", "label": "family_matter"},
    {"text": "election fraud voting", "label": "election_offense"},
    {"text": "corporate merger business contract", "label": "business_dispute"}
]

# Data expansion functions
def expand_sample_data(multiplier: int = 5) -> List[Dict[str, Any]]:
    """
    Expand the sample data using variations and paraphrasing.
    
    Args:
        multiplier: How many variations to create for each sample
        
    Returns:
        Expanded dataset
    """
    from src.data_generation.legal_data_generator import LegalDataGenerator
    
    generator = LegalDataGenerator()
    expanded_data = []
    
    # Add original samples
    expanded_data.extend(SAMPLE_QUERIES)
    
    # Generate variations for each original sample
    for original in SAMPLE_QUERIES:
        for i in range(multiplier):
            # Create variation based on original case type
            case_type = _extract_case_type_from_facts(original['expected_facts'])
            if case_type:
                variation = generator.generate_query(case_type)
                variation['id'] = len(expanded_data) + 1
                variation['source'] = f"variation_of_{original['id']}"
                expanded_data.append(variation)
    
    return expanded_data

def _extract_case_type_from_facts(facts: List[str]) -> str:
    """Extract case type from expected facts."""
    for fact in facts:
        if 'case_type(user,' in fact:
            # Extract case type from fact like 'case_type(user, "property_dispute")'
            case_type = fact.split('"')[1]
            return case_type
    return None

def create_training_splits(data: List[Dict[str, Any]], 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create train/validation/test splits from data.
    
    Args:
        data: Complete dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Dictionary with train/val/test splits
    """
    import random
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    splits = {
        'train': shuffled_data[:train_size],
        'validation': shuffled_data[train_size:train_size + val_size],
        'test': shuffled_data[train_size + val_size:]
    }
    
    return splits

def save_expanded_data(data: List[Dict[str, Any]], base_filename: str = "expanded_legal_data"):
    """Save expanded data to files."""
    import json
    from pathlib import Path
    
    # Create data directory
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete dataset
    with open(data_dir / f"{base_filename}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Create and save splits
    splits = create_training_splits(data)
    
    for split_name, split_data in splits.items():
        filename = data_dir / f"{base_filename}_{split_name}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(split_data)} samples to {filename}")
    
    return splits

# Additional realistic scenarios for specific case types
ADDITIONAL_SCENARIOS = {
    "property_disputes": [
        "Landlord increased rent illegally and threatening eviction",
        "Property dealer cheated me and sold same flat to multiple buyers", 
        "Society not allowing me to sell my apartment",
        "Neighbor built wall on my property boundary",
        "Tenant not paying rent for 8 months and refusing to vacate"
    ],
    
    "family_matters": [
        "Husband left me and children without any support",
        "Mother-in-law demanding dowry even after 10 years of marriage",
        "Ex-husband not allowing me to meet my children",
        "Wife filed false domestic violence case against me",
        "Getting divorced but husband hiding his property and income"
    ],
    
    "labor_disputes": [
        "Company terminated me during maternity leave",
        "Boss making me work overtime without payment",
        "Factory not providing protective equipment causing health issues",
        "Employer deducting salary for no reason",
        "Contract worker not getting same benefits as permanent employees"
    ],
    
    "criminal_matters": [
        "Police arrested me without proper warrant or evidence",
        "Filed complaint against harassment but police not taking action",
        "Falsely accused of theft by employer",
        "Victim of domestic violence but husband has political connections",
        "Witnesses threatening me to withdraw criminal case"
    ],
    
    "accident_compensation": [
        "Bus accident left me disabled but transport corporation denying compensation",
        "Doctor negligence during surgery caused permanent damage",
        "Factory accident injured me but company blaming my carelessness",
        "Road accident but other party insurance company not settling claim",
        "Government hospital wrong treatment led to death of my father"
    ]
}

def get_realistic_income_scenarios():
    """Get realistic income scenarios for different demographics."""
    return {
        "unemployed": [
            "lost job during COVID lockdown",
            "company shut down without notice",
            "laid off due to automation",
            "couldn't find work after graduation",
            "business failed due to competition"
        ],
        "low_income": [
            "work as domestic help earning 8000 per month",
            "auto rickshaw driver making 12000 monthly",
            "small shop owner with 15000 monthly income",
            "security guard with 10000 salary",
            "construction worker earning 400 per day"
        ],
        "moderate_income": [
            "government clerk with 25000 salary",
            "private school teacher earning 22000",
            "bank cashier with 28000 monthly income",
            "small business owner making 30000 per month",
            "skilled technician earning 26000"
        ],
        "above_threshold": [
            "software engineer earning 60000 per month",
            "doctor with private practice making 80000",
            "business executive with 75000 salary",
            "chartered accountant earning 90000 monthly",
            "consultant with 1 lakh monthly income"
        ]
    }

def get_social_category_variations():
    """Get variations for expressing social categories."""
    return {
        "woman": [
            "I am a woman", "I am female", "I am a lady", "I am a wife and mother",
            "As a woman", "Being female", "I am a working woman"
        ],
        "sc_st": [
            "I belong to scheduled caste", "I am from SC community", 
            "I am from scheduled tribe", "I belong to ST category",
            "I am a dalit", "I come from tribal community"
        ],
        "child": [
            "I am 16 years old", "I am a minor", "I am under 18",
            "I am 17 years old", "I am still a student", "I am underage"
        ],
        "disabled": [
            "I am disabled", "I use wheelchair", "I am visually impaired",
            "I have physical disability", "I am hearing impaired", "I am handicapped"
        ],
        "in_custody": [
            "I am in police custody", "I was arrested", "I am in jail",
            "I am detained by police", "I am behind bars", "I am in prison"
        ]
    }
