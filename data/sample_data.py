"""
Sample dataset for HybEx-Law system development and testing.

This module contains example queries and their expected extractions
for training and evaluating the NLP pipeline.
"""

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
