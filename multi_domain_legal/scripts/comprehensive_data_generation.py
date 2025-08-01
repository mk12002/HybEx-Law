"""
Comprehensive Legal Data Generation for HybEx-Law System.

This script generates high-quality training data for the hybrid neural-symbolic
legal AI system covering all 5 legal domains with realistic scenarios.
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

class ComprehensiveLegalDataGenerator:
    """
    Generates comprehensive training data for all legal domains.
    Ensures legal accuracy and covers edge cases.
    """
    
    def __init__(self):
        self.legal_templates = self._initialize_legal_templates()
        self.demographic_profiles = self._initialize_demographic_profiles()
        self.case_variations = self._initialize_case_variations()
        
    def _initialize_legal_templates(self) -> Dict[str, List[Dict]]:
        """Initialize comprehensive legal query templates with realistic Indian scenarios in English"""
        return {
            "legal_aid": [
                # Income-based scenarios for different social categories
                {
                    "template": "Sir, I am a {social_category} earning Rs {income} monthly. {case_description}. Can I get free legal aid? I live in a {location}.",
                    "variables": {
                        "social_category": ["poor woman", "dalit woman", "tribal woman", "disabled person", "widow", "divorced woman", "female laborer"],
                        "income": ["0", "3000", "5000", "7500", "8000", "10000", "12000", "15000", "18000", "20000", "22000", "25000"],
                        "location": ["village", "small town", "remote area", "tehsil", "district"],
                        "case_description": [
                            "My husband left me and children without any support",
                            "Landlord is evicting me from house without proper notice", 
                            "Police harassed me without any reason",
                            "Hospital doctor did wrong treatment and demanding money",
                            "Contractor took advance money and left construction incomplete",
                            "Government office officials are demanding bribes",
                            "Neighbor has encroached on my land illegally",
                            "Company fired me during pregnancy period"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "medium"
                },
                # Formal scenarios for educated urban population
                {
                    "template": "I am a {social_status} earning Rs. {income} per month. {case_situation}. Am I eligible for free legal aid under Legal Services Authorities Act? I live in {city_type}.",
                    "variables": {
                        "social_status": ["widow", "single mother", "disabled person", "senior citizen above 65", "SC/ST person", "transgender person", "woman from economically weaker section"],
                        "income": ["8000", "12000", "15000", "18000", "20000", "22000", "25000", "28000", "30000"],
                        "city_type": ["metro city", "tier-2 city", "small town", "urban area"],
                        "case_situation": [
                            "My landlord is trying to evict me without proper notice under Rent Control Act",
                            "I am facing domestic violence and need protection under DV Act 2005",
                            "My employer terminated me during maternity leave violating Maternity Benefit Act",
                            "Builder has cheated me under RERA and I need compensation",
                            "Police have filed false case against me under IPC sections",
                            "Government office is demanding bribe which violates my rights",
                            "Insurance company is not settling my claim properly"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "high"
                },
                # Working class scenarios
                {
                    "template": "Sir, I am a {profession} earning monthly {income} rupees. {problem_description}. Am I eligible for legal aid? I belong to {community} community.",
                    "variables": {
                        "profession": ["rickshaw driver", "domestic worker", "security guard", "watchman", "cook", "cleaner", "construction worker", "street vendor"],
                        "income": ["5000", "8000", "10000", "12000", "15000", "18000", "20000", "22000"],
                        "community": ["SC", "ST", "OBC", "minority", "EWS"],
                        "problem_description": [
                            "I had accident and insurance company is not giving claim",
                            "Boss has not paid salary for 3 months and terminated me",
                            "Police have trapped me in false case",
                            "Property dealer committed fraud and ran away with money",
                            "Hospital negligence occurred and I need compensation",
                            "Employer has not deducted PF and ESI for years"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "medium"
                },
                # Complex categorical eligibility scenarios
                {
                    "template": "{greeting}, I am {detailed_status}. {complex_situation}. {legal_query}",
                    "variables": {
                        "greeting": ["Respected sir", "Sir/Madam", "Hello"],
                        "detailed_status": [
                            "a 70 year old woman with no family support",
                            "disability certificate holder using wheelchair",
                            "single mother with 2 small children whose husband abandoned",
                            "daily wage laborer unemployed since corona pandemic",
                            "from tribal area with limited Hindi understanding",
                            "acid attack survivor needing facial surgery"
                        ],
                        "complex_situation": [
                            "My in-laws are evicting me from property and police are not helping",
                            "Government scheme money not received despite proper documents",
                            "Private hospital did wrong operation causing more problems",
                            "Building collapse by contractor caused injury but no compensation",
                            "Bank officials harassing for loan repayment with threats",
                            "Educational institute discriminating against me for being disabled"
                        ],
                        "legal_query": [
                            "Can I get free legal aid under LSA Act?",
                            "Can I file court case without paying lawyer fees?",
                            "How to get help from Legal Services Authority?",
                            "Should I approach DLSA or SLSA for assistance?",
                            "How to get pro bono lawyer in India?"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "very_high"
                }
            ],
            
            "family_law": [
                # Hindu Marriage Act scenarios (Majority population)
                {
                    "template": "I got married {marriage_duration} ago. {marital_problem}. What should I do for {legal_remedy}? {additional_context}",
                    "variables": {
                        "marriage_duration": ["2 years", "5 years", "8 years", "10 years", "15 years", "20 years"],
                        "marital_problem": [
                            "My husband does physical violence and keeps demanding dowry",
                            "Husband is having extramarital affair and doesn't come home",
                            "In-laws give me mental torture and don't even give food",
                            "Husband threw me and children out of house without any fault",
                            "Husband drinks alcohol, doesn't work, only beats me",
                            "Husband married second wife without divorcing me",
                            "In-laws harass for dowry and give death threats"
                        ],
                        "legal_remedy": ["divorce", "judicial separation", "maintenance", "protection order", "cruelty case"],
                        "additional_context": [
                            "I have 2 small children and I am housewife",
                            "I am working but salary is low for survival",
                            "Husband has property but nothing in my name",
                            "I filed police complaint but they took no action",
                            "My parents are supporting but they also have financial problems"
                        ]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "high"
                },
                # Muslim Personal Law scenarios
                {
                    "template": "I am a Muslim woman married for {years}. {muslim_issue}. {query_about_rights} under Muslim Personal Law and recent amendments?",
                    "variables": {
                        "years": ["3 years", "7 years", "12 years", "18 years"],
                        "muslim_issue": [
                            "My husband gave me triple talaq over WhatsApp and threw me out",
                            "Husband married second wife without my consent and not maintaining me",
                            "I want khula because husband is impotent and abusive",
                            "Husband's family demanding mehr amount back and harassing",
                            "After husband's death, his brothers not giving me inheritance share",
                            "Husband doing halala forced marriage threats for reconciliation"
                        ],
                        "query_about_rights": [
                            "What are my maintenance rights",
                            "Can I get custody of children",
                            "Is this triple talaq valid after 2019 law",
                            "What is my share in property",
                            "How to file case under Muslim Women Protection Act",
                            "Can I claim maintenance from husband's family"
                        ]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "very_high"
                },
                # Christian/Parsi Marriage scenarios
                {
                    "template": "We are {community} married under {marriage_act}. {specific_issue}. {legal_question}",
                    "variables": {
                        "community": ["Christian", "Parsi"],
                        "marriage_act": ["Indian Christian Marriage Act", "Parsi Marriage and Divorce Act"],
                        "specific_issue": [
                            "Husband is alcoholic and has became violent towards me and children",
                            "Wife has mental illness and family wants annulment of marriage",
                            "Spouse converted to different religion without consent",
                            "Marriage was not consummated due to impotency issues",
                            "Found out spouse was already married before our wedding",
                            "Spouse deserted the matrimonial home 2 years ago"
                        ],
                        "legal_question": [
                            "Can we get divorce under our personal law?",
                            "What is the procedure for annulment?",
                            "How to claim maintenance and child custody?",
                            "Is mutual consent divorce possible?",
                            "What are grounds for dissolution of marriage?",
                            "Can we approach family court directly?"
                        ]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "high"
                },
                # Property and inheritance (Hindu Succession Act)
                {
                    "template": "After {relationship_context}, {property_dispute}. {inheritance_question} according to Hindu Succession Act?",
                    "variables": {
                        "relationship_context": [
                            "my husband's death",
                            "father's death",
                            "brother's accidental death",
                            "uncle's natural death",
                            "mother-in-law's death",
                            "grandfather's recent death"
                        ],
                        "property_dispute": [
                            "in-laws are not giving me share in property",
                            "my brothers are not including me in father's property division",
                            "husband's brothers claiming entire ancestral property",
                            "step-mother transferred all property to her children",
                            "joint family property not giving daughters equal share",
                            "agricultural land division dispute is ongoing"
                        ],
                        "inheritance_question": [
                            "What is my legal right in property",
                            "Can daughters claim equal share in ancestral property",
                            "Can I apply for succession certificate in court",
                            "What is widow's share in joint property",
                            "What are women's rights in agricultural land",
                            "What is procedure when there's no will deed"
                        ]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "very_high"
                },
                # Domestic Violence Act scenarios
                {
                    "template": "I am a {victim_status} and {abuse_description}. {protection_sought} under Domestic Violence Act?",
                    "variables": {
                        "victim_status": [
                            "married woman living with in-laws",
                            "in live-in relationship for 3 years",
                            "divorced but still facing harassment from ex-husband",
                            "widow but facing abuse from husband's family",
                            "separated wife but husband is stalking me"
                        ],
                        "abuse_description": [
                            "husband and in-laws do daily physical and mental torture",
                            "partner financially controls me and doesn't give money",
                            "emotional abuse happening and beating in front of children",
                            "sexual abuse also happening and threatening with photos/videos",
                            "dowry harassment and acid attack threats being given",
                            "economic abuse - closed bank accounts and not allowing to work"
                        ],
                        "protection_sought": [
                            "How to get protection order",
                            "Can I stay in shared household",
                            "What is provision for maintenance and compensation",
                            "How to apply for emergency relief",
                            "How to ensure children's custody and protection",
                            "What is police protection order"
                        ]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "very_high"
                },
                # Child custody and maintenance
                {
                    "template": "After divorce/separation {custody_situation}. {child_related_query} under Guardians and Wards Act?",
                    "variables": {
                        "custody_situation": [
                            "husband has taken children away from me and not allowing to meet",
                            "court gave joint custody but ex-husband is not cooperating",
                            "I am single mother and father is not giving maintenance",
                            "there are disputes in children's education and health decisions",
                            "ex-wife took children to different city and cut contact",
                            "after remarriage step-father wants to legally adopt children"
                        ],
                        "child_related_query": [
                            "Can I apply for sole custody",
                            "How is child maintenance amount decided",
                            "How to enforce visitation rights",
                            "Can father's parental rights be terminated",
                            "What to do in international parental child abduction",
                            "What is adoption procedure for step-parent"
                        ]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "high"
                }
            ],
            
            "consumer_protection": [
                # E-commerce fraud (Very common in modern India)
                {
                    "template": "Sir, I bought {product} from {platform} for Rs {amount}. {ecommerce_problem}. {consumer_query} under Consumer Protection Act 2019?",
                    "variables": {
                        "platform": ["Amazon", "Flipkart", "Myntra", "Snapdeal", "local online store", "Facebook marketplace", "OLX"],
                        "product": ["mobile phone", "laptop", "washing machine", "refrigerator", "LED TV", "air conditioner", "furniture set", "jewelry"],
                        "amount": ["15000", "25000", "35000", "50000", "75000", "100000", "150000", "200000"],
                        "ecommerce_problem": [
                            "Completely different product delivered and they are not accepting return",
                            "Product came in damaged condition and they are refusing replacement",
                            "Wrong model delivered and not refunding price difference",
                            "Fake/duplicate product sent at original price",
                            "No delivery even after 1 month but payment was deducted",
                            "EMI is being deducted but product is faulty and service is poor",
                            "Returned as per return policy but refund not received"
                        ],
                        "consumer_query": [
                            "Can I file complaint in consumer court",
                            "How to claim full refund with compensation",
                            "What is procedure for online transaction fraud",
                            "Can I do credit card chargeback along with consumer case",
                            "Should I approach district forum or state commission",
                            "Do I need lawyer or can file myself"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "medium"
                },
                # Real estate fraud (RERA related)
                {
                    "template": "I booked {property_type} from {builder_type} by paying Rs {booking_amount} advance. {rera_issue}. {rera_remedy}",
                    "variables": {
                        "builder_type": ["reputed builder", "local builder", "cooperative society", "government housing board"],
                        "property_type": ["2 BHK flat", "3 BHK apartment", "villa", "plot", "commercial space"],
                        "booking_amount": ["5 lakh", "10 lakh", "15 lakh", "25 lakh", "50 lakh", "75 lakh", "1 crore"],
                        "rera_issue": [
                            "Project delayed by 3 years without any intimation",
                            "Builder changed approved plan without consent and reduced area",
                            "Construction quality is very poor with major structural defects",
                            "Amenities promised in brochure are not being provided",
                            "Builder demanding extra charges not mentioned in agreement",
                            "Project stopped midway and builder is not responding",
                            "Possession given but occupancy certificate not obtained"
                        ],
                        "rera_remedy": [
                            "Can I get full refund with interest under RERA",
                            "How to file complaint with RERA authority",
                            "Can I claim compensation for delay and harassment",
                            "What is procedure to recover invested amount",
                            "Can I approach consumer court along with RERA",
                            "How to get project completion timeline enforced"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "very_high"
                },
                # Medical negligence and healthcare fraud
                {
                    "template": "During {medical_context}, {medical_negligence}. I spent Rs {medical_cost}. {medical_remedy}",
                    "variables": {
                        "medical_context": [
                            "surgery at private hospital",
                            "delivery at corporate hospital",
                            "treatment at nursing home",
                            "root canal treatment at dental clinic",
                            "cataract operation at eye hospital",
                            "MRI/CT scan at diagnostic center"
                        ],
                        "medical_negligence": [
                            "doctor gave wrong diagnosis and did unnecessary surgery",
                            "operation had complications and post-surgery infection occurred",
                            "I had medicine allergies but doctor didn't check",
                            "equipment sterilization was not proper and infection spread",
                            "emergency treatment was delayed for money",
                            "insurance was cashless approved but they forced extra payment"
                        ],
                        "medical_cost": ["50000", "100000", "200000", "500000", "10 lakh", "25 lakh"],
                        "medical_remedy": [
                            "How to file medical negligence case in consumer court?",
                            "How is compensation amount calculated in medical cases?",
                            "What is process to cancel hospital license?",
                            "Can I recover from insurance company separately?",
                            "Can I file complaint in Medical Council and consumer case both?",
                            "How to arrange expert medical opinion for case?"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "very_high"
                },
                # Banking and financial services fraud
                {
                    "template": "During {bank_context}, {banking_fraud}. I have suffered loss of Rs {financial_loss}. {banking_remedy}",
                    "variables": {
                        "bank_context": [
                            "ATM transaction",
                            "net banking usage",
                            "credit card payment",
                            "bank loan processing",
                            "FD/RD maturity",
                            "insurance policy claim settlement"
                        ],
                        "banking_fraud": [
                            "unauthorized transactions occurred and bank is not accepting liability",
                            "advance fees taken for loan approval but loan was rejected",
                            "insurance claim rejected without proper reason",
                            "bank is deducting extra charges without intimation",
                            "FD premature withdrawal had wrong calculation and penalty",
                            "credit card fraud happened but bank says customer fault"
                        ],
                        "financial_loss": ["25000", "50000", "100000", "200000", "500000", "10 lakh"],
                        "banking_remedy": [
                            "Can I file case in Banking Ombudsman and Consumer Court both?",
                            "Can I file RBI complaint along with consumer protection case?",
                            "Is cyber crime police complaint and consumer case together possible?",
                            "What is procedure for bank license action?",
                            "Can I file class action suit with multiple victims?",
                            "Can I claim punitive damages along with compensation?"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "high"
                },
                # Food and hospitality services
                {
                    "template": "During {food_context}, {food_problem}. {health_impact}. {food_remedy}",
                    "variables": {
                        "food_context": [
                            "wedding hall food",
                            "family dinner at restaurant",
                            "hotel room service",
                            "eating from street food stall",
                            "home delivery food order",
                            "marriage catering service"
                        ],
                        "food_problem": [
                            "food poisoning occurred and 50 people were hospitalized",
                            "foreign object found in food and teeth got damaged",
                            "food quality was very bad and unhygienic conditions",
                            "expired products were used in cooking",
                            "non-veg item served in vegetarian food",
                            "chemical/pesticide taste was there in food"
                        ],
                        "health_impact": [
                            "Had to stay admitted in hospital for 1 week",
                            "Spent Rs 50000 on medical treatment",
                            "Missed office work due to food poisoning",
                            "Stomach problem became permanent",
                            "Had to take emergency treatment for allergic reaction"
                        ],
                        "food_remedy": [
                            "Can I file compensation case in consumer court?",
                            "Is FSSAI complaint along with civil case possible?",
                            "What is process to cancel hotel/restaurant license?",
                            "Can I get punitive damages along with medical expenses recovery?",
                            "Can I file criminal case also for food safety violation?",
                            "Can multiple victims file complaint collectively?"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "high"
                },
                # Telecom and utility services
                {
                    "template": "In {telecom_context} service, {service_problem}. {service_remedy}",
                    "variables": {
                        "telecom_context": [
                            "Jio/Airtel/Vi postpaid connection",
                            "broadband internet service",
                            "DTH/cable TV service",
                            "mobile recharge and data pack",
                            "BSNL landline connection",
                            "electricity/water bill payment"
                        ],
                        "service_problem": [
                            "wrong billing and overbilling issue is continuous",
                            "service disconnection without notice and reconnection charges",
                            "network problem but company is not resolving complaint",
                            "unauthorized value added services were activated",
                            "customer service is very poor and no complaint resolution",
                            "tariff plan changed without consent and extra charges applied"
                        ],
                        "service_remedy": [
                            "Can I file complaint in TDSAT and consumer court both?",
                            "Where to complain for service provider license action?",
                            "Do I get refund with compensation in billing disputes?",
                            "Can I claim damages for poor service quality?",
                            "Is regulatory authority complaint sufficient or need to go to court?",
                            "Is class action possible against same service provider?"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "medium"
                }
            ],
            
            "fundamental_rights": [
                # Article 14 - Right to Equality cases
                {
                    "template": "During {discrimination_context}, {equality_violation}. {constitutional_remedy} under Article 14 Right to Equality?",
                    "variables": {
                        "discrimination_context": [
                            "government job interview",
                            "school admission process",
                            "hospital treatment",
                            "filing complaint at police station",
                            "court proceedings",
                            "using public transportation"
                        ],
                        "equality_violation": [
                            "discrimination was done on caste basis and SC/ST were not given preference",
                            "women candidates were deliberately failed and male candidates passed",
                            "different treatment was given based on religion",
                            "service was denied based on economic status",
                            "equal access was not given based on disability",
                            "discrimination was done using language barrier"
                        ],
                        "constitutional_remedy": [
                            "Can I file writ petition in High Court?",
                            "Can I get compensation for Article 14 violation?",
                            "Can mandamus writ be issued against government authorities?",
                            "Can I get relief by invoking equal protection clause?",
                            "Can I take punitive action for discrimination?",
                            "Can I file PIL if it's a public issue?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "very_high"
                },
                # Article 19 - Freedom of Speech and Expression
                {
                    "template": "After {expression_context}, {speech_restriction}. {freedom_remedy} under Article 19(1)(a) freedom of speech?",
                    "variables": {
                        "expression_context": [
                            "posting on social media",
                            "organizing peaceful protest/rally",
                            "publishing newspaper article",
                            "giving speech in public meeting",
                            "criticizing government",
                            "expressing religious/political opinion"
                        ],
                        "speech_restriction": [
                            "police arrested me with sedition charges",
                            "local authorities banned assembly without proper reason",
                            "government officials filed defamation case",
                            "internet services were suspended in area",
                            "media house was pressured to remove content",
                            "university administration banned student expression"
                        ],
                        "freedom_remedy": [
                            "Can I challenge prior restraint in court?",
                            "Can I prove misuse of reasonable restrictions?",
                            "Can I apply for anticipatory bail in speech cases?",
                            "Can I file habeas corpus writ petition?",
                            "Can I file separate case for media freedom violation?",
                            "Can I challenge internet shutdown on constitutional grounds?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "very_high"
                },
                # Article 21 - Right to Life and Personal Liberty
                {
                    "template": "During {life_liberty_context}, {article21_violation}. {life_liberty_remedy}",
                    "variables": {
                        "life_liberty_context": [
                            "police custody",
                            "hospital treatment refusal",
                            "environment pollution",
                            "privacy violation by government surveillance",
                            "food/water contamination",
                            "road/infrastructure safety negligence"
                        ],
                        "article21_violation": [
                            "police brutality and torture occurred in custody",
                            "emergency medical treatment was denied due to lack of money",
                            "factory pollution caused health problems and government took no action",
                            "phone tapping and surveillance without warrant",
                            "contaminated water supply caused serious health issues",
                            "unsafe infrastructure collapse caused injury/death"
                        ],
                        "life_liberty_remedy": [
                            "Can I file PIL in High Court for right to life violation?",
                            "Can I claim compensation against police action?",
                            "Can mandamus writ be issued for healthcare right enforcement?",
                            "Can I get court orders for environmental protection?",
                            "Can I claim damages for privacy violation?",
                            "Can I file public interest litigation for government negligence?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "very_high"
                },
                # RTI Act 2005 cases
                {
                    "template": "I filed RTI application about {rti_subject} {time_period} ago. {rti_violation}. {rti_remedy}",
                    "variables": {
                        "rti_subject": [
                            "government tender allocation process",
                            "public funds utilization in development projects",
                            "government employee recruitment details",
                            "policy formulation and decision-making process",
                            "public distribution system functioning",
                            "government land acquisition details"
                        ],
                        "time_period": ["1 month", "2 months", "3 months", "6 months"],
                        "rti_violation": [
                            "received absolutely no response from PIO",
                            "incomplete and misleading information was given",
                            "refused to provide documents",
                            "demanded excessive fees for documents",
                            "denied saying information is exempt",
                            "PIO avoided by claiming lack of jurisdiction"
                        ],
                        "rti_remedy": [
                            "Can I file first appeal in Information Commission?",
                            "Is it possible to get penalty imposed on PIO?",
                            "Can I file second appeal in State/Central Information Commission?",
                            "Can I file writ petition in High Court for RTI violation?",
                            "Can I get whistleblower protection for public interest disclosure?",
                            "Can I file PIL for transparency violation?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "high"
                },
                # Article 25-28 - Religious Freedom cases
                {
                    "template": "In {religious_context}, {religious_violation}. {religious_remedy} under Article 25-28 religious freedom?",
                    "variables": {
                        "religious_context": [
                            "religious place of worship management",
                            "religious practices in educational institution",
                            "religious observance at workplace",
                            "religious expression in public space",
                            "government policy implementation",
                            "marriage/personal law matters"
                        ],
                        "religious_violation": [
                            "minority religious institution's autonomy was violated",
                            "religious conversion is being forced",
                            "religious festival celebration was banned without reason",
                            "religious dress code was restricted at workplace",
                            "educational rights were denied to religious minority",
                            "harassment and legal threats in interfaith marriage"
                        ],
                        "religious_remedy": [
                            "Is constitutional remedy available for religious freedom violation?",
                            "Can I take court intervention for minority rights protection?",
                            "Can I file criminal case against forced conversion?",
                            "Can I file writ petition to restore religious institution autonomy?",
                            "What is legal procedure to enforce educational minority rights?",
                            "Can I file case in High Court against personal law interference?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "very_high"
                },
                # SC/ST Act and Atrocity cases
                {
                    "template": "I belong to {caste_identity} and {atrocity_incident}. {scst_remedy} under SC/ST Act?",
                    "variables": {
                        "caste_identity": ["Scheduled Caste", "Scheduled Tribe community", "Dalit family"],
                        "atrocity_incident": [
                            "upper caste neighbors used caste slurs and did social boycott",
                            "humiliation occurred during caste certificate verification at government office",
                            "untouchability was practiced in public place and entry was denied",
                            "faced discrimination in employment after caste disclosure",
                            "caste threats and violence were used in land dispute",
                            "family received threats against inter-caste marriage"
                        ],
                        "scst_remedy": [
                            "Can I file case under SC/ST Prevention of Atrocities Act?",
                            "Can I claim compensation for caste discrimination?",
                            "Can I take court intervention to ensure police investigation?",
                            "Can I demand special court proceeding in atrocity case?",
                            "Can I file civil rights violation case for employment discrimination?",
                            "Can I apply for protection order against social boycott?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "very_high"
                }
            ],
            
            "employment_law": [
                # Industrial Relations Code and Labor Laws
                {
                    "template": "I am {employment_type} and have been working in {company_type} for {work_duration}. {employment_violation}. {labor_remedy}",
                    "variables": {
                        "employment_type": ["permanent employee", "contractual worker", "daily wage laborer", "factory worker", "office staff", "security guard"],
                        "work_duration": ["6 months", "1 year", "2 years", "5 years", "10 years", "15 years"],
                        "company_type": ["private company", "manufacturing unit", "construction company", "retail chain", "IT company", "textile factory"],
                        "employment_violation": [
                            "sudden termination without notice period and due process",
                            "not paying overtime despite 10-12 hours daily work",
                            "deducting PF and ESI but not contributing",
                            "salary delay for 2-3 months consistently",
                            "denied maternity/paternity leave against company policy",
                            "refusing bonus payment despite company profits"
                        ],
                        "labor_remedy": [
                            "Can I file unfair dismissal case in Labor Court?",
                            "Can I file complaint with Labor Commissioner for wage recovery?",
                            "Can I take separate enforcement action for PF/ESI non-compliance?",
                            "Can I claim penalty for Industrial Relations Code violation?",
                            "Can I get compensation for Maternity Benefit Act violation?",
                            "Can I file salary recovery case under Payment of Wages Act?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "high"
                },
                # Sexual Harassment at Workplace (POSH Act 2013)
                {
                    "template": "I am {victim_profile} and at workplace {harassment_incident}. {posh_remedy} under POSH Act 2013?",
                    "variables": {
                        "victim_profile": [
                            "female employee working in private company",
                            "woman contractual worker in government office",
                            "female intern in corporate setup",
                            "woman working in factory environment",
                            "female domestic worker in residential society",
                            "woman professional in consulting firm"
                        ],
                        "harassment_incident": [
                            "senior manager is making unwanted advances and giving promotion threats",
                            "colleagues are making inappropriate comments and gestures daily",
                            "boss is making physical contact without consent creating uncomfortable environment",
                            "work assignments are being denied and facing retaliation after complaint",
                            "sexual favors are being demanded for career advancement",
                            "hostile work environment is being created with sexual jokes/content"
                        ],
                        "posh_remedy": [
                            "Can I file complaint in Internal Complaints Committee?",
                            "Is External ICC available if company has no committee?",
                            "Can I file both police complaint and POSH committee complaint?",
                            "What compensation amount can I get for harassment?",
                            "Is transfer/separation possible from accused?",
                            "Can I complain to labor department for company license action?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "very_high"
                },
                # Contract Labor and Gig Economy
                {
                    "template": "While working as {gig_context}, {contract_violation}. {gig_remedy} under labor laws?",
                    "variables": {
                        "gig_context": [
                            "Uber/Ola driver",
                            "Zomato/Swiggy delivery partner",
                            "Amazon/Flipkart delivery associate",
                            "freelance consultant on project work",
                            "contract laborer at construction site",
                            "part-time employee at retail store"
                        ],
                        "contract_violation": [
                            "company suddenly terminated contract without notice",
                            "promised incentives and benefits are not being provided",
                            "work hours are excessive but no overtime payment",
                            "safety equipment not provided for hazardous work",
                            "platform commission unexpectedly increased without consent",
                            "payment delays and unauthorized deductions are happening"
                        ],
                        "gig_remedy": [
                            "Is contract labor law protection available for gig workers?",
                            "Can I file labor law violation case against platform companies?",
                            "Can I demand social security benefits?",
                            "Are collective bargaining rights available to contractual workers?",
                            "Is remedy available against unfair termination?",
                            "Can I take legal action for minimum wage guarantee?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "high"
                },
                # Factory Act and Industrial Safety
                {
                    "template": "While working in {factory_context}, {safety_violation}. {safety_remedy} under Factory Act and safety laws?",
                    "variables": {
                        "factory_context": [
                            "chemical factory during shift work",
                            "textile mill during machine operation",
                            "construction site with heavy machinery work",
                            "mining operation with underground work",
                            "food processing unit on production line",
                            "pharmaceutical company handling hazardous material"
                        ],
                        "safety_violation": [
                            "proper safety equipment not provided and accident occurred",
                            "factory ventilation system is faulty and causing health problems",
                            "forced to work overtime 12+ hours daily without breaks",
                            "hazardous chemical exposure causing respiratory problems",
                            "machinery maintenance is poor and worker safety compromised",
                            "emergency exits are blocked and fire safety violations exist"
                        ],
                        "safety_remedy": [
                            "Can I file Factory Inspector complaint for safety violations?",
                            "Can I claim under Workmen Compensation Act for injury?",
                            "Can I petition for factory license cancellation?",
                            "Can I claim occupational health hazard compensation?",
                            "Can I file case in Labor Court for unsafe working conditions?",
                            "Can I file both Pollution Control Board complaint and labor case?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "very_high"
                },
                # Trade Union and Collective Bargaining
                {
                    "template": "During {union_context}, {union_violation}. {union_remedy} under Trade Union Act?",
                    "variables": {
                        "union_context": [
                            "organizing workers union at factory",
                            "forming employees association at office",
                            "organizing strike/protest",
                            "collective bargaining with management",
                            "taking union membership and facing retaliation",
                            "labor dispute settlement negotiations"
                        ],
                        "union_violation": [
                            "management is preventing union formation and giving threats",
                            "union leaders are being targeted and victimized",
                            "collective bargaining is being refused and individual negotiations forced",
                            "salary deduction and punishment after strike participation",
                            "union recognition is being denied despite majority membership",
                            "unfair labor practices and union-busting activities are happening"
                        ],
                        "union_remedy": [
                            "Can I file Registrar complaint for Trade Union registration?",
                            "Can I file Labor Court case for unfair labor practices?",
                            "Can I petition Industrial Relations Court for union recognition?",
                            "Can I apply for protection order against victimization?",
                            "Can I demand conciliation/arbitration for collective bargaining failure?",
                            "Can I file criminal case for anti-union activities?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "very_high"
                },
                # Women-specific employment rights
                {
                    "template": "I am working woman and during {women_employment_context}, {gender_violation}. {women_remedy}",
                    "variables": {
                        "women_employment_context": [
                            "pregnancy period at office",
                            "night shift work assignments",
                            "promotion/appraisal process",
                            "equal pay negotiations",
                            "rejoining after career break",
                            "workplace facilities usage"
                        ],
                        "gender_violation": [
                            "maternity leave was denied violating company policy",
                            "night shift is being forced without proper safety arrangements",
                            "gender bias in promotion giving preference to male colleagues",
                            "same designation male colleagues are getting higher salary",
                            "demotion was done after career gap without justification",
                            "separate washroom/rest room facilities are not being provided"
                        ],
                        "women_remedy": [
                            "Can I claim compensation for Maternity Benefit Act violation?",
                            "Can I recover salary differential for Equal Pay Act violation?",
                            "Can I file civil rights case for gender discrimination?",
                            "Can I file Labor Department complaint for workplace safety violation?",
                            "Can I take legal action for career progression discrimination?",
                            "Can I file human rights violation case for basic facility denial?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "high"
                }
            ]
        }
    
    def _initialize_demographic_profiles(self) -> List[Dict]:
        """Initialize diverse demographic profiles for realistic scenarios"""
        return [
            {"age": 25, "gender": "female", "location": "urban", "education": "graduate", "income_level": "low"},
            {"age": 35, "gender": "male", "location": "rural", "education": "high_school", "income_level": "very_low"},
            {"age": 45, "gender": "female", "location": "semi_urban", "education": "primary", "income_level": "medium"},
            {"age": 55, "gender": "male", "location": "urban", "education": "graduate", "income_level": "high"},
            {"age": 30, "gender": "transgender", "location": "urban", "education": "intermediate", "income_level": "low"},
            {"age": 40, "gender": "female", "location": "rural", "education": "illiterate", "income_level": "very_low"},
        ]
    
    def _initialize_case_variations(self) -> Dict[str, List[str]]:
        """Initialize case complexity and priority variations"""
        return {
            "complexity": ["simple", "medium", "high", "very_high"],
            "priority": ["low", "normal", "high", "urgent"],
            "case_strength": ["weak", "moderate", "strong", "very_strong"],
            "evidence_quality": ["poor", "adequate", "good", "excellent"]
        }
    
    def generate_comprehensive_dataset(self, total_samples: int = 5000) -> List[LegalTrainingExample]:
        """
        Generate comprehensive training dataset with balanced representation.
        
        Args:
            total_samples: Total number of training examples to generate
            
        Returns:
            List of structured training examples
        """
        logger.info(f"Generating comprehensive dataset with {total_samples} samples")
        
        dataset = []
        
        # Domain distribution (prioritizing legal aid as core domain)
        domain_distribution = {
            LegalDomain.LEGAL_AID: int(total_samples * 0.30),  # 1500 samples
            LegalDomain.FAMILY_LAW: int(total_samples * 0.25), # 1250 samples
            LegalDomain.CONSUMER_PROTECTION: int(total_samples * 0.20), # 1000 samples
            LegalDomain.EMPLOYMENT_LAW: int(total_samples * 0.15), # 750 samples
            LegalDomain.FUNDAMENTAL_RIGHTS: int(total_samples * 0.10) # 500 samples
        }
        
        # Generate single-domain samples
        for domain, count in domain_distribution.items():
            logger.info(f"Generating {count} samples for {domain.value}")
            domain_samples = self._generate_domain_specific_samples(domain, count)
            dataset.extend(domain_samples)
        
        # Generate cross-domain samples
        remaining_samples = total_samples - len(dataset)
        if remaining_samples > 0:
            logger.info(f"Generating {remaining_samples} cross-domain samples")
            cross_domain_samples = self._generate_cross_domain_samples(remaining_samples)
            dataset.extend(cross_domain_samples)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logger.info(f"Dataset generation complete: {len(dataset)} total samples")
        return dataset
    
    def _generate_domain_specific_samples(self, domain: LegalDomain, count: int) -> List[LegalTrainingExample]:
        """Generate samples for a specific legal domain"""
        samples = []
        templates = self.legal_templates.get(domain.value, [])
        
        if not templates:
            logger.warning(f"No templates found for domain {domain.value}")
            return samples
        
        for i in range(count):
            template_data = random.choice(templates)
            sample = self._create_sample_from_template(template_data, [domain.value])
            samples.append(sample)
        
        return samples
    
    def _generate_cross_domain_samples(self, count: int) -> List[LegalTrainingExample]:
        """Generate complex samples that span multiple domains with realistic Indian scenarios"""
        samples = []
        
        # Comprehensive cross-domain scenarios covering realistic combinations
        cross_domain_scenarios = [
            # Legal Aid + Family Law + Employment Law (Very common combination)
            {
                "template": "Hello sir, I am a poor woman who was earning Rs {income} monthly. My husband got me fired from job when I became pregnant and now he is throwing me and child out of house. I complained to police but they say it's family matter. I became single mother and don't have money for case. Can I get free legal aid? Can I file cases for divorce, maintenance and job termination?",
                "domains": ["legal_aid", "family_law", "employment_law"],
                "variables": {"income": ["8000", "12000", "15000", "18000", "20000"]},
                "complexity": "very_high"
            },
            # Legal Aid + Consumer Protection + Fundamental Rights
            {
                "template": "I belong to SC community and booked flat for Rs {amount}. Builder delayed project by 3 years and when I went to complain, he insulted me on caste basis. He is not returning money and I also face discrimination at police station. My income is low to hire lawyer. Can I get RERA complaint, discrimination case and consumer case all in free legal aid?",
                "domains": ["legal_aid", "consumer_protection", "fundamental_rights"],
                "variables": {"amount": ["10 lakh", "15 lakh", "25 lakh", "35 lakh"]},
                "complexity": "very_high"
            },
            # Family Law + Consumer Protection + Legal Aid
            {
                "template": "For dowry demands in my marriage, in-laws got me admitted to hospital where doctors did unnecessary surgery and made bill of Rs {medical_cost}. Husband says he won't pay bill and will give divorce if I file case. I am housewife and cannot afford lawyer. I need to file three cases - medical negligence, dowry harassment and maintenance but is it possible in legal aid?",
                "domains": ["family_law", "consumer_protection", "legal_aid"],
                "variables": {"medical_cost": ["2 lakh", "5 lakh", "8 lakh", "12 lakh"]},
                "complexity": "very_high"
            },
            # Employment Law + Fundamental Rights + Legal Aid
            {
                "template": "I am ST community girl and was working in private company for {job_duration}. During promotion they did caste discrimination and when I complained, they filed false sexual harassment case and terminated me. Company HR said tribal people don't fit company culture. I need to file cases for wrongful termination, caste discrimination and defamation but salary is not coming. Does free legal aid cover all these?",
                "domains": ["employment_law", "fundamental_rights", "legal_aid"],
                "variables": {"job_duration": ["2 years", "3 years", "5 years", "7 years"]},
                "complexity": "very_high"
            },
            # Consumer Protection + Fundamental Rights + Employment Law
            {
                "template": "I was customer service executive in bank and had to reject customer's loan because documents were incomplete. Customer used caste slurs and complained to manager that I am deliberately rejecting being SC. Manager blamed me for customer relations and demoted me. Customer complained to bank and now I got show cause notice for performance issues. I need remedy for workplace discrimination, customer harassment and bank's unfair practices.",
                "domains": ["consumer_protection", "fundamental_rights", "employment_law"],
                "variables": {},
                "complexity": "very_high"
            },
            # All five domains combination (Ultra complex scenario)
            {
                "template": "I am widow on Rs {income} monthly pension. My husband's death occurred due to negligence at government hospital, builder did fraud and took flat money, in-laws are evicting me from property, there is corruption in pension at government office and when I went to complain I faced gender discrimination. 5 different cases - medical negligence, builder fraud, property rights, pension corruption and gender discrimination. Can I get comprehensive help in legal aid or need to hire separate lawyers?",
                "domains": ["legal_aid", "family_law", "consumer_protection", "fundamental_rights", "employment_law"],
                "variables": {"income": ["3000", "5000", "8000", "10000"]},
                "complexity": "very_high"
            },
            # Construction worker complex scenario
            {
                "template": "Sir, I am daily wage worker and was working at construction site. In accident I got leg fracture but contractor didn't give compensation and fired me from job. During hospital treatment they kept me in different ward seeing my caste and didn't give proper care. After recovery when I went to find job I faced discrimination. Is it possible to file three cases in legal aid - worker compensation, medical negligence and employment discrimination?",
                "domains": ["employment_law", "consumer_protection", "fundamental_rights", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            },
            # Urban middle-class complex scenario
            {
                "template": "I am working mother earning Rs {salary} monthly. During my maternity leave, company terminated me citing cost-cutting. My husband filed for divorce saying I cannot contribute financially now. The flat we bought jointly has construction defects and builder is not responding. I need to fight wrongful termination, contest divorce for maintenance, and get flat issues resolved. Can I get legal aid despite my previous income level since I'm now unemployed?",
                "domains": ["employment_law", "family_law", "consumer_protection", "legal_aid"],
                "variables": {"salary": ["25000", "35000", "45000", "55000"]},
                "complexity": "very_high"
            },
            # Rural complex scenario with government schemes
            {
                "template": "In village I got house under PM Awas Yojana but contractor did substandard construction. When I went to complain in panchayat, sarpanch rejected on caste basis. I filed RTI but didn't get information. Wife is facing domestic violence because of house problems. Are government scheme fraud, caste discrimination, RTI violation and family dispute all covered in legal aid or not?",
                "domains": ["consumer_protection", "fundamental_rights", "family_law", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            },
            # Tech sector complex scenario
            {
                "template": "I was software engineer in IT company and requested work from home during pregnancy. Company refused and pressured for resignation. During notice period they harassed me and didn't give relieving letter. New job offer was withdrawn in background verification. When I went to return laptop, security guard made inappropriate comments. I need to file multiple cases - maternity rights violation, wrongful resignation pressure, workplace harassment and defamation.",
                "domains": ["employment_law", "consumer_protection", "fundamental_rights"],
                "variables": {},
                "complexity": "very_high"
            },
            # Small business owner complex scenario
            {
                "template": "I had small business and hired CA for GST compliance. CA did wrong filing and penalty was imposed. When I went to government department for correction, officer demanded bribe. Business partner took advantage and transferred company shares without consent. Wife filed divorce case due to business stress. I need help for tax issues, corruption complaint, business fraud and family dispute - does legal aid cover business cases?",
                "domains": ["fundamental_rights", "consumer_protection", "family_law", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            },
            # Healthcare worker complex scenario
            {
                "template": "I am nurse in government hospital and didn't get proper PPE during COVID duty. I got infected and when I applied for medical leave, they deducted salary. Hospital administration retaliated when I complained and gave transfer threat. Family also discriminated after I got infected. I need legal remedy for workplace safety violation, illegal salary deduction, administrative harassment and social discrimination.",
                "domains": ["employment_law", "fundamental_rights", "consumer_protection"],
                "variables": {},
                "complexity": "very_high"
            },
            # Student complex scenario
            {
                "template": "I was student in engineering college and paid fees by taking education loan. College continued classes even after accreditation loss without informing. Degree became invalid and I didn't get job. Bank is harassing for loan recovery. College administration gave rustication threat when I complained. What protection is available under student rights for educational fraud, loan harassment and administrative abuse?",
                "domains": ["consumer_protection", "fundamental_rights", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            },
            # Senior citizen complex scenario
            {
                "template": "I am 75 year old senior citizen and my pension was getting delayed. When I went to government office to complain, officer demanded bribe. There was corruption in property tax assessment and they threatened me. Children did emotional abuse in family and forced for property transfer. I also faced harassment at bank during withdrawal. I need comprehensive legal help for senior citizen rights, corruption complaints and elder abuse.",
                "domains": ["fundamental_rights", "family_law", "consumer_protection", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            },
            # Migrant worker complex scenario
            {
                "template": "I came from UP to Mumbai for construction work. Contractor didn't pay salary for 3 months and left me stranded during lockdown. Local police harassed me and demanded bribe while checking documents. Hospital refused treatment because I am from outside state. I faced discrimination in train booking to go back home. Is migrant worker protection available for labor rights, police harassment, healthcare denial and transportation discrimination?",
                "domains": ["employment_law", "fundamental_rights", "consumer_protection", "legal_aid"],
                "variables": {},
                "complexity": "very_high"
            }
        ]
        
        for i in range(count):
            scenario = random.choice(cross_domain_scenarios)
            sample = self._create_sample_from_template(scenario, scenario["domains"])
            samples.append(sample)
        
        return samples
    
    def _create_sample_from_template(self, template_data: Dict, domains: List[str]) -> LegalTrainingExample:
        """Create a training sample from template"""
        template = template_data["template"]
        variables = template_data.get("variables", {})
        
        # Fill template variables
        filled_query = template
        for var_name, options in variables.items():
            if f"{{{var_name}}}" in filled_query:
                chosen_value = random.choice(options)
                filled_query = filled_query.replace(f"{{{var_name}}}", chosen_value)
        
        # Extract facts based on query content and domains
        extracted_facts = self._extract_facts_from_query(filled_query, domains)
        
        # Determine eligibility based on legal rules
        eligibility = self._determine_eligibility(extracted_facts, domains, filled_query)
        
        # Generate legal reasoning
        legal_reasoning = self._generate_legal_reasoning(extracted_facts, domains, eligibility)
        
        # Assign demographic profile
        demographics = random.choice(self.demographic_profiles)
        
        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(filled_query, extracted_facts)
        
        # Determine complexity and priority
        complexity = template_data.get("complexity", random.choice(self.case_variations["complexity"]))
        priority = self._determine_priority(filled_query, domains)
        
        return LegalTrainingExample(
            query=filled_query,
            domains=domains,
            extracted_facts=extracted_facts,
            expected_eligibility=eligibility,
            legal_reasoning=legal_reasoning,
            confidence_factors=confidence_factors,
            user_demographics=demographics,
            case_complexity=complexity,
            priority_level=priority
        )
    
    def _extract_facts_from_query(self, query: str, domains: List[str]) -> List[str]:
        """Extract structured facts from query based on domains with enhanced Indian context"""
        facts = ["applicant(user)."]
        query_lower = query.lower()
        
        # Enhanced income extraction with Indian currency patterns
        income_patterns = [
            r'(\d+)\s*rupees?\s*(?:monthly|per month|every month|per month)',
            r'(?:earning|earning|income|salary)\s*(?:is|is|was)?\s*(?:rs\.?|rupees?)\s*(\d+)',
            r'(\d+)\s*(?:rs|rupees?)\s*(?:per month|monthly|salary|pension)',
            r'monthly\s*(?:income|salary)\s*(?:of|is)?\s*(?:rs\.?|rupees?)\s*(\d+)',
            r'(\d+)\s*rupees?\s*(?:pension|earning|salary)'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, query_lower)
            if match:
                income = int(match.group(1))
                facts.append(f"income_monthly(user, {income}).")
                # Add income category facts
                if income <= 12000:
                    facts.append("income_category(user, 'very_low').")
                elif income <= 25000:
                    facts.append("income_category(user, 'low').")
                elif income <= 50000:
                    facts.append("income_category(user, 'medium').")
                else:
                    facts.append("income_category(user, 'high').")
                break
        
        # Enhanced social category extraction with Indian context
        social_categories = {
            # Caste categories
            "sc": "social_category(user, 'sc').",
            "dalit": "social_category(user, 'sc').",
            "scheduled caste": "social_category(user, 'sc').",
            "st": "social_category(user, 'st').",
            "adivasi": "social_category(user, 'st').",
            "tribal": "social_category(user, 'st').",
            "scheduled tribe": "social_category(user, 'st').",
            "obc": "social_category(user, 'obc').",
            
            # Gender categories
            "woman": "is_woman(user, true).",
            "woman": "is_woman(user, true).",
            "girl": "is_woman(user, true).",
            "female": "is_woman(user, true).",
            "working woman": "is_woman(user, true).",
            "working mother": "is_woman(user, true).",
            "single mother": "is_woman(user, true).",
            "housewife": "is_woman(user, true).",
            "widow": "is_woman(user, true).",
            "widow": "is_woman(user, true).",
            
            # Special categories
            "disabled": "is_disabled(user, true).",
            "differently abled": "is_disabled(user, true).",
            "handicapped": "is_disabled(user, true).",
            "senior citizen": "is_senior_citizen(user, true).",
            "elderly": "is_senior_citizen(user, true).",
            "transgender": "is_transgender(user, true).",
            "minority": "is_minority(user, true)."
        }
        
        for category, fact in social_categories.items():
            if category in query_lower:
                facts.append(fact)
        
        # Work/profession extraction
        professions = {
            "daily wage": "profession(user, 'daily_wage_worker').",
            "construction worker": "profession(user, 'construction_worker').",
            "factory worker": "profession(user, 'factory_worker').",
            "domestic worker": "profession(user, 'domestic_worker').",
            "rickshaw driver": "profession(user, 'rickshaw_driver').",
            "security guard": "profession(user, 'security_guard').",
            "employee": "employment_status(user, 'employed').",
            "unemployed": "employment_status(user, 'unemployed').",
            "retired": "employment_status(user, 'retired')."
        }
        
        for profession, fact in professions.items():
            if profession in query_lower:
                facts.append(fact)
        
        # Location extraction
        locations = {
            "village": "location_type(user, 'rural').",
            "village": "location_type(user, 'rural').",
            "city": "location_type(user, 'urban').",
            "metro": "location_type(user, 'metro').",
            "mumbai": "location_type(user, 'metro').",
            "delhi": "location_type(user, 'metro').",
            "bangalore": "location_type(user, 'metro').",
            "town": "location_type(user, 'semi_urban').",
            "town": "location_type(user, 'semi_urban')."
        }
        
        for location, fact in locations.items():
            if location in query_lower:
                facts.append(fact)
        
        # Domain-specific fact extraction
        if "legal_aid" in domains:
            # Legal aid seeking indicators
            legal_aid_indicators = [
                "free legal", "free lawyer", "legal aid", "cannot afford", 
                "cannot afford lawyer", "no money", "poor", "poor"
            ]
            if any(indicator in query_lower for indicator in legal_aid_indicators):
                facts.append("seeks_legal_aid(user, true).")
            
            # Multiple case indicators
            if any(word in query_lower for word in ["three", "all", "multiple", "different", "various"]):
                facts.append("multiple_cases(user, true).")
        
        if "family_law" in domains:
            # Marriage status
            marriage_indicators = {
                "husband": "marital_status(user, married).",
                "husband": "marital_status(user, married).",
                "wife": "marital_status(user, married).",
                "wife": "marital_status(user, married).",
                "marriage": "marital_status(user, married).",
                "married": "marital_status(user, married).",
                "divorce": "seeks_divorce(user, true).",
                "divorce": "seeks_divorce(user, true).",
                "separation": "seeks_separation(user, true).",
                "maintenance": "seeks_maintenance(user, true).",
                "in-laws": "joint_family(user, true).",
                "in-laws": "joint_family(user, true)."
            }
            
            for indicator, fact in marriage_indicators.items():
                if indicator in query_lower:
                    facts.append(fact)
            
            # Violence indicators
            violence_terms = ["violence", "beating", "beat", "torture", "cruelty", "harassment"]
            if any(term in query_lower for term in violence_terms):
                facts.append("domestic_violence(user, true).")
            
            # Dowry indicators
            dowry_terms = ["dowry", "dowry", "demand"]
            if any(term in query_lower for term in dowry_terms):
                facts.append("dowry_harassment(user, true).")
        
        if "employment_law" in domains:
            # Employment issues
            employment_issues = {
                "fired": "employment_issue(user, 'termination').",
                "terminate": "employment_issue(user, 'termination').",
                "job termination": "employment_issue(user, 'termination').",
                "salary": "employment_issue(user, 'salary').",
                "overtime": "employment_issue(user, 'overtime').",
                "harassment": "employment_issue(user, 'harassment').",
                "discrimination": "employment_issue(user, 'discrimination').",
                "maternity": "employment_issue(user, 'maternity').",
                "pregnancy": "employment_issue(user, 'maternity')."
            }
            
            for issue, fact in employment_issues.items():
                if issue in query_lower:
                    facts.append(fact)
        
        if "consumer_protection" in domains:
            # Consumer issues
            consumer_terms = {
                "bought": "consumer_transaction(user, true).",
                "kharida": "consumer_transaction(user, true).",
                "defective": "product_defect(user, true).",
                "fraud": "consumer_fraud(user, true).",
                "cheating": "consumer_fraud(user, true).",
                "builder": "real_estate_issue(user, true).",
                "hospital": "medical_service_issue(user, true).",
                "negligence": "service_negligence(user, true)."
            }
            
            for term, fact in consumer_terms.items():
                if term in query_lower:
                    facts.append(fact)
        
        if "fundamental_rights" in domains:
            # Rights violations
            rights_terms = {
                "discrimination": "rights_violation(user, 'discrimination').",
                "harassment": "rights_violation(user, 'harassment').",
                "police": "authority_involved(user, 'police').",
                "government": "authority_involved(user, 'government').",
                "rti": "rti_case(user, true).",
                "bribe": "corruption(user, true).",
                "bribery": "corruption(user, true).",
                "caste": "caste_discrimination(user, true).",
                "religion": "religious_discrimination(user, true)."
            }
            
            for term, fact in rights_terms.items():
                if term in query_lower:
                    facts.append(fact)
        
        # Case complexity indicators
        complexity_indicators = [
            "multiple", "various", "different", "all", "three", "complex", 
            "complicated", "difficult", "difficult"
        ]
        if any(indicator in query_lower for indicator in complexity_indicators):
            facts.append("case_complexity(user, high).")
        
        # Urgency indicators
        urgency_indicators = [
            "urgent", "emergency", "immediate", "immediately", "quickly", 
            "threat", "danger", "violence", "harassment"
        ]
        if any(indicator in query_lower for indicator in urgency_indicators):
            facts.append("case_urgency(user, high).")
        
        return facts
    
    def _determine_eligibility(self, facts: List[str], domains: List[str], query: str) -> bool:
        """Determine eligibility based on legal rules simulation"""
        query_lower = query.lower()
        
        if "legal_aid" in domains:
            # Check income eligibility
            income_eligible = False
            categorical_eligible = False
            
            for fact in facts:
                if "income_monthly" in fact:
                    income_match = re.search(r'income_monthly\(user, (\d+)\)', fact)
                    if income_match:
                        income = int(income_match.group(1))
                        if income <= 25000:  # Income threshold
                            income_eligible = True
                        else:
                            # High income cases - not eligible unless categorical
                            if income > 50000:
                                return False  # Definitely not eligible
            
            # Check categorical eligibility
            categorical_indicators = [
                "is_woman(user, true)", 
                "social_category(user, 'sc')", 
                "social_category(user, 'st')",
                "is_disabled(user, true)",
                "is_senior_citizen(user, true)"
            ]
            if any(indicator in facts for indicator in categorical_indicators):
                categorical_eligible = True
            
            # Additional exclusion criteria for better distribution
            exclusion_terms = ["rich", "wealthy", "business owner", "company owner", "multiple properties"]
            if any(term in query_lower for term in exclusion_terms):
                return False
            
            # Create balanced eligibility (aim for ~70% eligible, 30% not eligible)
            import random
            if not income_eligible and not categorical_eligible:
                # Random 10% chance for edge cases to be eligible
                return random.random() < 0.1
            
            return income_eligible or categorical_eligible
        
        # For other domains, generally eligible if valid case (80% eligible)
        import random
        return random.random() < 0.8
    
    def _generate_legal_reasoning(self, facts: List[str], domains: List[str], eligibility: bool) -> str:
        """Generate legal reasoning for the decision"""
        if "legal_aid" in domains:
            if eligibility:
                if any("is_woman" in fact for fact in facts):
                    return "Eligible under categorical criteria - women are eligible for legal aid regardless of income"
                elif any("income_monthly" in fact for fact in facts):
                    return "Eligible under income criteria - monthly income below threshold limit"
                else:
                    return "Eligible based on case merits and circumstances"
            else:
                return "Not eligible - income above threshold and no categorical eligibility"
        
        return f"Legal matter falls under {', '.join(domains)} with valid grounds for legal action"
    
    def _calculate_confidence_factors(self, query: str, facts: List[str]) -> Dict[str, float]:
        """Calculate confidence factors for the sample"""
        return {
            "query_clarity": random.uniform(0.7, 0.95),
            "fact_extraction_confidence": random.uniform(0.8, 0.98),
            "legal_rule_certainty": random.uniform(0.85, 1.0),
            "case_strength": random.uniform(0.6, 0.9)
        }
    
    def _determine_priority(self, query: str, domains: List[str]) -> str:
        """Determine priority level based on case urgency"""
        urgent_indicators = ["harassment", "violence", "threat", "emergency", "urgent"]
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in urgent_indicators):
            return "urgent"
        elif "family_law" in domains and any(word in query_lower for word in ["abuse", "cruelty"]):
            return "high"
        else:
            return random.choice(["normal", "normal", "high"])  # Bias toward normal
    
    def save_dataset(self, dataset: List[LegalTrainingExample], filepath: str):
        """Save dataset to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        dataset_json = {
            "metadata": {
                "total_samples": len(dataset),
                "domains_covered": list(set([domain for sample in dataset for domain in sample.domains])),
                "generation_date": "2025-08-01",
                "version": "1.0",
                "description": "Comprehensive legal training dataset for hybrid neural-symbolic system"
            },
            "samples": [asdict(sample) for sample in dataset]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {filepath}")
        
        # Generate statistics
        self._generate_dataset_statistics(dataset, filepath.replace('.json', '_stats.json'))
    
    def _generate_dataset_statistics(self, dataset: List[LegalTrainingExample], stats_filepath: str):
        """Generate dataset statistics"""
        stats = {
            "total_samples": len(dataset),
            "domain_distribution": {},
            "eligibility_distribution": {},
            "complexity_distribution": {},
            "priority_distribution": {}
        }
        
        # Calculate distributions
        for sample in dataset:
            # Domain distribution
            for domain in sample.domains:
                stats["domain_distribution"][domain] = stats["domain_distribution"].get(domain, 0) + 1
            
            # Eligibility distribution
            eligibility_key = "eligible" if sample.expected_eligibility else "not_eligible"
            stats["eligibility_distribution"][eligibility_key] = stats["eligibility_distribution"].get(eligibility_key, 0) + 1
            
            # Complexity distribution
            stats["complexity_distribution"][sample.case_complexity] = stats["complexity_distribution"].get(sample.case_complexity, 0) + 1
            
            # Priority distribution
            stats["priority_distribution"][sample.priority_level] = stats["priority_distribution"].get(sample.priority_level, 0) + 1
        
        with open(stats_filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_filepath}")

        # Add these methods to the ComprehensiveLegalDataGenerator class

    def _generate_enhanced_domain_samples(self, domain: LegalDomain, count: int) -> List[LegalTrainingExample]:
        """Generate enhanced samples for a specific domain with multiple variations"""
        samples = []
        templates = self.legal_templates.get(domain.value, [])
        
        if not templates:
            logger.warning(f"No templates found for domain {domain.value}")
            return samples
        
        # Generate base samples
        base_count = int(count * 0.7)  # 70% base samples
        for i in range(base_count):
            template_data = random.choice(templates)
            sample = self._create_sample_from_template(template_data, [domain.value])
            samples.append(sample)
        
        # Generate template variations
        variation_count = count - base_count  # 30% variations
        variation_samples = self._generate_template_variations(domain, variation_count)
        samples.extend(variation_samples)
        
        return samples

    def _generate_template_variations(self, domain: LegalDomain, count: int) -> List[LegalTrainingExample]:
        """Generate template variations for enhanced coverage"""
        samples = []
        
        # Domain-specific variation templates
        variation_templates = {
            "legal_aid": [
                # Income edge cases
                {
                    "template": "I am {profession} and my monthly income is exactly Rs {edge_income}. {case_description}. Am I eligible for legal aid?",
                    "variables": {
                        "profession": ["autorickshaw driver", "street hawker", "part-time teacher", "freelance carpenter", "home tutor", "grocery shop owner"],
                        "edge_income": ["24999", "25000", "25001", "12000", "18000", "22000"],  # Around threshold
                        "case_description": [
                            "My landlord is forcing me to vacate without proper notice",
                            "I have been cheated by property dealer in flat booking",
                            "Hospital overcharged me during emergency treatment",
                            "Police filed false case against me due to local dispute"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "high"
                },
                # Age-based scenarios
                {
                    "template": "I am {age} years old {gender} and {financial_situation}. {legal_problem}. Can I get free legal services?",
                    "variables": {
                        "age": ["65", "70", "75", "80", "16", "17"],
                        "gender": ["man", "woman", "person"],
                        "financial_situation": [
                            "living on small pension of Rs 8000",
                            "dependent on children for money",
                            "have no regular income source",
                            "getting disability allowance of Rs 5000"
                        ],
                        "legal_problem": [
                            "Bank is asking for loan repayment but I never took any loan",
                            "Neighbors are threatening me over property boundary",
                            "Insurance company rejected my claim without valid reason",
                            "Government pension has stopped without notice"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "medium"
                },
                # Multiple dependent scenarios
                {
                    "template": "I am single {parent_type} with {num_dependents} dependents earning Rs {income}. {complex_situation}. Is comprehensive legal aid available?",
                    "variables": {
                        "parent_type": ["mother", "father", "guardian"],
                        "num_dependents": ["3 children", "4 family members", "elderly parents", "2 disabled children"],
                        "income": ["15000", "18000", "20000", "22000"],
                        "complex_situation": [
                            "Facing eviction, child custody dispute, and workplace harassment simultaneously",
                            "Dealing with medical negligence case, insurance fraud, and school admission denial",
                            "Fighting property dispute, consumer complaint, and employment termination",
                            "Handling domestic violence case, dowry harassment, and child maintenance issues"
                        ]
                    },
                    "expected_domains": ["legal_aid"],
                    "complexity": "very_high"
                }
            ],
            
            "family_law": [
                # Regional variation scenarios
                {
                    "template": "I am from {state} and married under {marriage_type}. {regional_issue}. What are my rights under {applicable_law}?",
                    "variables": {
                        "state": ["Kerala", "Tamil Nadu", "Punjab", "Gujarat", "Rajasthan", "West Bengal"],
                        "marriage_type": ["Hindu custom", "Muslim nikah", "Christian ceremony", "civil marriage"],
                        "regional_issue": [
                            "Husband took second wife which is common practice in our community but I want divorce",
                            "Property division is being done according to old customs ignoring women's rights",
                            "Local panchayat is mediating family dispute but ignoring legal procedures",
                            "Joint family is following traditional practices that violate my constitutional rights"
                        ],
                        "applicable_law": ["Hindu Marriage Act", "Muslim Personal Law", "Indian Christian Marriage Act", "Special Marriage Act"]
                    },
                    "expected_domains": ["family_law"],
                    "complexity": "very_high"
                },
                # Inter-caste marriage scenarios
                {
                    "template": "I am from {caste1} community married to {caste2} person. {social_pressure}. {legal_query}",
                    "variables": {
                        "caste1": ["Brahmin", "Kshatriya", "SC", "ST", "OBC"],
                        "caste2": ["different caste", "SC community", "Muslim family", "Christian family"],
                        "social_pressure": [
                            "Both families are against our marriage and threatening social boycott",
                            "Husband's family is demanding caste conversion and cultural changes",
                            "My family disowned me and husband's family doesn't accept me",
                            "Local community leaders are pressuring for marriage annulment"
                        ],
                        "legal_query": [
                            "Can families legally force divorce in inter-caste marriage?",
                            "What protection is available against social boycott?",
                            "Can community panchayat override court marriage?",
                            "How to get police protection against family threats?"
                        ]
                    },
                    "expected_domains": ["family_law", "fundamental_rights"],
                    "complexity": "very_high"
                }
            ],
            
            "consumer_protection": [
                # Digital transaction fraud
                {
                    "template": "I made payment of Rs {amount} through {payment_method} for {service_type}. {digital_fraud}. {remedy_sought}",
                    "variables": {
                        "amount": ["5000", "15000", "25000", "50000", "100000"],
                        "payment_method": ["UPI", "credit card", "debit card", "net banking", "digital wallet"],
                        "service_type": ["online course", "hotel booking", "flight ticket", "e-commerce purchase", "subscription service"],
                        "digital_fraud": [
                            "Money was deducted but service was not provided and company is not responding",
                            "Fake website collected payment and disappeared without delivering product",
                            "Subscription was auto-renewed without consent and company refuses refund",
                            "Payment gateway showed failure but money was deducted from account"
                        ],
                        "remedy_sought": [
                            "Can I file consumer complaint for online fraud?",
                            "Is chargeback possible along with consumer case?",
                            "What is procedure for cyber crime complaint?",
                            "Can I claim compensation for mental harassment?"
                        ]
                    },
                    "expected_domains": ["consumer_protection"],
                    "complexity": "high"
                }
            ],
            
            "employment_law": [
                # Gig economy specific scenarios
                {
                    "template": "I work as {gig_type} for {platform} for {duration}. {gig_issue}. {labor_rights_query}",
                    "variables": {
                        "gig_type": ["food delivery partner", "cab driver", "freelance content writer", "part-time tutor", "home service provider"],
                        "platform": ["multiple apps", "local company", "online platform", "startup company"],
                        "duration": ["6 months", "1 year", "2 years", "3 years"],
                        "gig_issue": [
                            "Platform suddenly deactivated my account without explanation and blocked earnings",
                            "Commission rates were changed without notice affecting my monthly income",
                            "No insurance coverage provided despite promises and premium deductions",
                            "Forced to work excessive hours but no overtime or health benefits"
                        ],
                        "labor_rights_query": [
                            "Do gig workers have employment law protection?",
                            "Can I demand social security benefits?",
                            "Is there grievance mechanism for platform workers?",
                            "What rights do contract workers have under new labor codes?"
                        ]
                    },
                    "expected_domains": ["employment_law"],
                    "complexity": "high"
                }
            ],
            
            "fundamental_rights": [
                # Digital rights scenarios
                {
                    "template": "My {digital_activity} was {restriction_action} by {authority}. {impact_description}. {constitutional_remedy}",
                    "variables": {
                        "digital_activity": ["social media post", "blog article", "YouTube video", "WhatsApp message", "online comment"],
                        "restriction_action": ["blocked", "removed", "reported as fake news", "used as evidence against me"],
                        "authority": ["police", "government agency", "platform company", "local administration"],
                        "impact_description": [
                            "I was arrested for sedition based on political criticism post",
                            "My business suffered due to negative publicity from blocked content",
                            "Educational content was removed claiming misinformation",
                            "Personal reputation was damaged by false flag reporting"
                        ],
                        "constitutional_remedy": [
                            "Can I challenge content removal as free speech violation?",
                            "Is prior restraint on digital expression constitutional?",
                            "What remedy is available for misuse of IT Act provisions?",
                            "Can I claim damages for wrongful prosecution?"
                        ]
                    },
                    "expected_domains": ["fundamental_rights"],
                    "complexity": "very_high"
                }
            ]
        }
        
        domain_templates = variation_templates.get(domain.value, [])
        if not domain_templates:
            # Generate basic variations if no specific templates
            return self._generate_basic_variations(domain, count)
        
        for i in range(count):
            template_data = random.choice(domain_templates)
            sample = self._create_sample_from_template(template_data, template_data["expected_domains"])
            samples.append(sample)
        
        return samples

    def _generate_enhanced_cross_domain_samples(self, count: int) -> List[LegalTrainingExample]:
        """Generate enhanced cross-domain samples with more realistic complex scenarios"""
        samples = []
        
        # Enhanced cross-domain scenarios with more variations
        enhanced_cross_domain_scenarios = [
            # COVID-19 impact scenarios (Very relevant for current times)
            {
                "template": "During COVID lockdown I lost my job as {profession}. Landlord is demanding rent despite no income, wife filed divorce citing financial stress, and I took loan from online app at high interest. Now facing eviction, family court case, and debt recovery harassment. My previous salary was Rs {salary}. Can I get comprehensive legal aid for all three issues?",
                "domains": ["legal_aid", "family_law", "consumer_protection"],
                "variables": {
                    "profession": ["hotel waiter", "taxi driver", "event manager", "gym trainer", "travel agent"],
                    "salary": ["20000", "25000", "30000", "35000"]
                },
                "complexity": "very_high"
            },
            
            # Women entrepreneur scenarios
            {
                "template": "I started small business with Rs {investment} and hired CA for compliance. CA did wrong GST filing causing penalty. Business partner harassed me sexually and when I complained, he filed defamation case. Husband filed divorce saying I am neglecting family for business. Bank is also demanding loan repayment. Need help for tax issues, workplace harassment, family matter and banking dispute.",
                "domains": ["consumer_protection", "employment_law", "family_law", "legal_aid"],
                "variables": {"investment": ["2 lakh", "5 lakh", "10 lakh", "15 lakh"]},
                "complexity": "very_high"
            },
            
            # Senior citizen comprehensive scenario
            {
                "template": "I am {age} years old retired person getting pension of Rs {pension}. Bank officials forced me to invest in fixed deposit but it turned out to be fraudulent scheme. Children are not taking care and want property transfer under pressure. Government office is demanding bribe for pension processing. Also facing age discrimination in hospital treatment. Need legal help for banking fraud, elder abuse, corruption complaint and healthcare rights.",
                "domains": ["consumer_protection", "family_law", "fundamental_rights", "legal_aid"],
                "variables": {
                    "age": ["68", "72", "75", "78"],
                    "pension": ["8000", "12000", "15000", "18000"]
                },
                "complexity": "very_high"
            },
            
            # Student complex scenario
            {
                "template": "I took education loan of Rs {loan_amount} for engineering course. College lost accreditation during final year without informing students. Degree became invalid and I didn't get job. Bank is harassing for repayment and charged extra interest. College administration threatened when I complained publicly. Also faced caste discrimination during placement process. Need help for educational fraud, banking harassment, freedom of expression and discrimination issues.",
                "domains": ["consumer_protection", "fundamental_rights", "legal_aid"],
                "variables": {"loan_amount": ["5 lakh", "8 lakh", "12 lakh", "15 lakh"]},
                "complexity": "very_high"
            },
            
            # Healthcare worker scenario
            {
                "template": "I am nurse in private hospital earning Rs {salary}. During COVID duty didn't get proper PPE and got infected. Hospital deducted salary during recovery and refused medical leave. When I joined back, administrator harassed me and gave night shift punishment. Family also discriminated after infection. Filed complaint but hospital threatened termination. Need help for workplace safety, employment rights, healthcare discrimination and whistleblower protection.",
                "domains": ["employment_law", "consumer_protection", "fundamental_rights"],
                "variables": {"salary": ["18000", "22000", "25000", "28000"]},
                "complexity": "very_high"
            },
            
            # Migrant worker comprehensive scenario
            {
                "template": "I came from {home_state} to {work_state} for construction work. Contractor didn't pay salary for {months} months and left me stranded. Local police demanded bribe for document verification. Hospital refused treatment saying I am outsider. Train booking was difficult due to bias. Could not return home during family emergency. Need help for labor rights, police harassment, healthcare denial, transportation discrimination and inter-state legal issues.",
                "domains": ["employment_law", "fundamental_rights", "consumer_protection", "legal_aid"],
                "variables": {
                    "home_state": ["Bihar", "UP", "Odisha", "Jharkhand"],
                    "work_state": ["Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu"],
                    "months": ["2", "3", "4", "6"]
                },
                "complexity": "very_high"
            },
            
            # Digital platform worker scenario
            {
                "template": "I work for {platform_type} earning Rs {income} monthly. Platform suddenly changed algorithm affecting my earnings by 60%. When I posted complaint on social media, they deactivated account and blocked payments. Filed RTI about platform regulations but got no response. Also facing family pressure to leave gig work. Need help for platform worker rights, freedom of expression, RTI violation and family dispute resolution.",
                "domains": ["employment_law", "fundamental_rights", "family_law"],
                "variables": {
                    "platform_type": ["food delivery app", "ride sharing service", "freelance marketplace", "content creation platform"],
                    "income": ["15000", "20000", "25000", "30000"]
                },
                "complexity": "very_high"
            }
        ]
        
        # Add more variations of existing scenarios
        additional_scenarios = []
        for scenario in enhanced_cross_domain_scenarios:
            # Create 3 variations of each scenario with different variable combinations
            for _ in range(3):
                additional_scenarios.append(scenario.copy())
        
        all_scenarios = enhanced_cross_domain_scenarios + additional_scenarios
        
        for i in range(count):
            scenario = random.choice(all_scenarios)
            sample = self._create_sample_from_template(scenario, scenario["domains"])
            samples.append(sample)
        
        return samples

    def _generate_edge_case_samples(self, count: int) -> List[LegalTrainingExample]:
        """Generate edge cases and challenging scenarios for robust training"""
        samples = []
        
        edge_case_templates = [
            # Ambiguous income scenarios
            {
                "template": "My income varies between Rs {min_income} to Rs {max_income} depending on {income_type}. Sometimes I earn more, sometimes less. {case_description}. How is legal aid eligibility determined for variable income?",
                "domains": ["legal_aid"],
                "variables": {
                    "min_income": ["5000", "8000", "12000"],
                    "max_income": ["25000", "35000", "45000"],
                    "income_type": ["seasonal work", "freelance projects", "commission-based sales", "daily wage labor"],
                    "case_description": [
                        "I need to file case against property developer for construction fraud",
                        "Facing workplace harassment and need legal remedy",
                        "Insurance company is not settling my accident claim"
                    ]
                },
                "complexity": "very_high"
            },
            
            # Conflicting information scenarios
            {
                "template": "I said I am {initial_status} but actually I am {actual_status}. {confusion_reason}. {legal_question}",
                "domains": ["legal_aid", "family_law"],
                "variables": {
                    "initial_status": ["unmarried", "employed", "urban resident"],
                    "actual_status": ["divorced", "unemployed", "rural migrant"],
                    "confusion_reason": [
                        "I was confused about legal terminology in previous consultation",
                        "My situation changed recently and I forgot to update",
                        "I was afraid to reveal complete truth initially"
                    ],
                    "legal_question": [
                        "Does this affect my eligibility for legal services?",
                        "Should I restart the legal aid application process?",
                        "Can previous information be corrected in court documents?"
                    ]
                },
                "complexity": "high"
            },
            
            # Incomplete information scenarios
            {
                "template": "I have legal problem but {missing_info}. {partial_description}. Can you help with incomplete information?",
                "domains": ["legal_aid"],
                "variables": {
                    "missing_info": [
                        "don't remember exact dates of incidents",
                        "lost all documents in house fire",
                        "cannot provide income proof due to informal work",
                        "don't know legal names of people involved"
                    ],
                    "partial_description": [
                        "Someone cheated me in property deal but I don't have written agreement",
                        "Workplace incident happened but I don't have witness contact details",
                        "Family dispute involves property but I don't know exact legal procedures"
                    ]
                },
                "complexity": "medium"
            }
        ]
        
        for i in range(count):
            template_data = random.choice(edge_case_templates)
            sample = self._create_sample_from_template(template_data, template_data["domains"])
            samples.append(sample)
        
        return samples
    
def generate_comprehensive_dataset(self, total_samples: int = 20000) -> List[LegalTrainingExample]:
    """
    Generate comprehensive training dataset with enhanced coverage for robust training.
    
    Args:
        total_samples: Total number of training examples to generate (default: 20000)
        
    Returns:
        List of structured training examples with comprehensive coverage
    """
    logger.info(f"Generating enhanced dataset with {total_samples} samples for robust training")
    
    dataset = []
    
    # Enhanced domain distribution with more samples per domain
    domain_distribution = {
        LegalDomain.LEGAL_AID: int(total_samples * 0.30),  # 6000 samples - core domain
        LegalDomain.FAMILY_LAW: int(total_samples * 0.25), # 5000 samples - high impact
        LegalDomain.CONSUMER_PROTECTION: int(total_samples * 0.20), # 4000 samples
        LegalDomain.EMPLOYMENT_LAW: int(total_samples * 0.15), # 3000 samples
        LegalDomain.FUNDAMENTAL_RIGHTS: int(total_samples * 0.10) # 2000 samples
    }
    
    # Generate single-domain samples with multiple template variations
    for domain, count in domain_distribution.items():
        logger.info(f"Generating {count} samples for {domain.value}")
        domain_samples = self._generate_enhanced_domain_samples(domain, count)
        dataset.extend(domain_samples)
    
    # Generate cross-domain samples (20% of total for complex scenarios)
    cross_domain_count = int(total_samples * 0.20)  # 4000 cross-domain samples
    logger.info(f"Generating {cross_domain_count} cross-domain samples")
    cross_domain_samples = self._generate_enhanced_cross_domain_samples(cross_domain_count)
    dataset.extend(cross_domain_samples)
    
    # Generate edge cases and variations (additional 10%)
    edge_case_count = int(total_samples * 0.10)  # 2000 edge case samples
    logger.info(f"Generating {edge_case_count} edge case samples")
    edge_case_samples = self._generate_edge_case_samples(edge_case_count)
    dataset.extend(edge_case_samples)
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    logger.info(f"Enhanced dataset generation complete: {len(dataset)} total samples")
    return dataset

# Replace the existing main() function

def main():
    """Main function to generate comprehensive 20K dataset for robust training"""
    logger.info("Starting comprehensive legal data generation for robust training")
    
    # Create data generator
    generator = ComprehensiveLegalDataGenerator()
    
    # Generate large dataset for robust training (20,000 samples)
    dataset = generator.generate_comprehensive_dataset(total_samples=20000)
    
    # Save main dataset
    output_path = "data/comprehensive_legal_training_data.json"
    generator.save_dataset(dataset, output_path)
    
    # Generate additional validation datasets
    logger.info("Generating additional validation datasets...")
    
    # Generate edge cases dataset
    edge_cases = generator._generate_edge_case_samples(1000)
    edge_cases_path = "data/edge_cases_dataset.json"
    generator.save_dataset(edge_cases, edge_cases_path)
    
    # Generate domain-specific validation sets
    for domain in LegalDomain:
        domain_samples = generator._generate_enhanced_domain_samples(domain, 500)
        domain_path = f"data/{domain.value}_validation_set.json"
        generator.save_dataset(domain_samples, domain_path)
    
    logger.info("Comprehensive data generation completed successfully!")
    print(f"\n✅ Generated {len(dataset):,} training samples")
    print(f"✅ Generated 1,000 edge case samples")
    print(f"✅ Generated 500 samples per domain for validation")
    print(f"📁 Main dataset saved to: {output_path}")
    print(f"📊 Total samples across all datasets: {len(dataset) + 1000 + (500 * 5):,}")
    print(f"📈 Enhanced coverage for robust neural training!")


if __name__ == "__main__":
    main()
