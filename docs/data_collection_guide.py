"""
Legal Data Sources and Collection Strategies for HybEx-Law

This module provides guidance on collecting authentic legal aid data
and generating realistic training examples.
"""

# Official Indian Legal Sources
OFFICIAL_LEGAL_SOURCES = {
    "primary_legislation": {
        "Legal Services Authorities Act, 1987": "https://legislative.gov.in/sites/default/files/A1987-39.pdf",
        "National Legal Services Authority": "https://nalsa.gov.in/",
        "Legal Aid Schemes": "https://nalsa.gov.in/content/legal-aid-schemes",
        "State Legal Services Authorities": "https://nalsa.gov.in/content/state-legal-services-authorities"
    },
    
    "case_databases": {
        "Supreme Court of India": "https://main.sci.gov.in/judgments",
        "High Court Judgments": "https://www.sci.gov.in/judgments/",
        "District Court Cases": "Available through state judiciary websites",
        "Legal Aid Case Studies": "NALSA annual reports and publications"
    },
    
    "government_portals": {
        "eCourts Services": "https://ecourts.gov.in/",
        "Department of Justice": "https://doj.gov.in/",
        "Ministry of Law and Justice": "https://lawmin.gov.in/",
        "National Judicial Data Grid": "https://njdg.ecourts.gov.in/"
    },
    
    "legal_aid_resources": {
        "NALSA Publications": "Annual reports, scheme guidelines, case studies",
        "State Legal Aid Boards": "Local eligibility criteria and case examples",
        "Legal Aid Clinics": "Real case scenarios and outcomes",
        "Pro Bono Organizations": "Case documentation and client profiles"
    }
}

# Data Collection Strategies
DATA_COLLECTION_STRATEGIES = {
    "direct_sources": {
        "description": "Collect from official legal databases",
        "advantages": ["Authentic legal language", "Real case scenarios", "Official eligibility criteria"],
        "challenges": ["Privacy concerns", "Access restrictions", "Format standardization"],
        "methods": [
            "Web scraping public legal databases",
            "API access to court records",
            "Freedom of Information Act requests",
            "Partnership with legal aid organizations"
        ]
    },
    
    "synthetic_generation": {
        "description": "Generate realistic legal scenarios",
        "advantages": ["Privacy-safe", "Scalable", "Controllable diversity"],
        "challenges": ["Ensuring legal accuracy", "Avoiding bias", "Maintaining realism"],
        "methods": [
            "Template-based generation",
            "LLM-assisted creation with legal review",
            "Paraphrasing real cases (anonymized)",
            "Expert lawyer validation"
        ]
    },
    
    "crowd_sourcing": {
        "description": "Collect from legal practitioners and students",
        "advantages": ["Diverse perspectives", "Cost-effective", "Large scale"],
        "challenges": ["Quality control", "Legal accuracy", "Consistency"],
        "methods": [
            "Law student projects",
            "Legal aid volunteer contributions",
            "Online legal forums (anonymized)",
            "Professional legal networks"
        ]
    }
}

# Data Quality Guidelines
QUALITY_GUIDELINES = {
    "legal_accuracy": {
        "requirements": [
            "Compliance with Legal Services Authorities Act, 1987",
            "State-specific variations in eligibility criteria",
            "Current income thresholds and categories",
            "Accurate case type classifications"
        ],
        "validation_methods": [
            "Legal expert review",
            "Cross-reference with official guidelines",
            "Test against known outcomes",
            "Regular updates with legal changes"
        ]
    },
    
    "linguistic_diversity": {
        "requirements": [
            "Multiple ways to express same concepts",
            "Various educational backgrounds",
            "Regional language patterns",
            "Different levels of legal awareness"
        ],
        "techniques": [
            "Paraphrasing exercises",
            "Multi-lingual translation and back-translation",
            "Socio-economic variation in language use",
            "Age and education level considerations"
        ]
    },
    
    "scenario_coverage": {
        "requirements": [
            "All major case types",
            "Edge cases and boundary conditions",
            "Multiple eligibility pathways",
            "Common misconceptions and errors"
        ],
        "categories": [
            "Clear eligible cases",
            "Clear ineligible cases", 
            "Borderline/complex cases",
            "Multi-factor scenarios"
        ]
    }
}

# Privacy and Ethics Guidelines
PRIVACY_ETHICS = {
    "data_protection": [
        "Remove all personally identifiable information",
        "Use synthetic names and locations",
        "Anonymize financial details",
        "Protect sensitive case information"
    ],
    
    "ethical_considerations": [
        "Avoid bias against protected groups",
        "Ensure fair representation across demographics",
        "Respect cultural sensitivities",
        "Consider impact on vulnerable populations"
    ],
    
    "legal_compliance": [
        "Follow data protection regulations",
        "Obtain necessary permissions for data use",
        "Respect copyright and intellectual property",
        "Maintain ethical review standards"
    ]
}

# Recommended Data Collection Workflow
COLLECTION_WORKFLOW = [
    {
        "step": 1,
        "task": "Legal Framework Analysis",
        "description": "Study Legal Services Authorities Act, 1987 and state variations",
        "deliverables": ["Eligibility criteria documentation", "Case type taxonomy", "Decision tree logic"]
    },
    {
        "step": 2, 
        "task": "Source Identification",
        "description": "Identify and gain access to legal databases and resources",
        "deliverables": ["Source inventory", "Access agreements", "Data collection permissions"]
    },
    {
        "step": 3,
        "task": "Pilot Data Collection",
        "description": "Collect small sample for methodology validation",
        "deliverables": ["100-200 pilot cases", "Quality assessment", "Process refinement"]
    },
    {
        "step": 4,
        "task": "Large-scale Collection",
        "description": "Scale up data collection using validated methodology",
        "deliverables": ["1000+ training cases", "Validation dataset", "Test dataset"]
    },
    {
        "step": 5,
        "task": "Expert Validation",
        "description": "Legal expert review and validation of collected data",
        "deliverables": ["Validated dataset", "Quality metrics", "Error analysis"]
    },
    {
        "step": 6,
        "task": "Continuous Updates",
        "description": "Regular updates to reflect legal and social changes",
        "deliverables": ["Update procedures", "Version control", "Change tracking"]
    }
]
