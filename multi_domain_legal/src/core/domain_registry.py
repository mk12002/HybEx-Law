"""
Legal Domain Definitions for Multi-Domain Legal AI System.

This module defines the legal domains and their associated metadata
for the comprehensive legal AI system covering 5 major areas of Indian law.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass

class LegalDomain(Enum):
    """Enumeration of supported legal domains"""
    LEGAL_AID = "legal_aid"
    FAMILY_LAW = "family_law"
    CONSUMER_PROTECTION = "consumer_protection"
    FUNDAMENTAL_RIGHTS = "fundamental_rights"
    EMPLOYMENT_LAW = "employment_law"

@dataclass
class LegalAct:
    """Represents a legal act with its metadata"""
    name: str
    year: int
    short_name: str
    domain: LegalDomain
    keywords: List[str]
    description: str

class LegalDomainRegistry:
    """Registry of all legal domains and their associated acts"""
    
    def __init__(self):
        self.acts = self._initialize_acts()
        self.domain_keywords = self._initialize_domain_keywords()
        
    def _initialize_acts(self) -> List[LegalAct]:
        """Initialize all legal acts in the system"""
        return [
            # Legal Aid & Access to Justice
            LegalAct(
                name="Legal Services Authorities Act",
                year=1987,
                short_name="LSAA_1987",
                domain=LegalDomain.LEGAL_AID,
                keywords=["legal aid", "free lawyer", "government lawyer", "legal assistance", "pro bono"],
                description="Provides for legal services to ensure access to justice"
            ),
            LegalAct(
                name="Legal Aid to Women (Protection of Rights) Act",
                year=2013,
                short_name="LAWPRA_2013",
                domain=LegalDomain.LEGAL_AID,
                keywords=["women legal aid", "women rights", "protection", "gender justice"],
                description="Special legal aid provisions for women"
            ),
            LegalAct(
                name="Juvenile Justice (Care and Protection of Children) Act",
                year=2015,
                short_name="JJ_2015",
                domain=LegalDomain.LEGAL_AID,
                keywords=["juvenile", "children", "minor", "child protection", "adoption"],
                description="Protection and care of children in need"
            ),
            LegalAct(
                name="Rights of Persons with Disabilities Act",
                year=2016,
                short_name="RPWD_2016",
                domain=LegalDomain.LEGAL_AID,
                keywords=["disability", "disabled person", "special needs", "accessibility"],
                description="Rights and entitlements of persons with disabilities"
            ),
            LegalAct(  
                name="Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act",
                year=1989,
                short_name="SC_ST_POA_1989",
                domain=LegalDomain.LEGAL_AID,
                keywords=["sc st", "scheduled caste", "scheduled tribe", "atrocity", "discrimination"],
                description="Prevention of atrocities against SC/ST communities"
            ),
            
            # Family Law
            LegalAct(
                name="Hindu Marriage Act",
                year=1955,
                short_name="HMA_1955",
                domain=LegalDomain.FAMILY_LAW,
                keywords=["hindu marriage", "divorce", "hindu wedding", "marriage dissolution"],
                description="Marriage and divorce laws for Hindus"
            ),
            LegalAct(
                name="Hindu Succession Act",
                year=1956,
                short_name="HSA_1956", 
                domain=LegalDomain.FAMILY_LAW,
                keywords=["inheritance", "succession", "property", "hindu property"],
                description="Inheritance and succession laws for Hindus"
            ),
            LegalAct(
                name="Indian Christian Marriage Act",
                year=1872,
                short_name="ICMA_1872",
                domain=LegalDomain.FAMILY_LAW,
                keywords=["christian marriage", "christian divorce", "christian wedding"],
                description="Marriage laws for Christians in India"
            ),
            LegalAct(
                name="Parsi Marriage and Divorce Act",
                year=1936,
                short_name="PMDA_1936",
                domain=LegalDomain.FAMILY_LAW,
                keywords=["parsi marriage", "parsi divorce", "zoroastrian"],
                description="Marriage and divorce laws for Parsis"
            ),
            LegalAct(
                name="Muslim Personal Law (Shariat) Application Act",
                year=1937,
                short_name="MPL_1937",
                domain=LegalDomain.FAMILY_LAW,
                keywords=["muslim marriage", "shariat", "islamic law", "nikah", "talaq"],
                description="Personal law for Muslims"
            ),
            
            # Consumer Protection
            LegalAct(
                name="Consumer Protection Act",
                year=2019,
                short_name="CPA_2019",
                domain=LegalDomain.CONSUMER_PROTECTION,
                keywords=["consumer", "defective product", "consumer court", "refund", "warranty"],
                description="Protection of consumer rights and interests"
            ),
            LegalAct(
                name="Food Safety and Standards Act",
                year=2006,
                short_name="FSSA_2006",
                domain=LegalDomain.CONSUMER_PROTECTION,
                keywords=["food safety", "food quality", "restaurant", "food poisoning", "fssai"],
                description="Food safety and standards regulation"
            ),
            LegalAct(
                name="Information Technology Act (Consumer Provisions)",
                year=2000,
                short_name="IT_2000_Consumer",
                domain=LegalDomain.CONSUMER_PROTECTION,
                keywords=["cyber fraud", "online shopping", "digital payment", "e-commerce", "data protection"],
                description="Consumer protection in digital transactions"
            ),
            LegalAct(
                name="Real Estate (Regulation and Development) Act",
                year=2016,
                short_name="RERA_2016",
                domain=LegalDomain.CONSUMER_PROTECTION,
                keywords=["real estate", "builder", "property", "apartment", "rera", "possession delay"],
                description="Regulation of real estate sector"
            ),
            
            # Fundamental Rights
            LegalAct(
                name="Constitution of India (Fundamental Rights)",
                year=1950,
                short_name="Constitution_FR",
                domain=LegalDomain.FUNDAMENTAL_RIGHTS,
                keywords=["fundamental rights", "constitutional rights", "equality", "freedom", "discrimination"],
                description="Fundamental rights guaranteed by Constitution"
            ),
            LegalAct(
                name="Right to Information Act",
                year=2005,
                short_name="RTI_2005",
                domain=LegalDomain.FUNDAMENTAL_RIGHTS,
                keywords=["rti", "right to information", "government information", "transparency"],
                description="Right to access government information"
            ),
            LegalAct(
                name="Protection of Human Rights Act",
                year=1993,
                short_name="PHRA_1993",
                domain=LegalDomain.FUNDAMENTAL_RIGHTS,
                keywords=["human rights", "rights violation", "nhrc", "human rights commission"],
                description="Protection and promotion of human rights"
            ),
            
            # Employment & Labor Rights
            LegalAct(
                name="Industrial Disputes Act",
                year=1947,
                short_name="IDA_1947",
                domain=LegalDomain.EMPLOYMENT_LAW,
                keywords=["industrial dispute", "labor dispute", "strike", "lockout", "retrenchment"],
                description="Settlement of industrial disputes"
            ),
            LegalAct(
                name="Minimum Wages Act",
                year=1948,
                short_name="MWA_1948",
                domain=LegalDomain.EMPLOYMENT_LAW,
                keywords=["minimum wage", "wage", "salary", "daily wage", "overtime"],
                description="Minimum wages for workers"
            ),
            LegalAct(
                name="Equal Remuneration Act",
                year=1976,
                short_name="ERA_1976",
                domain=LegalDomain.EMPLOYMENT_LAW,
                keywords=["equal pay", "gender pay gap", "discrimination", "equal remuneration"],
                description="Equal remuneration for men and women"
            ),
            LegalAct(
                name="Sexual Harassment of Women at Workplace Act",
                year=2013,
                short_name="POSH_2013",
                domain=LegalDomain.EMPLOYMENT_LAW,
                keywords=["sexual harassment", "workplace harassment", "posh", "women safety", "icc"],
                description="Prevention of sexual harassment at workplace"
            )
        ]
    
    def _initialize_domain_keywords(self) -> Dict[LegalDomain, List[str]]:
        """Initialize domain-level keywords for classification"""
        return {
            LegalDomain.LEGAL_AID: [
                "legal aid", "free lawyer", "government lawyer", "legal assistance", 
                "pro bono", "cannot afford lawyer", "legal help", "poor", "below poverty line",
                "sc st", "scheduled caste", "scheduled tribe", "disability", "children",
                "juvenile", "minor", "women protection"
            ],
            LegalDomain.FAMILY_LAW: [
                "marriage", "divorce", "husband", "wife", "matrimonial", "dowry",
                "domestic violence", "maintenance", "alimony", "custody", "children",
                "inheritance", "succession", "property", "will", "hindu", "muslim",
                "christian", "parsi", "nikah", "talaq", "separation"
            ],
            LegalDomain.CONSUMER_PROTECTION: [
                "consumer", "product", "service", "defective", "warranty", "refund",
                "shop", "seller", "manufacturer", "quality", "fraud", "cheating",
                "food", "restaurant", "online", "e-commerce", "builder", "property",
                "real estate", "apartment", "possession", "delay"
            ],
            LegalDomain.FUNDAMENTAL_RIGHTS: [
                "fundamental rights", "constitutional rights", "discrimination", "equality",
                "freedom", "liberty", "religion", "speech", "assembly", "government",
                "police", "arrest", "detention", "rti", "information", "transparency",
                "human rights", "violation"
            ],
            LegalDomain.EMPLOYMENT_LAW: [
                "job", "work", "employee", "employer", "company", "office", "workplace",
                "salary", "wage", "termination", "firing", "dismissal", "harassment",
                "discrimination", "overtime", "leave", "maternity", "bonus", "pf",
                "provident fund", "esi", "labor", "worker", "union"
            ]
        }
    
    def get_acts_by_domain(self, domain: LegalDomain) -> List[LegalAct]:
        """Get all acts for a specific domain"""
        return [act for act in self.acts if act.domain == domain]
    
    def get_domain_keywords(self, domain: LegalDomain) -> List[str]:
        """Get keywords for a specific domain"""
        return self.domain_keywords.get(domain, [])
    
    def get_all_domains(self) -> List[LegalDomain]:
        """Get all supported domains"""
        return list(LegalDomain)
    
    def search_acts_by_keyword(self, keyword: str) -> List[LegalAct]:
        """Search acts by keyword"""
        keyword_lower = keyword.lower()
        matching_acts = []
        
        for act in self.acts:
            if any(keyword_lower in kw.lower() for kw in act.keywords):
                matching_acts.append(act)
        
        return matching_acts
