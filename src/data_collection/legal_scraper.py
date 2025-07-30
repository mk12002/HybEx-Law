"""
Legal Document Scraper for HybEx-Law

This module provides tools for collecting legal documents and cases
from public sources while respecting privacy and legal requirements.
"""

import requests
import time
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import logging


@dataclass
class LegalDocument:
    """Structure for legal document metadata."""
    title: str
    content: str
    source: str
    date: Optional[str] = None
    case_type: Optional[str] = None
    outcome: Optional[str] = None
    url: Optional[str] = None


class LegalDocumentScraper:
    """
    Scraper for collecting legal documents from public sources.
    
    Focuses on anonymized case studies, legal aid guidelines,
    and publicly available legal information.
    """
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize the scraper.
        
        Args:
            delay: Delay between requests to be respectful
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Legal data sources (public domains only)
        self.sources = {
            'nalsa': {
                'base_url': 'https://nalsa.gov.in',
                'endpoints': {
                    'schemes': '/content/legal-aid-schemes',
                    'guidelines': '/content/guidelines',
                    'annual_reports': '/content/annual-reports'
                }
            },
            'sci': {
                'base_url': 'https://main.sci.gov.in',
                'endpoints': {
                    'judgments': '/judgments'
                }
            },
            'legislative': {
                'base_url': 'https://legislative.gov.in',
                'endpoints': {
                    'acts': '/acts-of-parliament'
                }
            }
        }
    
    def scrape_nalsa_guidelines(self) -> List[LegalDocument]:
        """
        Scrape NALSA legal aid guidelines and scheme documents.
        
        Returns:
            List of legal documents with guidelines
        """
        documents = []
        
        try:
            # Get main guidelines page
            guidelines_url = self.sources['nalsa']['base_url'] + self.sources['nalsa']['endpoints']['guidelines']
            response = self.session.get(guidelines_url)
            
            if response.status_code == 200:
                # Parse guidelines content
                content = response.text
                
                # Extract relevant sections
                guidelines_sections = self._extract_guidelines_sections(content)
                
                for section in guidelines_sections:
                    doc = LegalDocument(
                        title=section['title'],
                        content=section['content'],
                        source='NALSA Guidelines',
                        url=guidelines_url
                    )
                    documents.append(doc)
            
            self._respectful_delay()
            
        except Exception as e:
            self.logger.error(f"Error scraping NALSA guidelines: {e}")
        
        return documents
    
    def _extract_guidelines_sections(self, html_content: str) -> List[Dict[str, str]]:
        """
        Extract structured sections from NALSA guidelines.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # This is a simplified example - real implementation would need
        # proper HTML parsing with BeautifulSoup
        
        # Extract eligibility criteria sections
        eligibility_patterns = [
            r'eligibility\s+criteria.*?(?=\n\n|\n[A-Z])',
            r'income\s+limit.*?(?=\n\n|\n[A-Z])',
            r'categorical\s+eligibility.*?(?=\n\n|\n[A-Z])'
        ]
        
        for pattern in eligibility_patterns:
            matches = re.finditer(pattern, html_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                sections.append({
                    'title': f"Eligibility Criteria Section",
                    'content': match.group(0).strip()
                })
        
        return sections
    
    def scrape_case_studies(self, num_cases: int = 50) -> List[LegalDocument]:
        """
        Scrape anonymized case studies from legal databases.
        
        Args:
            num_cases: Number of cases to collect
            
        Returns:
            List of case study documents
        """
        case_studies = []
        
        # Example sources for case studies (would need real implementation)
        case_study_sources = [
            'https://example-legal-aid-cases.org/cases',  # Placeholder
            'https://anonymized-legal-cases.edu/database'  # Placeholder
        ]
        
        # This is a template - real implementation would require
        # actual legal databases with public access
        
        for source in case_study_sources:
            try:
                # Simulated case extraction
                cases = self._extract_anonymized_cases(source, num_cases // len(case_study_sources))
                case_studies.extend(cases)
                
            except Exception as e:
                self.logger.error(f"Error scraping cases from {source}: {e}")
        
        return case_studies
    
    def _extract_anonymized_cases(self, source_url: str, num_cases: int) -> List[LegalDocument]:
        """
        Extract anonymized case studies from a source.
        
        Args:
            source_url: URL of the case database
            num_cases: Number of cases to extract
            
        Returns:
            List of case documents
        """
        cases = []
        
        # Placeholder implementation
        # Real implementation would:
        # 1. Access legal databases with proper permissions
        # 2. Extract case details while preserving anonymity
        # 3. Structure the data for training
        
        for i in range(num_cases):
            # Simulate case extraction
            case = LegalDocument(
                title=f"Anonymized Case Study {i+1}",
                content=f"Legal aid case involving [redacted details] with outcome [redacted]",
                source=source_url,
                case_type="property_dispute",  # Would be determined from content
                outcome="eligible"  # Would be extracted from case
            )
            cases.append(case)
        
        return cases
    
    def extract_legal_patterns(self, documents: List[LegalDocument]) -> Dict[str, Any]:
        """
        Extract common patterns from legal documents.
        
        Args:
            documents: List of legal documents
            
        Returns:
            Dictionary of extracted patterns
        """
        patterns = {
            'income_expressions': set(),
            'case_type_keywords': {},
            'eligibility_phrases': set(),
            'social_category_mentions': set()
        }
        
        for doc in documents:
            content = doc.content.lower()
            
            # Extract income-related expressions
            income_matches = re.findall(
                r'(?:income|salary|earn|revenue).*?(?:rupees?|rs\.?|\d+)',
                content
            )
            patterns['income_expressions'].update(income_matches)
            
            # Extract case type keywords
            case_keywords = re.findall(
                r'(?:property|domestic|labor|criminal|accident|defamation).*?(?:dispute|case|matter)',
                content
            )
            for keyword in case_keywords:
                case_type = self._classify_case_type(keyword)
                if case_type:
                    if case_type not in patterns['case_type_keywords']:
                        patterns['case_type_keywords'][case_type] = set()
                    patterns['case_type_keywords'][case_type].add(keyword)
            
            # Extract eligibility phrases
            eligibility_matches = re.findall(
                r'(?:eligible|qualify|entitled).*?(?:legal aid|assistance|lawyer)',
                content
            )
            patterns['eligibility_phrases'].update(eligibility_matches)
            
            # Extract social category mentions
            social_matches = re.findall(
                r'(?:woman|child|scheduled caste|scheduled tribe|disabled|industrial worker)',
                content
            )
            patterns['social_category_mentions'].update(social_matches)
        
        # Convert sets to lists for JSON serialization
        for key in patterns:
            if isinstance(patterns[key], set):
                patterns[key] = list(patterns[key])
            elif isinstance(patterns[key], dict):
                for subkey in patterns[key]:
                    if isinstance(patterns[key][subkey], set):
                        patterns[key][subkey] = list(patterns[key][subkey])
        
        return patterns
    
    def _classify_case_type(self, keyword: str) -> Optional[str]:
        """Classify case type from keyword."""
        keyword_lower = keyword.lower()
        
        if any(word in keyword_lower for word in ['property', 'landlord', 'tenant', 'evict']):
            return 'property_dispute'
        elif any(word in keyword_lower for word in ['domestic', 'marriage', 'divorce', 'family']):
            return 'family_matter'
        elif any(word in keyword_lower for word in ['labor', 'employment', 'worker', 'salary']):
            return 'labor_dispute'
        elif any(word in keyword_lower for word in ['criminal', 'arrest', 'police', 'theft']):
            return 'criminal_matter'
        elif any(word in keyword_lower for word in ['accident', 'compensation', 'insurance']):
            return 'accident_compensation'
        elif any(word in keyword_lower for word in ['defamation', 'reputation', 'character']):
            return 'defamation'
        elif any(word in keyword_lower for word in ['business', 'commercial', 'contract']):
            return 'business_dispute'
        
        return None
    
    def _respectful_delay(self):
        """Add delay between requests."""
        time.sleep(self.delay)
    
    def save_documents(self, documents: List[LegalDocument], output_dir: str):
        """
        Save scraped documents to files.
        
        Args:
            documents: List of documents to save
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, doc in enumerate(documents):
            filename = output_path / f"legal_doc_{i+1:03d}.json"
            
            doc_dict = {
                'title': doc.title,
                'content': doc.content,
                'source': doc.source,
                'date': doc.date,
                'case_type': doc.case_type,
                'outcome': doc.outcome,
                'url': doc.url
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(documents)} documents to {output_dir}")


class EthicalDataCollector:
    """
    Ethical data collection with privacy protection and legal compliance.
    """
    
    def __init__(self):
        """Initialize ethical data collector."""
        self.privacy_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\b\d{10,12}\b',  # Phone numbers
            r'\b[A-Z]{2}\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\b',  # PAN numbers
            r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Card numbers
        ]
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize sensitive information in text.
        
        Args:
            text: Original text
            
        Returns:
            Anonymized text
        """
        anonymized = text
        
        # Replace names with placeholders
        anonymized = re.sub(self.privacy_patterns[0], '[NAME]', anonymized)
        
        # Replace phone numbers
        anonymized = re.sub(self.privacy_patterns[1], '[PHONE]', anonymized)
        
        # Replace PAN numbers
        anonymized = re.sub(self.privacy_patterns[2], '[PAN]', anonymized)
        
        # Replace card numbers
        anonymized = re.sub(self.privacy_patterns[3], '[CARD]', anonymized)
        
        # Replace specific locations with generic ones
        location_replacements = {
            r'\b[A-Z][a-z]+\s+(?:Street|Road|Avenue|Colony)\b': '[ADDRESS]',
            r'\b[A-Z][a-z]+\s+(?:District|City)\b': '[LOCATION]'
        }
        
        for pattern, replacement in location_replacements.items():
            anonymized = re.sub(pattern, replacement, anonymized)
        
        return anonymized
    
    def validate_legal_compliance(self, data_collection_plan: Dict[str, Any]) -> List[str]:
        """
        Validate data collection plan for legal compliance.
        
        Args:
            data_collection_plan: Plan for data collection
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        # Check for privacy protection
        if 'privacy_protection' not in data_collection_plan:
            issues.append("Missing privacy protection measures")
        
        # Check for consent mechanisms
        if 'consent_mechanism' not in data_collection_plan:
            issues.append("Missing consent mechanism for data collection")
        
        # Check for data retention policy
        if 'data_retention' not in data_collection_plan:
            issues.append("Missing data retention and deletion policy")
        
        # Check for anonymization procedures
        if 'anonymization' not in data_collection_plan:
            issues.append("Missing anonymization procedures")
        
        return issues


def main():
    """Demonstrate legal data collection."""
    print("⚖️  Legal Document Scraper Demo")
    print("=" * 40)
    
    # Initialize scraper
    scraper = LegalDocumentScraper()
    
    print("📋 Note: This demo shows the framework.")
    print("Real implementation requires:")
    print("- Proper permissions for data access")
    print("- Legal compliance review")
    print("- Privacy protection measures")
    print("- Ethical approval for research use")
    
    # Demonstrate ethical considerations
    ethical_collector = EthicalDataCollector()
    
    sample_text = "John Doe from Mumbai called 9876543210 about his case"
    anonymized = ethical_collector.anonymize_text(sample_text)
    
    print(f"\n🔒 Privacy Protection Demo:")
    print(f"Original: {sample_text}")
    print(f"Anonymized: {anonymized}")
    
    # Validate collection plan
    collection_plan = {
        'source': 'public_legal_databases',
        'purpose': 'research_and_development',
        'privacy_protection': 'anonymization',
        'consent_mechanism': 'opt_in_only'
    }
    
    issues = ethical_collector.validate_legal_compliance(collection_plan)
    
    print(f"\n✅ Compliance Check:")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No compliance issues found")


if __name__ == "__main__":
    main()
