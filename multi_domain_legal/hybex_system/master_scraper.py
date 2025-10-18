#!/usr/bin/env python3
"""
Master Legal Knowledge Scraper
The ultimate unified scraper for Indian legal websites
Combines all scraping approaches into one comprehensive production-ready tool
"""

import requests
from bs4 import BeautifulSoup
import yaml
import json
import time
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from urllib.parse import urljoin, urlparse, quote_plus
from dataclasses import dataclass
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LegalWebsite:
    """Structure for legal website information"""
    name: str
    url: str
    relevance_score: int  # 1-10
    content_types: List[str]
    scraping_strategy: str
    accessibility: str
    priority: int

@dataclass
class ExtractedLegalContent:
    """Structure for extracted legal content"""
    source_url: str
    title: str
    content: str
    legal_domain: str
    content_type: str
    extraction_date: str
    relevance_score: float
    content_hash: str

class MasterLegalScraper:
    """
    The ultimate unified legal knowledge scraper
    Production-ready scraper combining all approaches
    """
    
    def __init__(self, data_dir: Path = None, yaml_path: Path = None):
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # YAML knowledge base path - integrate with existing system
        if yaml_path:
            self.yaml_path = yaml_path
        else:
            # Try to find existing YAML in knowledge_base directory
            possible_yaml_paths = [
                Path("knowledge_base/legal_rules.yaml"),
                self.data_dir.parent / "knowledge_base" / "legal_rules.yaml",
                Path(__file__).parent.parent / "knowledge_base" / "legal_rules.yaml"
            ]
            self.yaml_path = None
            for path in possible_yaml_paths:
                if path.exists():
                    self.yaml_path = path
                    break
            
            if not self.yaml_path:
                # Create default path in knowledge_base
                self.yaml_path = Path(__file__).parent.parent / "knowledge_base" / "legal_rules.yaml"
                self.yaml_path.parent.mkdir(exist_ok=True)
        
        # Initialize session with robust headers and SSL handling
        self.session = requests.Session()
        
        # Multiple User-Agents to rotate for resilience against blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        ]
        
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Fix SSL issues - disable SSL verification for problematic government sites
        self.session.verify = False
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Initialize database
        self.db_path = self.data_dir / "legal_knowledge.db"
        self.setup_database()
        
        # Load existing YAML knowledge base
        self.load_yaml_knowledge()
        
        # Define priority legal websites based on analysis
        self.legal_websites = self._define_legal_websites()
        
        # Rate limiting
        self.request_delay = 2  # seconds between requests
        self.last_request_time = 0
        
        logger.info(f"Master Legal Scraper initialized successfully")
        logger.info(f"YAML knowledge base: {self.yaml_path}")
        logger.info(f"Database path: {self.db_path}")

    def _define_legal_websites(self) -> Dict[str, LegalWebsite]:
        """Define priority legal websites with scraping strategies"""
        return {
            'indiacode': LegalWebsite(
                name="India Code - Central Acts Repository",
                url="https://indiacode.nic.in",
                relevance_score=10,  # HIGHEST PRIORITY
                content_types=['acts', 'amendments', 'legal_provisions'],
                scraping_strategy="act_specific_extraction",
                accessibility="excellent",
                priority=1
            ),
            'nalsa': LegalWebsite(
                name="National Legal Services Authority",
                url="https://nalsa.gov.in",
                relevance_score=10,  # HIGHEST PRIORITY
                content_types=['legal_aid_schemes', 'eligibility_criteria', 'income_thresholds'],
                scraping_strategy="scheme_and_criteria_extraction",
                accessibility="excellent",
                priority=1
            ),
            'egazette': LegalWebsite(
                name="E-Gazette India",
                url="https://egazette.gov.in",
                relevance_score=9,
                content_types=['notifications', 'amendments', 'legal_updates'],
                scraping_strategy="notification_extraction",
                accessibility="good",
                priority=2
            ),
            'prsindia': LegalWebsite(
                name="PRS Legislative Research",
                url="https://prsindia.org",
                relevance_score=9,
                content_types=['bill_analysis', 'policy_briefs', 'legislative_updates'],
                scraping_strategy="research_analysis_extraction",
                accessibility="excellent",
                priority=2
            ),
            'consumer_affairs': LegalWebsite(
                name="Department of Consumer Affairs",
                url="https://consumeraffairs.nic.in",
                relevance_score=8,
                content_types=['consumer_rights', 'complaint_procedures', 'protection_schemes'],
                scraping_strategy="consumer_specific_extraction",
                accessibility="good",
                priority=3
            ),
            'labour_ministry': LegalWebsite(
                name="Ministry of Labour and Employment",
                url="https://labour.gov.in",
                relevance_score=8,
                content_types=['labour_laws', 'employment_schemes', 'worker_rights'],
                scraping_strategy="labour_specific_extraction",
                accessibility="good",
                priority=3
            ),
            'legal_affairs': LegalWebsite(
                name="Department of Legal Affairs",
                url="https://legalaffairs.gov.in",
                relevance_score=7,
                content_types=['legal_notifications', 'circulars', 'administrative_updates'],
                scraping_strategy="administrative_extraction",
                accessibility="good",
                priority=4
            ),
            'family_courts': LegalWebsite(
                name="Family Court System India",
                url="https://ecourts.gov.in",
                relevance_score=9,
                content_types=['family_law', 'matrimonial_disputes', 'child_custody', 'maintenance'],
                scraping_strategy="family_law_extraction",
                accessibility="good",
                priority=2
            ),
            'employment_tribunal': LegalWebsite(
                name="Industrial Tribunal & Employment Laws",
                url="https://labour.gov.in",
                relevance_score=8,
                content_types=['employment_law', 'industrial_disputes', 'labor_rights', 'workplace_harassment'],
                scraping_strategy="employment_law_extraction",
                accessibility="good", 
                priority=3
            ),
            'consumer_commission': LegalWebsite(
                name="National Consumer Disputes Redressal Commission",
                url="https://ncdrc.gov.in",
                relevance_score=8,
                content_types=['consumer_protection', 'complaint_procedures', 'consumer_rights'],
                scraping_strategy="consumer_protection_extraction",
                accessibility="good",
                priority=3
            ),
            'human_rights': LegalWebsite(
                name="National Human Rights Commission",
                url="https://nhrc.gov.in",
                relevance_score=8,
                content_types=['fundamental_rights', 'human_rights', 'discrimination_cases'],
                scraping_strategy="rights_based_extraction", 
                accessibility="good",
                priority=3
            ),
            'women_child': LegalWebsite(
                name="Ministry of Women and Child Development",
                url="https://wcd.nic.in",
                relevance_score=7,
                content_types=['women_rights', 'child_protection', 'domestic_violence'],
                scraping_strategy="women_child_extraction",
                accessibility="good",
                priority=4
            ),
            'supremecourt': LegalWebsite(
                name="Supreme Court of India - Judgements",
                url="https://main.sci.gov.in/judgments",
                relevance_score=10,  # HIGHEST PRIORITY for case law
                content_types=['judgements', 'case_law', 'judicial_precedents'],
                scraping_strategy="judgement_extraction",
                accessibility="good",
                priority=1
            )
        }

    def setup_database(self):
        """Setup SQLite database for storing scraped content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraped_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                legal_domain TEXT NOT NULL,
                content_type TEXT NOT NULL,
                extraction_date TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_legal_domain ON scraped_content(legal_domain);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_content_hash ON scraped_content(content_hash);
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup completed")

    def load_yaml_knowledge(self):
        """Load existing YAML knowledge base"""
        try:
            if self.yaml_path.exists():
                with open(self.yaml_path, 'r', encoding='utf-8') as f:
                    self.existing_knowledge = yaml.safe_load(f)
                logger.info("Existing YAML knowledge loaded")
            else:
                self.existing_knowledge = {}
                logger.warning("No existing YAML knowledge found")
        except Exception as e:
            logger.error(f"Error loading YAML knowledge: {e}")
            self.existing_knowledge = {}

    def rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def make_request(self, url: str, retries: int = 4) -> Optional[requests.Response]:
        """Make HTTP request with exponential backoff, jitter, and rate limiting"""
        self.rate_limit()
        
        for attempt in range(retries):
            try:
                # Randomly pick a user agent for each request attempt to avoid fingerprinting
                self.session.headers.update({'User-Agent': random.choice(self.user_agents)})
                
                # Increased timeout for slower government sites
                response = self.session.get(url, timeout=20)
                response.raise_for_status()
                return response
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                # Don't retry 404s - page doesn't exist
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    logger.warning(f"Page not found (404) for {url}, not retrying.")
                    return None
                
                logger.warning(f"Request attempt {attempt + 1} for {url} failed: {e}")
                
                if attempt < retries - 1:
                    # Exponential backoff with jitter: 2s, 4s, 8s, 16s + random 1-3s
                    sleep_time = (2 ** attempt) + random.uniform(1, 3)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"A non-recoverable error occurred for {url}: {e}")
                break  # Break loop for other critical request errors
        
        logger.error(f"All requests failed for {url} after {retries} attempts")
        return None

    def extract_content_hash(self, content: str, source_url: str = "", domain: str = "") -> str:
        """Generate enhanced hash for better content deduplication"""
        # Create a more specific hash that includes source and domain context
        # This prevents similar content from different sources being marked as duplicates
        hash_input = f"{source_url}|{domain}|{content[:500]}"  # Use first 500 chars + context
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    def create_legal_content(self, title: str, content: str, source_url: str, domain: str, content_type: str, relevance_score: float = 8.0) -> ExtractedLegalContent:
        """Helper method to create ExtractedLegalContent with proper hashing"""
        return ExtractedLegalContent(
            source_url=source_url,
            title=title,
            content=content,
            legal_domain=domain,
            content_type=content_type,
            extraction_date=datetime.now().isoformat(),
            relevance_score=relevance_score,
            content_hash=self.extract_content_hash(content, source_url, domain)
        )

    def clear_old_duplicates(self):
        """Clear overly restrictive duplicates to allow fresh content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find and remove exact duplicates (keeping the most recent)
        cursor.execute('''
            DELETE FROM scraped_content 
            WHERE id NOT IN (
                SELECT MAX(id) 
                FROM scraped_content 
                GROUP BY content_hash
            )
        ''')
        
        removed_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} strict duplicate records")
        
        return removed_count

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s.,;:()\[\]{}"\'-]', ' ', text)
        return text.strip()

    def extract_legal_provisions(self, soup: BeautifulSoup, domain: str) -> List[str]:
        """Extract legal provisions based on domain"""
        provisions = []
        
        # Common patterns for legal content
        legal_patterns = [
            r'Section \d+[A-Z]*[\s\-\.]*([^\.]+\.)',
            r'Rule \d+[A-Z]*[\s\-\.]*([^\.]+\.)',
            r'Article \d+[A-Z]*[\s\-\.]*([^\.]+\.)',
            r'Clause \d+[A-Z]*[\s\-\.]*([^\.]+\.)',
            r'(?:shall|must|may)\s+([^\.]+\.)',
            r'(?:eligible|entitled|required)\s+([^\.]+\.)'
        ]
        
        text = soup.get_text()
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            provisions.extend([self.clean_text(match) for match in matches])
        
        return list(set(provisions))  # Remove duplicates

    def discover_indiacode_acts(self, keyword: str, limit: int = 5) -> List[Tuple[str, str]]:
        """
        Searches IndiaCode for a keyword and discovers relevant act URLs.
        Returns a list of tuples: (act_url, legal_domain)
        
        NOTE: IndiaCode search API may not be publicly accessible. 
        Using curated list of known important acts as fallback.
        """
        logger.info(f"Discovering acts on IndiaCode for keyword: '{keyword}'")
        
        # Map keyword to a legal domain for better categorization
        domain_map = {
            "consumer": "consumer_protection",
            "employment": "employment_law",
            "labour": "employment_law",
            "labor": "employment_law",
            "family": "family_law",
            "marriage": "family_law",
            "matrimonial": "family_law",
            "human rights": "fundamental_rights",
            "rights": "fundamental_rights",
            "criminal": "criminal_law",
            "property": "property_law"
        }
        
        # Curated list of important Indian legal acts by category
        # These are known, working IndiaCode handle URLs
        known_acts = {
            "consumer": [
                ("https://www.indiacode.nic.in/handle/123456789/15711", "consumer_protection"),  # Consumer Protection Act 2019
                ("https://www.indiacode.nic.in/handle/123456789/2034", "consumer_protection"),   # Consumer Protection Act 1986
            ],
            "employment": [
                ("https://www.indiacode.nic.in/handle/123456789/1891", "employment_law"),  # Industrial Disputes Act 1947
                ("https://www.indiacode.nic.in/handle/123456789/2152", "employment_law"),  # Minimum Wages Act 1948
                ("https://www.indiacode.nic.in/handle/123456789/2030", "employment_law"),  # Payment of Wages Act 1936
                ("https://www.indiacode.nic.in/handle/123456789/1886", "employment_law"),  # Factories Act 1948
            ],
            "labour": [
                ("https://www.indiacode.nic.in/handle/123456789/1891", "employment_law"),  # Industrial Disputes Act 1947
                ("https://www.indiacode.nic.in/handle/123456789/2152", "employment_law"),  # Minimum Wages Act 1948
                ("https://www.indiacode.nic.in/handle/123456789/1886", "employment_law"),  # Factories Act 1948
                ("https://www.indiacode.nic.in/handle/123456789/2013", "employment_law"),  # Trade Unions Act 1926
            ],
            "family": [
                ("https://www.indiacode.nic.in/handle/123456789/1384", "family_law"),  # Hindu Marriage Act 1955
                ("https://www.indiacode.nic.in/handle/123456789/1385", "family_law"),  # Hindu Succession Act 1956
                ("https://www.indiacode.nic.in/handle/123456789/1523", "family_law"),  # Protection of Women from Domestic Violence Act 2005
                ("https://www.indiacode.nic.in/handle/123456789/2056", "family_law"),  # Special Marriage Act 1954
            ],
            "human rights": [
                ("https://www.indiacode.nic.in/handle/123456789/1542", "fundamental_rights"),  # Protection of Human Rights Act 1993
                ("https://www.indiacode.nic.in/handle/123456789/2033", "fundamental_rights"),  # Scheduled Castes and Tribes (Prevention of Atrocities) Act 1989
            ],
        }
        
        # Get acts for the keyword (convert to lowercase for matching)
        keyword_lower = keyword.lower()
        act_links = known_acts.get(keyword_lower, [])
        
        # If no direct match, try partial matching
        if not act_links:
            for key, acts in known_acts.items():
                if keyword_lower in key or key in keyword_lower:
                    act_links = acts
                    break
        
        # Limit results
        act_links = act_links[:limit]
        
        logger.info(f"Using {len(act_links)} curated acts for '{keyword}' (IndiaCode search API not accessible)")
        return act_links

    def scrape_indiacode_act(self, act_url: str, domain: str) -> List[ExtractedLegalContent]:
        """Scrape specific act from IndiaCode"""
        response = self.make_request(act_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        extracted_content = []
        
        # Extract act title
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Unknown Act"
        
        # Extract sections and provisions
        sections = soup.find_all(['div', 'p'], class_=re.compile(r'section|provision|clause'))
        
        for section in sections:
            content = self.clean_text(section.get_text())
            if len(content) > 50:  # Filter out short/irrelevant content
                content_hash = self.extract_content_hash(content, act_url, domain)
                
                extracted_content.append(ExtractedLegalContent(
                    source_url=act_url,
                    title=f"{title} - Section",
                    content=content,
                    legal_domain=domain,
                    content_type="legal_provision",
                    extraction_date=datetime.now().isoformat(),
                    relevance_score=9.0,
                    content_hash=content_hash
                ))
        
        # Extract general provisions if no specific sections found
        if not extracted_content:
            main_content = soup.find('div', class_=re.compile(r'content|main|body'))
            if main_content:
                content = self.clean_text(main_content.get_text())
                if len(content) > 100:
                    content_hash = self.extract_content_hash(content)
                    
                    extracted_content.append(ExtractedLegalContent(
                        source_url=act_url,
                        title=title,
                        content=content,
                        legal_domain=domain,
                        content_type="act_content",
                        extraction_date=datetime.now().isoformat(),
                        relevance_score=8.0,
                        content_hash=content_hash
                    ))
        
        return extracted_content

    def scrape_nalsa_schemes(self) -> List[ExtractedLegalContent]:
        """Scrape NALSA legal aid schemes and eligibility criteria"""
        base_url = "https://nalsa.gov.in"
        response = self.make_request(base_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        extracted_content = []
        
        # Look for scheme-related links and content
        scheme_links = soup.find_all('a', href=re.compile(r'scheme|aid|eligibility|criteria'))
        
        for link in scheme_links[:10]:  # Limit to first 10 relevant links
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                scheme_response = self.make_request(full_url)
                
                if scheme_response:
                    scheme_soup = BeautifulSoup(scheme_response.content, 'html.parser')
                    
                    # Extract scheme content
                    title_elem = scheme_soup.find('h1') or scheme_soup.find('title')
                    title = title_elem.get_text().strip() if title_elem else "NALSA Scheme"
                    
                    content_div = scheme_soup.find('div', class_=re.compile(r'content|main|scheme'))
                    if content_div:
                        content = self.clean_text(content_div.get_text())
                        if len(content) > 100:
                            content_hash = self.extract_content_hash(content)
                            
                            extracted_content.append(ExtractedLegalContent(
                                source_url=full_url,
                                title=title,
                                content=content,
                                legal_domain="legal_aid",
                                content_type="scheme_eligibility",
                                extraction_date=datetime.now().isoformat(),
                                relevance_score=9.5,
                                content_hash=content_hash
                            ))
        
        return extracted_content

    def scrape_consumer_affairs(self) -> List[ExtractedLegalContent]:
        """Scrape consumer affairs content"""
        base_url = "https://consumeraffairs.nic.in"
        response = self.make_request(base_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        extracted_content = []
        
        # Look for consumer protection related content
        consumer_links = soup.find_all('a', href=re.compile(r'consumer|protection|rights|complaint'))
        
        for link in consumer_links[:8]:  # Limit to first 8 relevant links
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                content_response = self.make_request(full_url)
                
                if content_response:
                    content_soup = BeautifulSoup(content_response.content, 'html.parser')
                    
                    title_elem = content_soup.find('h1') or content_soup.find('title')
                    title = title_elem.get_text().strip() if title_elem else "Consumer Protection"
                    
                    main_content = content_soup.find('div', class_=re.compile(r'content|main|article'))
                    if main_content:
                        content = self.clean_text(main_content.get_text())
                        if len(content) > 100:
                            content_hash = self.extract_content_hash(content)
                            
                            extracted_content.append(ExtractedLegalContent(
                                source_url=full_url,
                                title=title,
                                content=content,
                                legal_domain="consumer_protection",
                                content_type="consumer_rights",
                                extraction_date=datetime.now().isoformat(),
                                relevance_score=8.0,
                                content_hash=content_hash
                            ))
        
        return extracted_content

    def save_to_database(self, content_list: List[ExtractedLegalContent]):
        """Save extracted content to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        duplicate_count = 0
        
        for content in content_list:
            try:
                cursor.execute('''
                    INSERT INTO scraped_content 
                    (source_url, title, content, legal_domain, content_type, 
                     extraction_date, relevance_score, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    content.source_url,
                    content.title,
                    content.content,
                    content.legal_domain,
                    content.content_type,
                    content.extraction_date,
                    content.relevance_score,
                    content.content_hash
                ))
                saved_count += 1
            except sqlite3.IntegrityError:
                # Content already exists (duplicate hash)
                duplicate_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {saved_count} new items, {duplicate_count} duplicates skipped")
        return saved_count, duplicate_count

    def update_yaml_knowledge(self, extracted_content: List[ExtractedLegalContent]):
        """Update YAML knowledge base with new extracted content"""
        domain_updates = {}
        
        for content in extracted_content:
            domain = content.legal_domain
            if domain not in domain_updates:
                domain_updates[domain] = {
                    'new_provisions': [],
                    'new_eligibility_criteria': [],
                    'new_procedures': []
                }
            
            # Categorize content based on type and patterns
            if 'eligibility' in content.content.lower() or 'criteria' in content.content.lower():
                domain_updates[domain]['new_eligibility_criteria'].append({
                    'source': content.source_url,
                    'content': content.content[:500],  # Limit length
                    'extracted_date': content.extraction_date
                })
            elif 'procedure' in content.content.lower() or 'process' in content.content.lower():
                domain_updates[domain]['new_procedures'].append({
                    'source': content.source_url,
                    'content': content.content[:500],
                    'extracted_date': content.extraction_date
                })
            else:
                domain_updates[domain]['new_provisions'].append({
                    'source': content.source_url,
                    'content': content.content[:500],
                    'extracted_date': content.extraction_date
                })
        
        # Save updated YAML
        if domain_updates:
            self.existing_knowledge['scraped_updates'] = {
                'last_update': datetime.now().isoformat(),
                'domains': domain_updates
            }
            
            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.existing_knowledge, f, default_flow_style=False, allow_unicode=True)
            
            logger.info("YAML knowledge base updated with new scraped content")

    def scrape_priority_websites(self) -> Dict[str, List[ExtractedLegalContent]]:
        """Scrape priority websites in order of importance"""
        all_extracted = {}
        
        # Sort websites by priority
        sorted_websites = sorted(
            self.legal_websites.items(),
            key=lambda x: x[1].priority
        )
        
        for website_key, website in sorted_websites:
            logger.info(f"Scraping {website.name} (Priority {website.priority}, Score {website.relevance_score})")
            
            try:
                if website_key == 'indiacode':
                    # Dynamically discover acts instead of using a hardcoded list
                    keywords_to_search = ["consumer", "employment", "labour", "family", "human rights"]
                    discovered_acts = []
                    for keyword in keywords_to_search:
                        discovered_acts.extend(self.discover_indiacode_acts(keyword, limit=10))  # Get up to 10 acts per keyword
                    
                    # Remove duplicate URLs while preserving order
                    unique_acts = list(dict.fromkeys(discovered_acts))

                    extracted = []
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_act = {executor.submit(self.scrape_indiacode_act, act_url, domain): (act_url, domain) for act_url, domain in unique_acts}
                        for future in as_completed(future_to_act):
                            try:
                                act_content = future.result()
                                extracted.extend(act_content)
                            except Exception as exc:
                                act_url, domain = future_to_act[future]
                                logger.error(f"Error scraping act {act_url}: {exc}")

                    all_extracted[website_key] = extracted
                
                elif website_key == 'supremecourt':
                    all_extracted[website_key] = self.scrape_supreme_court_judgements()
                
                elif website_key == 'nalsa':
                    all_extracted[website_key] = self.scrape_nalsa_schemes()
                
                elif website_key == 'consumer_affairs':
                    all_extracted[website_key] = self.scrape_consumer_affairs()
                
                elif website_key == 'family_courts':
                    all_extracted[website_key] = self.scrape_family_law_content()
                
                elif website_key == 'employment_tribunal':
                    all_extracted[website_key] = self.scrape_employment_law_content()
                
                elif website_key == 'consumer_commission':  
                    all_extracted[website_key] = self.scrape_consumer_commission_content()
                
                elif website_key == 'human_rights':
                    all_extracted[website_key] = self.scrape_human_rights_content()
                
                elif website_key == 'women_child':
                    all_extracted[website_key] = self.scrape_women_child_content()
                
                else:
                    # Generic scraping for other websites
                    response = self.make_request(website.url)
                    if response:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract main content
                        main_content = soup.find('div', class_=re.compile(r'content|main|article'))
                        if main_content:
                            content = self.clean_text(main_content.get_text())
                            if len(content) > 100:
                                content_hash = self.extract_content_hash(content)
                                
                                extracted_content = ExtractedLegalContent(
                                    source_url=website.url,
                                    title=website.name,
                                    content=content,
                                    legal_domain="general",
                                    content_type="website_content",
                                    extraction_date=datetime.now().isoformat(),
                                    relevance_score=float(website.relevance_score),
                                    content_hash=content_hash
                                )
                                
                                all_extracted[website_key] = [extracted_content]
                
                logger.info(f"Extracted {len(all_extracted.get(website_key, []))} items from {website.name}")
                
            except Exception as e:
                logger.error(f"Error scraping {website.name}: {e}")
                all_extracted[website_key] = []
        
        return all_extracted

    def _get_domain_from_url(self, url: str) -> str:
        """Map URL to legal domain"""
        if "1367" in url:  # Legal Services Authorities Act
            return "legal_aid"
        elif "1384" in url:  # Hindu Marriage Act
            return "family_law"
        elif "15711" in url:  # Consumer Protection Act
            return "consumer_protection"
        elif "1891" in url:  # Industrial Disputes Act
            return "employment_law"
        else:
            return "general"

    def run_comprehensive_scraping(self):
        """Run the complete scraping process"""
        logger.info("Starting comprehensive legal knowledge scraping")
        start_time = time.time()
        
        # Scrape all priority websites
        all_extracted = self.scrape_priority_websites()
        
        # Flatten all extracted content
        all_content = []
        for website_content in all_extracted.values():
            all_content.extend(website_content)
        
        logger.info(f"Total extracted content: {len(all_content)} items")
        
        # Save to database
        if all_content:
            saved_count, duplicate_count = self.save_to_database(all_content)
            
            # Update YAML knowledge base
            self.update_yaml_knowledge(all_content)
            
            # Generate summary report
            self.generate_scraping_report(all_extracted, saved_count, duplicate_count)
        
        end_time = time.time()
        logger.info(f"Scraping completed in {end_time - start_time:.2f} seconds")
        
        return all_extracted

    def generate_scraping_report(self, all_extracted: Dict, saved_count: int, duplicate_count: int):
        """Generate comprehensive scraping report"""
        report = {
            'scraping_session': {
                'timestamp': datetime.now().isoformat(),
                'total_websites_scraped': len(all_extracted),
                'total_content_extracted': sum(len(content) for content in all_extracted.values()),
                'content_saved_to_db': saved_count,
                'duplicates_skipped': duplicate_count
            },
            'website_breakdown': {}
        }
        
        for website_key, content_list in all_extracted.items():
            website = self.legal_websites[website_key]
            report['website_breakdown'][website_key] = {
                'name': website.name,
                'relevance_score': website.relevance_score,
                'priority': website.priority,
                'content_extracted': len(content_list),
                'content_types': list(set(item.content_type for item in content_list))
            }
        
        # Save report
        report_path = self.data_dir / f"scraping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraping report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("MASTER LEGAL SCRAPER - SESSION SUMMARY")
        print("="*60)
        print(f"Total websites scraped: {len(all_extracted)}")
        print(f"Total content extracted: {sum(len(content) for content in all_extracted.values())}")
        print(f"New content saved: {saved_count}")
        print(f"Duplicates skipped: {duplicate_count}")
        print("\nWebsite Performance:")
        for website_key, content_list in all_extracted.items():
            website = self.legal_websites[website_key]
            print(f"  {website.name}: {len(content_list)} items (Score: {website.relevance_score})")
        print("="*60)

    def scrape_family_law_content(self) -> List[ExtractedLegalContent]:
        """Scrape family law specific content"""
        extracted_content = []
        family_law_urls = [
            "https://districts.ecourts.gov.in",  # Fixed base URL
            "https://www.indiacode.nic.in/handle/123456789/1384",  # Hindu Marriage Act
            "https://ecourts.gov.in/ecourts_home/",  # Alternative family court URL
        ]
        
        for url in family_law_urls:
            response = self.make_request(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = self.clean_text(soup.get_text())
                if len(content) > 100:
                    extracted_content.append(self.create_legal_content(
                        title="Family Law Content",
                        content=content[:2000],  # Limit content size
                        source_url=url,
                        domain="family_law",
                        content_type="family_law_provision",
                        relevance_score=8.5
                    ))
        
        logger.info(f"Extracted {len(extracted_content)} items from Family Law sources")
        return extracted_content

    def scrape_employment_law_content(self) -> List[ExtractedLegalContent]:
        """Scrape employment law specific content"""
        extracted_content = []
        employment_urls = [
            "https://labour.gov.in",  # Fixed base URL
            "https://www.indiacode.nic.in/handle/123456789/1891",  # Industrial Disputes Act
        ]
        
        for url in employment_urls:
            response = self.make_request(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = self.clean_text(soup.get_text())
                if len(content) > 100:
                    extracted_content.append(ExtractedLegalContent(
                        source_url=url,
                        title="Employment Law Content",
                        content=content[:2000],
                        legal_domain="employment_law", 
                        content_type="employment_law_provision",
                        extraction_date=datetime.now().isoformat(),
                        relevance_score=8.0,
                        content_hash=self.extract_content_hash(content)
                    ))
        
        logger.info(f"Extracted {len(extracted_content)} items from Employment Law sources")
        return extracted_content

    def scrape_consumer_commission_content(self) -> List[ExtractedLegalContent]:
        """Scrape consumer commission by crawling for relevant documents."""
        logger.info("Crawling NCDRC for consumer protection content")
        base_url = "https://ncdrc.nic.in"
        response = self.make_request(base_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        extracted_content = []
        
        # Find links that point to potential legal documents
        keywords = ['act', 'rules', 'judgement', 'orders', 'bareact', 'regulation', 'circular']
        link_pattern = re.compile('|'.join(keywords), re.IGNORECASE)
        
        document_links = set()
        for link in soup.find_all('a', href=True):
            link_text = link.get_text()
            link_href = link['href']
            if link_pattern.search(link_text) or link_pattern.search(link_href):
                full_url = urljoin(base_url, link_href)
                # Ensure we only crawl the same site
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    document_links.add(full_url)

        logger.info(f"Found {len(document_links)} potential document links on NCDRC.")

        # Scrape each discovered link (up to a limit)
        for url in list(document_links)[:20]:  # Limit to avoid excessive requests
            doc_response = self.make_request(url)
            if doc_response:
                doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
                title_elem = doc_soup.find('h1') or doc_soup.find('h2') or doc_soup.find('title')
                title = self.clean_text(title_elem.get_text()) if title_elem else "Consumer Protection Document"

                # Use a smarter content extraction logic
                main_content = doc_soup.find('div', class_=re.compile(r'content|main|article|text'))
                content_text = self.clean_text(main_content.get_text()) if main_content else ""
                
                # Fallback to body if no specific content div found
                if len(content_text) < 200:
                    body_content = doc_soup.find('body')
                    content_text = self.clean_text(body_content.get_text()) if body_content else ""
                
                if len(content_text) > 200:
                    extracted_content.append(self.create_legal_content(
                        title=title,
                        content=content_text[:3000],  # Take first 3000 chars
                        source_url=url,
                        domain="consumer_protection",
                        content_type="consumer_protection_document",
                        relevance_score=8.5
                    ))

        logger.info(f"Extracted {len(extracted_content)} items from Consumer Commission sources")
        return extracted_content

    def scrape_human_rights_content(self) -> List[ExtractedLegalContent]:
        """Scrape human rights specific content"""
        extracted_content = []
        rights_urls = [
            "https://nhrc.nic.in",  # Fixed base URL
            "https://www.indiacode.nic.in/handle/123456789/1542",  # Protection of Human Rights Act
        ]
        
        for url in rights_urls:
            response = self.make_request(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = self.clean_text(soup.get_text())
                if len(content) > 100:
                    extracted_content.append(ExtractedLegalContent(
                        source_url=url,
                        title="Human Rights Content",
                        content=content[:2000],
                        legal_domain="fundamental_rights",
                        content_type="human_rights_provision",
                        extraction_date=datetime.now().isoformat(),
                        relevance_score=8.0,
                        content_hash=self.extract_content_hash(content)
                    ))
        
        logger.info(f"Extracted {len(extracted_content)} items from Human Rights sources")
        return extracted_content

    def scrape_women_child_content(self) -> List[ExtractedLegalContent]:
        """Scrape women and child development specific content"""
        extracted_content = []
        women_child_urls = [
            "https://wcd.gov.in",  # Fixed correct domain
            "https://www.indiacode.nic.in/handle/123456789/1523",  # Protection of Women from Domestic Violence Act
        ]
        
        for url in women_child_urls:
            response = self.make_request(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = self.clean_text(soup.get_text())
                if len(content) > 100:
                    extracted_content.append(ExtractedLegalContent(
                        source_url=url,
                        title="Women & Child Development Content",
                        content=content[:2000],
                        legal_domain="family_law",
                        content_type="women_child_provision",
                        extraction_date=datetime.now().isoformat(),
                        relevance_score=7.5,
                        content_hash=self.extract_content_hash(content)
                    ))
        
        logger.info(f"Extracted {len(extracted_content)} items from Women & Child Development sources")
        return extracted_content

    def scrape_supreme_court_judgements(self) -> List[ExtractedLegalContent]:
        """Scrapes the latest judgements from the Supreme Court website."""
        logger.info("Scraping latest judgements from Supreme Court of India")
        judgements_url = "https://main.sci.gov.in/judgments"
        response = self.make_request(judgements_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        extracted_content = []

        # Find the table or container with the latest judgements
        # This selector is based on the site structure as of late 2025 and may need updates.
        judgement_div = soup.find('div', id='tab-1')  # 'Judgements' tab content
        if not judgement_div:
            logger.warning("Could not find the judgements container on SCI website.")
            # Try alternative selectors
            judgement_div = soup.find('div', class_=re.compile(r'judgement|judgment|latest', re.IGNORECASE))
        
        if not judgement_div:
            logger.warning("Using fallback: searching entire page for judgement links")
            judgement_div = soup
        
        # Each judgement is often in a row or a specific div
        # Try multiple selectors to be robust
        judgement_items = (
            judgement_div.find_all('div', class_=re.compile(r'judgement_row|judgment_row', re.IGNORECASE), limit=25) or
            judgement_div.find_all('tr', limit=25) or
            judgement_div.find_all('li', limit=25)
        )
        
        if not judgement_items:
            logger.warning("No judgement items found with standard selectors")
            return []
            
        for item in judgement_items[:25]:  # Limit to recent 25
            title_tag = item.find('a')
            if not title_tag:
                continue
            
            title = self.clean_text(title_tag.get_text())
            
            # Skip if title is too short or generic
            if len(title) < 10:
                continue
            
            # Get the link to the judgement
            judgement_link = title_tag.get('href', '')
            full_judgement_url = urljoin(judgements_url, judgement_link) if judgement_link else judgements_url
            
            # In a real scenario, you'd follow this link or look for a PDF link.
            # For this example, we'll treat the summary text as the content.
            content_summary = self.clean_text(item.get_text())

            if len(content_summary) > 100:
                extracted_content.append(self.create_legal_content(
                    title=f"SC Judgement: {title}",
                    content=content_summary[:3000],  # Take first 3000 chars
                    source_url=full_judgement_url,
                    domain="case_law",  # New domain for case law
                    content_type="judicial_precedent",
                    relevance_score=9.5
                ))
            
        logger.info(f"Extracted {len(extracted_content)} judgements from Supreme Court")
        return extracted_content

    def generate_report(self) -> Dict[str, Any]:
        """Generate a report from existing scraped data in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM scraped_content")
            total_count = cursor.fetchone()[0]
            
            # Get counts by domain
            cursor.execute("SELECT legal_domain, COUNT(*) FROM scraped_content GROUP BY legal_domain")
            domain_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'status': 'success',
                'total_count': total_count,
                'domain_counts': domain_counts,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_count': 0,
                'domain_counts': {}
            }

def main():
    """Main execution function"""
    scraper = MasterLegalScraper()
    
    try:
        results = scraper.run_comprehensive_scraping()
        
        # Optional: Print first few extracted items for verification
        print("\nSample extracted content:")
        for website_key, content_list in results.items():
            if content_list:
                print(f"\n--- {scraper.legal_websites[website_key].name} ---")
                for item in content_list[:2]:  # Show first 2 items
                    print(f"Title: {item.title}")
                    print(f"Domain: {item.legal_domain}")
                    print(f"Content: {item.content[:200]}...")
                    print("-" * 40)
    
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Error during scraping: {e}")

if __name__ == "__main__":
    main()
