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
from urllib.parse import urljoin, urlparse
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
        
        # Initialize session with robust headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
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

    def make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retries and rate limiting"""
        self.rate_limit()
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(1, 3))
                else:
                    logger.error(f"All requests failed for {url}")
                    return None

    def extract_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

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
                content_hash = self.extract_content_hash(content)
                
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
                    # Scrape specific acts from IndiaCode
                    act_urls = [
                        "https://www.indiacode.nic.in/handle/123456789/1367",  # Legal Services Authorities Act
                        "https://www.indiacode.nic.in/handle/123456789/1384",  # Hindu Marriage Act
                        "https://www.indiacode.nic.in/handle/123456789/15711", # Consumer Protection Act 2019
                        "https://www.indiacode.nic.in/handle/123456789/1891",  # Industrial Disputes Act
                    ]
                    
                    extracted = []
                    for act_url in act_urls:
                        domain = self._get_domain_from_url(act_url)
                        act_content = self.scrape_indiacode_act(act_url, domain)
                        extracted.extend(act_content)
                    
                    all_extracted[website_key] = extracted
                
                elif website_key == 'nalsa':
                    all_extracted[website_key] = self.scrape_nalsa_schemes()
                
                elif website_key == 'consumer_affairs':
                    all_extracted[website_key] = self.scrape_consumer_affairs()
                
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
