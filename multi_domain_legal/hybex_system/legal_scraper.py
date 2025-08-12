# hybex_system/legal_scraper.py

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from lxml import html

from .config import HybExConfig

# Setup logging
logger = logging.getLogger(__name__)

class LegalDataScraper:
    """
    A comprehensive scraper to gather legal information from various online sources.
    This version combines multiple scraping sources for a more robust knowledge base.
    """
    def __init__(self, config: HybExConfig):
        self.config = config
        self.db_path = self.config.DATA_DIR / "legal_knowledge.db"
        self.setup_database()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HybEx-LawBot/1.0 (https://github.com/your-username/your-repo)',
        })
        self.setup_logging()
        logger.info("LegalDataScraper initialized.")

    def setup_logging(self):
        log_file = self.config.get_log_path('legal_scraper')
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('legal_scraper.log') for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(self.config.LOGGING_CONFIG['format'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Added file handler to LegalDataScraper logger.")

    def setup_database(self):
        """Initializes the SQLite database and table for storing scraped data."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    rule_text TEXT NOT NULL,
                    extracted_on TEXT NOT NULL,
                    metadata JSON
                )
            """)
            conn.commit()
            conn.close()
            logger.info("Legal knowledge database initialized or already exists.")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")

    def save_to_db(self, data: Dict[str, Any]):
        """Saves a single data entry to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO legal_knowledge (source, url, domain, rule_text, extracted_on, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data['source'],
                data['url'],
                data['domain'],
                data['rule_text'],
                data['extracted_on'],
                json.dumps(data.get('metadata', {}))
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error saving data to database: {e}")

    def fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetches content from a URL with error handling."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully fetched URL: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def scrape_nalsa(self, max_articles: int = 5):
        """Scrapes legal aid information from the NALSA website."""
        url = "https://nalsa.gov.in/legal-aid-schemes"
        response = self.fetch_url(url)
        if not response:
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        article_links = soup.select('.views-row a')[:max_articles]
        
        for link in article_links:
            full_url = "https://nalsa.gov.in" + link.get('href')
            article_response = self.fetch_url(full_url)
            if article_response:
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                title = article_soup.select_one('h1.page-title').text.strip()
                content = " ".join([p.text.strip() for p in article_soup.select('.content p')])
                
                if content:
                    self.save_to_db({
                        'source': 'NALSA',
                        'url': full_url,
                        'domain': 'Legal Aid',
                        'rule_text': content,
                        'extracted_on': datetime.now().isoformat(),
                        'metadata': {'title': title}
                    })
                    logger.info(f"Scraped and saved: '{title}' from NALSA.")

    def scrape_doj_rules(self):
        """Scrapes rules and articles from the Department of Justice website."""
        urls = [
            'https://doj.gov.in/legal-aid-schemes',
            'https://doj.gov.in/initiatives'
        ]
        
        for url in urls:
            response = self.fetch_url(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('article')
                if not articles:
                    articles = soup.find_all('div', class_='views-row')

                for article in articles:
                    title_tag = article.find(['h1', 'h2', 'h3', 'a'])
                    title = title_tag.text.strip() if title_tag else 'No Title'
                    
                    content_tag = article.find(class_='content')
                    content = content_tag.text.strip() if content_tag else ''

                    if content:
                        self.save_to_db({
                            'source': 'Department of Justice',
                            'url': url,
                            'domain': 'Legal Aid',
                            'rule_text': content,
                            'extracted_on': datetime.now().isoformat(),
                            'metadata': {'title': title}
                        })
                        logger.info(f"Scraped and saved: '{title}' from DoJ.")

    def scrape_indiacode(self, act_name: str = "Legal Services Authorities Act, 1987"):
        """Scrapes specific Acts from India Code (legislative data)."""
        # Note: This is a placeholder as India Code is complex to scrape
        # A proper implementation would require parsing the site structure
        logger.info(f"Scraping placeholder for '{act_name}' from India Code.")
        
        # Simulating a scrape for now
        self.save_to_db({
            'source': 'India Code',
            'url': 'https://www.indiacode.nic.in/handle/123456789/2281',
            'domain': 'Legal Aid',
            'rule_text': f"Scraped text for {act_name}, Section 12: Eligibility criteria for receiving legal services...",
            'extracted_on': datetime.now().isoformat(),
            'metadata': {'act': act_name, 'section': '12'}
        })

    def run_scraper(self):
        """Runs the complete scraping pipeline."""
        logger.info("Starting complete legal data scraping pipeline...")
        self.scrape_nalsa()
        self.scrape_doj_rules()
        self.scrape_indiacode()
        logger.info("Legal data scraping pipeline finished.")

if __name__ == '__main__':
    # This block is for testing the scraper independently.
    # A full run would be orchestrated by the main system script.
    config = HybExConfig()
    scraper = LegalDataScraper(config)
    scraper.run_scraper()