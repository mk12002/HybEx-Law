# hybex_system/legal_scraper.py

import sqlite3
import requests
from bs4 import BeautifulSoup
import logging
import json
from pathlib import Path
from datetime import datetime
import re # Add this
from typing import Dict, Any, Optional

from .config import HybExConfig

logger = logging.getLogger(__name__)

class LegalDataScraper:
    def __init__(self, config: HybExConfig):
        self.config = config
        self.db_path = config.DATA_DIR / "legal_knowledge.db"
        self._init_database()
        logger.info(f"LegalDataScraper initialized. Database: {self.db_path}")

    def _init_database(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS legal_knowledge (
                key TEXT PRIMARY KEY,
                value TEXT,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Legal knowledge database initialized or already exists.")

    def _store_knowledge(self, key: str, value: Any):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Store value as JSON string for complex types
            json_value = json.dumps(value)
            last_updated = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO legal_knowledge (key, value, last_updated)
                VALUES (?, ?, ?)
            """, (key, json_value, last_updated))
            conn.commit()
            logger.debug(f"Stored '{key}' in DB.")
        except Exception as e:
            logger.error(f"Error storing knowledge for key '{key}': {e}")
        finally:
            conn.close()

    def _retrieve_knowledge(self, key: str) -> Optional[Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM legal_knowledge WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON for key '{key}'. Returning raw string.")
                return result[0]
        return None

    def _update_config_with_scraped_data(self, key: str, data: Dict):
        """Updates the HybExConfig instance with scraped data."""
        if key == 'income_thresholds' and data:
            self.config.ENTITY_CONFIG['income_thresholds'].update(data)
            logger.info(f"Updated income thresholds in config with scraped data: {data}")
        # Add other specific updates if needed for other legal data points

    def scrape_nalsa_data(self) -> Dict[str, Any]:
        """Scrapes data from NALSA website for legal aid schemes and eligibility."""
        logger.info("Attempting to scrape NALSA data...")
        
        # Try multiple NALSA URLs in order of preference
        urls_to_try = [
            "https://nalsa.gov.in/about-nalsa/",
            "https://nalsa.gov.in/preventive-strategic-legal-services-schemes/",
            "https://nalsa.gov.in/"
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        for url in urls_to_try:
            try:
                logger.info(f"Trying URL: {url}")
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract comprehensive eligibility criteria and income thresholds
                eligibility_criteria = []
                scraped_income_thresholds = {}
                text_content = soup.get_text()
                
                # --- Extract Legal Services Authorities Act Section 12 criteria ---
                # Look for Section 12 eligibility criteria from the Act
                section_12_patterns = [
                    r'Section 12.*?criteria.*?legal services.*?person.*?is.*?–(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
                    r'12\..*?Criteria.*?giving legal services.*?person.*?is.*?–(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
                    r'member of a Scheduled Caste or Scheduled Tribe.*?industrial workman.*?annual income.*?rupees',
                ]
                
                for pattern in section_12_patterns:
                    match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
                    if match:
                        section_12_text = match.group(1) if match.groups() else match.group(0)
                        # Extract individual criteria points
                        criteria_lines = [line.strip() for line in section_12_text.split('\n') if line.strip()]
                        eligibility_criteria.extend(criteria_lines[:20])  # Limit to avoid noise
                        logger.info(f"Extracted {len(criteria_lines)} eligibility criteria from Section 12")
                        break
                
                # --- Extract Income Thresholds with multiple patterns ---
                income_patterns = [
                    # Current enhanced limits mentioned on NALSA website
                    (r'Enhancement.*?Rs\.?\s*1,25,000.*?Supreme Court', 'supreme_court', 125000),
                    (r'Enhancement.*?Rs\.?\s*1,00,000.*?High Court', 'high_court', 100000),
                    (r'Rs\.?\s*1,25,000.*?p\.a\..*?Supreme Court', 'supreme_court', 125000),
                    (r'Rs\.?\s*1,00,000.*?p\.a\..*?High Court', 'high_court', 100000),
                    
                    # Original Act amounts (historical reference)
                    (r'rupees nine thousand.*?other than.*?Supreme Court', 'original_general', 9000),
                    (r'rupees twelve thousand.*?Supreme Court', 'original_supreme', 12000),
                    
                    # General income threshold patterns
                    (r'annual income.*?not exceed.*?Rs\.?\s*([\d,]+)', 'general', None),
                    (r'family income.*?below.*?Rs\.?\s*([\d,]+)', 'general', None),
                    (r'income.*?less than.*?Rs\.?\s*([\d,]+)', 'general', None),
                ]
                
                for pattern, threshold_type, fixed_amount in income_patterns:
                    matches = re.finditer(pattern, text_content, re.IGNORECASE)
                    for match in matches:
                        if fixed_amount:
                            scraped_income_thresholds[threshold_type] = fixed_amount
                            logger.info(f"Found {threshold_type} income threshold: ₹{fixed_amount}")
                        elif match.groups():
                            amount_str = match.group(1).replace(',', '').replace(' ', '')
                            try:
                                amount = int(amount_str)
                                scraped_income_thresholds[threshold_type] = amount
                                logger.info(f"Extracted {threshold_type} income threshold: ₹{amount}")
                            except ValueError:
                                continue
                
                # --- Extract categorical eligibility information ---
                categorical_eligibility = []
                categorical_patterns = [
                    r'member of a Scheduled Caste or Scheduled Tribe',
                    r'victim of trafficking in human beings',
                    r'woman or a child',
                    r'mentally ill or otherwise disabled person',
                    r'industrial workman',
                    r'victim of.*?mass disaster.*?ethnic violence.*?caste atrocity',
                ]
                
                for pattern in categorical_patterns:
                    if re.search(pattern, text_content, re.IGNORECASE):
                        categorical_eligibility.append(pattern.replace(r'\s+', ' ').strip())
                
                # --- Store comprehensive data ---
                nalsa_data = {
                    'eligibility_criteria': eligibility_criteria,
                    'categorical_eligibility': categorical_eligibility,
                    'income_thresholds': scraped_income_thresholds,
                    'source_url': url,
                    'extraction_method': 'comprehensive_text_parsing',
                    'scraped_at': datetime.now().isoformat()
                }
                
                self._store_knowledge('nalsa_eligibility_criteria', eligibility_criteria)
                self._store_knowledge('nalsa_categorical_eligibility', categorical_eligibility)
                self._store_knowledge('nalsa_income_thresholds', scraped_income_thresholds)
                self._store_knowledge('nalsa_comprehensive_data', nalsa_data)
                
                # Update config with the most relevant income thresholds
                config_thresholds = {}
                if 'high_court' in scraped_income_thresholds:
                    config_thresholds['general'] = scraped_income_thresholds['high_court']
                if 'supreme_court' in scraped_income_thresholds:
                    config_thresholds['supreme_court'] = scraped_income_thresholds['supreme_court']
                
                if config_thresholds:
                    self._update_config_with_scraped_data('income_thresholds', config_thresholds)
                
                return {
                    "status": "success",
                    "source": "NALSA",
                    "source_url": url,
                    "eligibility_criteria_count": len(eligibility_criteria),
                    "categorical_eligibility_count": len(categorical_eligibility),
                    "income_thresholds_scraped": scraped_income_thresholds,
                    "config_updated": bool(config_thresholds),
                    "last_scraped": datetime.now().isoformat()
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to access {url}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing {url}: {e}")
                continue
        
        # If all URLs failed
        logger.error("All NALSA URLs failed to provide usable data")
        return {
            "status": "error", 
            "message": "All NALSA URLs failed",
            "urls_tried": urls_to_try
        }

    def scrape_doj_data(self) -> Dict[str, Any]:
        """Scrapes data from Department of Justice website for relevant legal schemes."""
        logger.info("Attempting to scrape Department of Justice data...")
        
        # Try multiple DOJ URLs
        urls_to_try = [
            "https://doj.gov.in/",
            "https://doj.gov.in/legal-aid/",
            "https://doj.gov.in/about-us/"
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        for url in urls_to_try:
            try:
                logger.info(f"Trying DOJ URL: {url}")
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                schemes = []
                legal_programs = []
                text_content = soup.get_text()
                
                # Look for legal aid related content
                legal_aid_keywords = [
                    'legal aid', 'legal services', 'free legal assistance',
                    'legal literacy', 'access to justice', 'legal awareness'
                ]
                
                # Extract paragraphs containing legal aid keywords
                for keyword in legal_aid_keywords:
                    pattern = rf'([^.]*{keyword}[^.]*\.)'
                    matches = re.finditer(pattern, text_content, re.IGNORECASE)
                    for match in matches:
                        sentence = match.group(1).strip()
                        if len(sentence) > 20 and sentence not in schemes:  # Avoid duplicates and very short matches
                            schemes.append(sentence)
                
                # Look for scheme/program lists
                scheme_elements = soup.find_all(['li', 'p'], text=re.compile(r'scheme|program|initiative', re.IGNORECASE))
                for element in scheme_elements[:10]:  # Limit to avoid too much noise
                    text = element.get_text(strip=True)
                    if len(text) > 15 and text not in legal_programs:
                        legal_programs.append(text)

                # Combine all extracted information
                all_content = schemes + legal_programs
                
                doj_data = {
                    'schemes': schemes,
                    'legal_programs': legal_programs,
                    'all_content': all_content,
                    'source_url': url,
                    'scraped_at': datetime.now().isoformat()
                }
                
                self._store_knowledge('doj_schemes', schemes)
                self._store_knowledge('doj_legal_programs', legal_programs)
                self._store_knowledge('doj_comprehensive_data', doj_data)
                
                return {
                    "status": "success",
                    "source": "Department of Justice",
                    "source_url": url,
                    "schemes_count": len(schemes),
                    "programs_count": len(legal_programs),
                    "total_content_items": len(all_content),
                    "last_scraped": datetime.now().isoformat()
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to access DOJ URL {url}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing DOJ URL {url}: {e}")
                continue
        
        # If all URLs failed
        logger.error("All DOJ URLs failed to provide usable data")
        return {
            "status": "error", 
            "message": "All DOJ URLs failed",
            "urls_tried": urls_to_try
        }

    def get_current_legal_knowledge(self) -> Dict[str, Any]:
        """Retrieves all stored legal knowledge."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM legal_knowledge")
        knowledge = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        conn.close()
        return knowledge

    def validate_scraped_data(self) -> Dict[str, bool]:
        """Validates the quality and presence of scraped data."""
        knowledge = self.get_current_legal_knowledge()
        validation_results = {
            'nalsa_eligibility_valid': bool(knowledge.get('nalsa_eligibility_criteria')),
            'nalsa_income_thresholds_valid': bool(knowledge.get('nalsa_income_thresholds')),
            'doj_schemes_valid': bool(knowledge.get('doj_schemes')),
            # Add more specific validation checks as needed
        }
        logger.info(f"Scraped data validation results: {validation_results}")
        return validation_results

    def update_legal_knowledge(self) -> Dict[str, Any]:
        """Performs a full update of legal knowledge from all sources."""
        logger.info("Starting full legal knowledge update...")
        
        results = {}
        overall_status = "success"
        
        # Always attempt NALSA scraping (primary source)
        nalsa_res = self.scrape_nalsa_data()
        results['nalsa'] = nalsa_res
        
        # Attempt DOJ scraping regardless of NALSA result (independent sources)
        logger.info("Proceeding with DOJ scraping...")
        doj_res = self.scrape_doj_data()
        results['doj'] = doj_res
        
        # Determine overall status based on results
        successful_sources = [source for source, result in results.items() if result.get('status') == 'success']
        failed_sources = [source for source, result in results.items() if result.get('status') == 'error']
        
        if len(successful_sources) == len(results):
            overall_status = "success"
            logger.info(f"All sources scraped successfully: {successful_sources}")
        elif len(successful_sources) > 0:
            overall_status = "partial_success"
            logger.info(f"Partial success - {len(successful_sources)} of {len(results)} sources successful")
        else:
            overall_status = "failed"
            logger.error(f"All sources failed: {failed_sources}")
        
        # Create summary of scraped data
        summary = {
            'successful_sources': successful_sources,
            'failed_sources': failed_sources,
            'total_eligibility_criteria': 0,
            'total_schemes': 0,
            'income_thresholds_found': False
        }
        
        # Count scraped content
        for source, result in results.items():
            if result.get('status') == 'success':
                summary['total_eligibility_criteria'] += result.get('eligibility_criteria_count', 0)
                summary['total_schemes'] += result.get('schemes_count', 0) + result.get('programs_count', 0)
                if result.get('income_thresholds_scraped'):
                    summary['income_thresholds_found'] = True

        final_result = {
            "status": overall_status,
            "summary": summary,
            "details": results,
            "last_updated": datetime.now().isoformat(),
            "next_update_recommendation": "Update successful - legal knowledge base refreshed with current data"
        }
        
        # Store the update summary
        self._store_knowledge('last_update_summary', final_result)
        
        logger.info(f"Legal knowledge update completed with status: {overall_status}")
        return final_result