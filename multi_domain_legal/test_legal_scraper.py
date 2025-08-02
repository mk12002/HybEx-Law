# test_legal_scraper.py

import unittest
import logging
import os
import shutil
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import sys

# Adjust sys.path to allow importing hybex_system when running this test directly
# This assumes test_legal_scraper.py is at the project root and hybex_system is a direct subdir
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybex_system.legal_scraper import LegalDataScraper
from hybex_system.config import HybExConfig

# Setup a dedicated logger for the test script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
test_logger = logging.getLogger(__name__)

class TestLegalScraper(unittest.TestCase):
    """
    Unit tests for the LegalDataScraper component.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a test environment before all tests in this class run.
        - Create a test-specific configuration.
        - Ensure test directories are clean and created.
        - Initialize the LegalDataScraper with the test config.
        """
        test_logger.info("\n" + "="*80)
        test_logger.info("Setting up LegalDataScraper Test Suite")
        test_logger.info("="*80)

        # Create a test-specific config to isolate test data/results
        cls.test_config = HybExConfig()
        cls.test_config.DATA_DIR = cls.test_config.BASE_DIR / "test_data_scraper"
        cls.test_config.RESULTS_DIR = cls.test_config.BASE_DIR / "test_results_scraper"
        cls.test_config.LOGS_DIR = cls.test_config.BASE_DIR / "test_logs_scraper"

        # Ensure test directories are clean before starting
        cls.cleanup_test_dirs(cls.test_config)
        cls.test_config.create_directories()

        # Initialize the scraper with the test-specific config
        cls.scraper = LegalDataScraper(cls.test_config)
        
        # Capture the initial state of income_thresholds from config BEFORE scraping
        # This allows us to verify if they've changed after scraping.
        cls.initial_income_thresholds = dict(cls.test_config.ENTITY_CONFIG['income_thresholds'])

        test_logger.info(f"Test config initialized. Data Dir: {cls.test_config.DATA_DIR}")
        test_logger.info(f"Scraper initialized. DB Path: {cls.scraper.db_path}")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the test environment after all tests in this class have run.
        - Remove test-specific directories.
        """
        test_logger.info("\n" + "="*80)
        test_logger.info("Tearing down LegalDataScraper Test Suite")
        test_logger.info("="*80)
        cls.cleanup_test_dirs(cls.test_config)

    @staticmethod
    def cleanup_test_dirs(config: HybExConfig):
        """Helper to remove test directories and their contents."""
        for path in [config.DATA_DIR, config.RESULTS_DIR, config.LOGS_DIR]:
            if path.exists() and path.is_dir():
                test_logger.info(f"Cleaning up test directory: {path}")
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    test_logger.error(f"Error removing directory {path}: {e}")

    def test_01_database_initialization(self):
        """Test that the SQLite database is initialized correctly."""
        test_logger.info("\n--- Running Test: 01_database_initialization ---")
        self.assertTrue(self.scraper.db_path.exists(), "Database file should exist after initialization.")
        
        conn = None
        try:
            conn = sqlite3.connect(self.scraper.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='legal_knowledge';")
            self.assertIsNotNone(cursor.fetchone(), "legal_knowledge table should exist.")
            test_logger.info("✅ Test 01 (Database initialization) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 01 (Database initialization) Failed: {e}", exc_info=True)
            self.fail(f"Test 01 failed: {e}")
        finally:
            if conn:
                conn.close()

    def test_02_update_legal_knowledge(self):
        """
        Test the core functionality: updating legal knowledge from web sources.
        This test is crucial and includes multiple assertions.
        """
        test_logger.info("\n--- Running Test: 02_update_legal_knowledge ---")
        try:
            results = self.scraper.update_legal_knowledge()

            self.assertIn('status', results, "Results should contain a 'status' key.")
            self.assertIn(results['status'], ['success', 'partial_success', 'failed'], "Status should be one of expected values.")
            
            # If scraping was successful or partially successful, check data.
            # Note: Live scraping can be flaky, so we test for *presence* and *format*
            # rather than specific values which might change on the website.
            if results['status'] in ['success', 'partial_success']:
                self.assertIn('nalsa', results['details'], "NALSA details should be present.")
                self.assertEqual(results['details']['nalsa']['status'], 'success', "NALSA scraping status should be 'success'.")
                self.assertGreater(results['details']['nalsa']['eligibility_criteria_count'], 0, 
                                   "NALSA should scrape some eligibility criteria.")
                self.assertIn('income_thresholds_scraped', results['details']['nalsa'],
                               "NALSA results should contain scraped income thresholds.")
                
                scraped_thresholds = results['details']['nalsa']['income_thresholds_scraped']
                self.assertIsInstance(scraped_thresholds, dict, "Scraped thresholds should be a dictionary.")
                if scraped_thresholds:
                    # Assert that at least one key (e.g., 'general') is present and its value is an int
                    self.assertTrue(any(isinstance(v, int) for v in scraped_thresholds.values()),
                                    "Scraped income thresholds should contain integer values.")
                
                # Verify that config thresholds have been updated (crucial data flow check)
                updated_config_thresholds = self.test_config.ENTITY_CONFIG['income_thresholds']
                test_logger.info(f"Initial Config Thresholds: {self.initial_income_thresholds}")
                test_logger.info(f"Updated Config Thresholds: {updated_config_thresholds}")
                
                # This assertion checks if the config's thresholds are *different* from the initial ones,
                # implying the scraper successfully updated them. It doesn't check specific values
                # due to potential variability in live website data, but ensures the update mechanism works.
                # If the website happens to have the exact same default thresholds as in config, this might pass without actual change.
                # A more robust test would mock the requests for predictable results.
                self.assertNotEqual(self.initial_income_thresholds, updated_config_thresholds,
                                    "Config income thresholds should be updated after scraping.")

                # Verify data persistence in SQLite DB
                retrieved_nalsa_eligibility = self.scraper._retrieve_knowledge('nalsa_eligibility_criteria')
                self.assertIsNotNone(retrieved_nalsa_eligibility, "NALSA eligibility should be stored in DB.")
                self.assertIsInstance(retrieved_nalsa_eligibility, list, "Retrieved NALSA eligibility should be a list.")
                self.assertGreater(len(retrieved_nalsa_eligibility), 0, "Retrieved NALSA eligibility should not be empty.")
                
                retrieved_nalsa_thresholds = self.scraper._retrieve_knowledge('nalsa_income_thresholds')
                self.assertIsNotNone(retrieved_nalsa_thresholds, "NALSA income thresholds should be stored in DB.")
                self.assertIsInstance(retrieved_nalsa_thresholds, dict, "Retrieved NALSA thresholds should be a dictionary.")

            test_logger.info("✅ Test 02 (Update legal knowledge) Passed.")

        except Exception as e:
            test_logger.error(f"❌ Test 02 (Update legal knowledge) Failed: {e}", exc_info=True)
            self.fail(f"Test 02 failed: {e}")

    def test_03_get_current_legal_knowledge(self):
        """Test retrieving all stored legal knowledge from the database."""
        test_logger.info("\n--- Running Test: 03_get_current_legal_knowledge ---")
        try:
            # First, ensure some data is present by running the update
            self.scraper.update_legal_knowledge() 

            all_knowledge = self.scraper.get_current_legal_knowledge()
            self.assertIsInstance(all_knowledge, dict, "Returned knowledge should be a dictionary.")
            self.assertIn('nalsa_eligibility_criteria', all_knowledge, "NALSA eligibility should be in retrieved knowledge.")
            self.assertIn('nalsa_income_thresholds', all_knowledge, "NALSA income thresholds should be in retrieved knowledge.")
            self.assertIsInstance(all_knowledge['nalsa_eligibility_criteria'], list, "Retrieved eligibility should be a list.")
            self.assertIsInstance(all_knowledge['nalsa_income_thresholds'], dict, "Retrieved thresholds should be a dictionary.")
            test_logger.info("✅ Test 03 (Get current legal knowledge) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 03 (Get current legal knowledge) Failed: {e}", exc_info=True)
            self.fail(f"Test 03 failed: {e}")

    def test_04_validate_scraped_data(self):
        """Test the validation of scraped data."""
        test_logger.info("\n--- Running Test: 04_validate_scraped_data ---")
        try:
            # Ensure data is scraped before validation
            self.scraper.update_legal_knowledge()

            validation_results = self.scraper.validate_scraped_data()
            self.assertIsInstance(validation_results, dict, "Validation results should be a dictionary.")
            self.assertIn('nalsa_eligibility_valid', validation_results, "Validation should check NALSA eligibility.")
            self.assertTrue(validation_results['nalsa_eligibility_valid'], "NALSA eligibility data should be valid.")
            self.assertTrue(validation_results['nalsa_income_thresholds_valid'], "NALSA income thresholds data should be valid.")
            test_logger.info("✅ Test 04 (Validate scraped data) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 04 (Validate scraped data) Failed: {e}", exc_info=True)
            self.fail(f"Test 04 failed: {e}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)