# comprehensive_system_test.py

import unittest
import logging
import json
from pathlib import Path
import os
from datetime import datetime
import sys

# Add the parent directory to the sys.path to allow importing hybex_system
# This assumes comprehensive_system_test.py is at the root level,
# and hybex_system is a sibling directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Adjust based on your actual structure
# More robust way to add project root to path, assuming hybex_system is a direct subdir of the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))


from hybex_system.main import HybExLawSystem
from hybex_system.config import HybExConfig

# Setup a dedicated logger for the test script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
test_logger = logging.getLogger(__name__)

class ComprehensiveSystemTest(unittest.TestCase):
    """
    Comprehensive integration tests for the HybEx-Law System.
    This test suite verifies the end-to-end flow from data scraping to prediction.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up for the entire test suite:
        - Initialize system with a test configuration.
        - Create necessary directories.
        - Define test data path.
        """
        test_logger.info("\n" + "="*80)
        test_logger.info("Setting up Comprehensive HybEx-Law System Test Suite")
        test_logger.info("="*80)

        # Use a specific test config to isolate test runs
        cls.config = HybExConfig()
        cls.config.DATA_DIR = cls.config.BASE_DIR / "test_data_pipeline"
        cls.config.RESULTS_DIR = cls.config.BASE_DIR / "test_results_pipeline"
        cls.config.MODELS_DIR = cls.config.BASE_DIR / "test_models_pipeline"
        cls.config.LOGS_DIR = cls.config.BASE_DIR / "test_logs_pipeline"

        # Ensure test directories are clean before starting
        cls.cleanup_test_dirs(cls.config)
        cls.config.create_directories() # Create fresh directories

        cls.system = HybExLawSystem(config_path=None) # Initialize HybExLawSystem directly with modified config object
        cls.system.config = cls.config # Override system's config with test config

        # Prepare dummy raw data for preprocessing if it doesn't exist
        cls.raw_data_dir = cls.config.DATA_DIR / "raw"
        cls.raw_data_dir.mkdir(parents=True, exist_ok=True)
        cls.dummy_raw_data_file = cls.raw_data_dir / "dummy_legal_data.json"
        
        # Ensure dummy data is available for preprocessing test
        if not cls.dummy_raw_data_file.exists():
            test_logger.info(f"Creating dummy raw data file: {cls.dummy_raw_data_file}")
            dummy_data = [
                {"sample_id": "test_001", "query": "I am a Scheduled Caste person and my annual income is 1.5 lakhs. I need legal aid for a family dispute.", "domains": ["Family Law", "Legal Aid"], "expected_eligibility": True},
                {"sample_id": "test_002", "query": "My income is 8 lakhs per year and I have a property dispute.", "domains": ["Property Law"], "expected_eligibility": False},
                {"sample_id": "test_003", "query": "I am a woman with 50,000 income, facing domestic violence.", "domains": ["Family Law", "Legal Aid"], "expected_eligibility": True},
                {"sample_id": "test_004", "query": "I earn 200000 rupees and am a ST category person, seeking help for a criminal case.", "domains": ["Criminal Law", "Legal Aid"], "expected_eligibility": True},
                {"sample_id": "test_005", "query": "I have an income of 4 lakh, and my case is about a corporate fraud.", "domains": ["Corporate Law"], "expected_eligibility": False}
            ]
            with open(cls.dummy_raw_data_file, 'w', encoding='utf-8') as f:
                json.dump(dummy_data, f, indent=2, ensure_ascii=False)
        else:
            test_logger.info(f"Dummy raw data file already exists: {cls.dummy_raw_data_file}")


    @classmethod
    def tearDownClass(cls):
        """
        Clean up after the entire test suite.
        """
        test_logger.info("\n" + "="*80)
        test_logger.info("Tearing down Comprehensive HybEx-Law System Test Suite")
        test_logger.info("="*80)
        cls.cleanup_test_dirs(cls.config) # Clean up all test-specific directories

    @staticmethod
    def cleanup_test_dirs(config: HybExConfig):
        """Helper to remove test directories and their contents."""
        for path in [config.DATA_DIR, config.RESULTS_DIR, config.MODELS_DIR, config.LOGS_DIR]:
            if path.exists() and path.is_dir():
                test_logger.info(f"Cleaning up test directory: {path}")
                import shutil
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    test_logger.error(f"Error removing directory {path}: {e}")

    def test_01_update_legal_knowledge(self):
        """Test Step 1: Legal knowledge update via scraper."""
        test_logger.info("\n--- Running Test: 01_update_legal_knowledge ---")
        try:
            results = self.system.update_legal_knowledge()
            self.assertEqual(results['status'], 'success', "Legal knowledge update should report success.")
            self.assertIn('nalsa', results['details'], "NALSA scraping details should be present.")
            self.assertIn('doj', results['details'], "DOJ scraping details should be present.")
            test_logger.info("✅ Test 01 (Legal knowledge update) Passed.")

            # Verify that config thresholds are updated, indicating successful integration
            initial_thresholds = self.system.config.ENTITY_CONFIG['income_thresholds']
            test_logger.info(f"Current Config Income Thresholds after scrape: {initial_thresholds}")
            # Assert that at least some default/scraped thresholds are present and are numbers
            self.assertGreater(len(initial_thresholds), 0, "Income thresholds should be updated in config.")
            self.assertIsInstance(initial_thresholds.get('general'), int, "General income threshold should be an integer.")

        except Exception as e:
            test_logger.error(f"❌ Test 01 (Legal knowledge update) Failed: {e}", exc_info=True)
            self.fail(f"Test 01 failed: {e}")

    def test_02_data_preprocessing(self):
        """Test Step 2: Data preprocessing pipeline."""
        test_logger.info("\n--- Running Test: 02_data_preprocessing ---")
        try:
            results = self.system.preprocess_data(str(self.raw_data_dir))
            
            self.assertIn('total_processed_samples', results)
            self.assertGreater(results['total_processed_samples'], 0, "Should process more than 0 samples.")
            
            # Verify processed files exist
            processed_data_dir = self.config.RESULTS_DIR / "processed_data"
            self.assertTrue(processed_data_dir.exists(), "Processed data directory should exist.")
            self.assertTrue((processed_data_dir / "train_samples.json").exists(), "Train samples file should exist.")
            self.assertTrue((processed_data_dir / "val_samples.json").exists(), "Validation samples file should exist.")
            self.assertTrue((processed_data_dir / "test_samples.json").exists(), "Test samples file should exist.")
            
            # Verify samples are enriched with extracted_entities
            with open(processed_data_dir / "test_samples.json", 'r', encoding='utf-8') as f:
                test_samples = json.load(f)
            self.assertGreater(len(test_samples), 0, "Test samples should not be empty.")
            self.assertIn('extracted_entities', test_samples[0], "Samples should contain 'extracted_entities'.")
            self.assertIn('income', test_samples[0]['extracted_entities'], "Extracted entities should include 'income'.")

            test_logger.info("✅ Test 02 (Data preprocessing) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 02 (Data preprocessing) Failed: {e}", exc_info=True)
            self.fail(f"Test 02 failed: {e}")

    def test_03_train_complete_system(self):
        """Test Step 3: Training the complete neural system."""
        test_logger.info("\n--- Running Test: 03_train_complete_system ---")
        try:
            # Data directory for training needs to be where processed data is saved
            processed_data_dir = self.config.RESULTS_DIR / "processed_data"
            results = self.system.train_complete_system(str(processed_data_dir))
            
            self.assertIn('status', results)
            self.assertEqual(results['status'], 'completed', "Training pipeline status should be 'completed'.")
            self.assertIn('final_models', results)
            self.assertIn('domain_classifier', results['final_models'], "Domain classifier model info should be present.")
            self.assertIn('eligibility_predictor', results['final_models'], "Eligibility predictor model info should be present.")
            
            # Verify model files exist
            domain_model_path = Path(results['final_models']['domain_classifier']['path']) / "model.pt"
            eligibility_model_path = Path(results['final_models']['eligibility_predictor']['path']) / "model.pt"
            self.assertTrue(domain_model_path.exists(), "Domain classifier model file should exist.")
            self.assertTrue(eligibility_model_path.exists(), "Eligibility predictor model file should exist.")

            test_logger.info("✅ Test 03 (System training) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 03 (System training) Failed: {e}", exc_info=True)
            self.fail(f"Test 03 failed: {e}")

    def test_04_evaluate_system(self):
        """Test Step 4: Comprehensive system evaluation."""
        test_logger.info("\n--- Running Test: 04_evaluate_system ---")
        try:
            # Pass the processed test data for evaluation
            test_data_path = self.config.RESULTS_DIR / "processed_data" / "test_samples.json"
            
            # The evaluator will load models from the MODELS_DIR defined in config
            # trainer.py saves models to this path.
            results = self.system.evaluate_system(test_data=str(test_data_path))
            
            self.assertIn('overall_status', results)
            self.assertEqual(results['overall_status'], 'Completed', "Evaluation status should be 'Completed'.")
            self.assertIn('neural_metrics', results)
            self.assertIn('prolog_metrics', results)
            self.assertIn('hybrid_metrics', results)
            self.assertIn('error_analysis', results)
            self.assertGreater(results['sample_size'], 0, "Evaluation should process samples.")

            # Basic check for non-zero accuracy (models should learn something)
            self.assertGreater(results['neural_metrics']['eligibility_predictor']['accuracy'], 0.5, "Neural predictor accuracy should be better than random.")
            self.assertGreater(results['prolog_metrics']['accuracy'], 0.5, "Prolog accuracy should be better than random.")
            self.assertGreater(results['hybrid_metrics']['accuracy'], 0.5, "Hybrid accuracy should be better than random.")

            test_logger.info("✅ Test 04 (System evaluation) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 04 (System evaluation) Failed: {e}", exc_info=True)
            self.fail(f"Test 04 failed: {e}")

    def test_05_end_to_end_prediction(self):
        """Test Step 5: End-to-end prediction with a sample query."""
        test_logger.info("\n--- Running Test: 05_end_to_end_prediction ---")
        query_eligible = "I am from SC category and my annual income is 1 lakh. Am I eligible for legal aid in a family dispute case?"
        query_ineligible = "My annual income is 10 lakhs and I need help for a business dispute."

        try:
            # Test an eligible query
            result_eligible = self.system.predict_legal_eligibility(query_eligible)
            self.assertIn('final_decision', result_eligible)
            self.assertTrue(result_eligible['final_decision']['eligible'], "SC/low income query should be eligible.")
            self.assertGreater(result_eligible['final_decision']['confidence'], 0.7, "Eligible prediction should have high confidence.")
            self.assertIn('Categorically eligible', result_eligible['prolog_reasoning']['reasoning'], "Prolog reasoning should reflect categorical eligibility.")
            test_logger.info(f"✅ Test 05 (Eligible Query) Passed. Result: {result_eligible['final_decision']['eligible']}")

            # Test an ineligible query
            result_ineligible = self.system.predict_legal_eligibility(query_ineligible)
            self.assertIn('final_decision', result_ineligible)
            self.assertFalse(result_ineligible['final_decision']['eligible'], "High income query should be ineligible.")
            self.assertLess(result_ineligible['final_decision']['confidence'], 0.7, "Ineligible prediction should have lower confidence.")
            self.assertIn('Not eligible', result_ineligible['prolog_reasoning']['reasoning'], "Prolog reasoning should reflect ineligibility.")
            test_logger.info(f"✅ Test 05 (Ineligible Query) Passed. Result: {result_ineligible['final_decision']['eligible']}")

            test_logger.info("✅ Test 05 (End-to-end prediction) Passed.")
        except Exception as e:
            test_logger.error(f"❌ Test 05 (End-to-end prediction) Failed: {e}", exc_info=True)
            self.fail(f"Test 05 failed: {e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)