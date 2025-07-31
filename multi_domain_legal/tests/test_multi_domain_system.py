"""
Test Suite for Multi-Domain Legal AI System.

This module contains comprehensive tests for all components of the
multi-domain legal analysis system.
"""

import unittest
import json
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.core.domain_registry import LegalDomain, DOMAIN_REGISTRY
from src.core.domain_classifier import DomainClassifier
from src.core.text_preprocessor import TextPreprocessor
from src.core.multi_domain_pipeline import MultiDomainLegalPipeline
from main import MultiDomainLegalAI

class TestDomainRegistry(unittest.TestCase):
    """Test domain registry functionality"""
    
    def test_all_domains_present(self):
        """Test that all expected domains are registered"""
        expected_domains = {
            LegalDomain.LEGAL_AID,
            LegalDomain.FAMILY_LAW,
            LegalDomain.CONSUMER_PROTECTION,
            LegalDomain.FUNDAMENTAL_RIGHTS,
            LegalDomain.EMPLOYMENT_LAW
        }
        
        registered_domains = set(DOMAIN_REGISTRY.keys())
        self.assertEqual(expected_domains, registered_domains)
    
    def test_domain_acts_not_empty(self):
        """Test that each domain has associated legal acts"""
        for domain, acts in DOMAIN_REGISTRY.items():
            self.assertGreater(len(acts), 0, f"Domain {domain} has no associated acts")
    
    def test_act_structure(self):
        """Test that acts have required fields"""
        for domain, acts in DOMAIN_REGISTRY.items():
            for act in acts:
                self.assertTrue(hasattr(act, 'title'))
                self.assertTrue(hasattr(act, 'year'))
                self.assertTrue(hasattr(act, 'keywords'))
                self.assertIsInstance(act.keywords, list)

class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing functionality"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing"""
        text = "I NEED HELP WITH DIVORCE CASE!!! Please advise..."
        processed = self.preprocessor.preprocess_text(text)
        
        self.assertNotIn("!!!", processed)
        self.assertNotIn("...", processed)
        self.assertIsInstance(processed, str)
    
    def test_legal_abbreviation_expansion(self):
        """Test expansion of legal abbreviations"""
        text = "I filed PIL under Art. 32 of Constitution"
        processed = self.preprocessor.preprocess_text(text)
        
        # Should expand common legal abbreviations
        self.assertIn("public interest litigation", processed.lower())
        self.assertIn("article", processed.lower())
    
    def test_currency_normalization(self):
        """Test currency amount normalization"""
        text = "I paid Rs. 50,000 for the product"
        processed = self.preprocessor.preprocess_text(text)
        
        # Should normalize currency representation
        self.assertIn("50000", processed)
    
    def test_entity_extraction(self):
        """Test entity extraction from legal text"""
        text = "I work at Microsoft and earn Rs. 50,000 per month in Mumbai"
        entities = self.preprocessor.extract_entities(text)
        
        self.assertIn('organizations', entities)
        self.assertIn('locations', entities)
        self.assertIn('amounts', entities)

class TestDomainClassifier(unittest.TestCase):
    """Test domain classification functionality"""
    
    def setUp(self):
        self.classifier = DomainClassifier()
    
    def test_legal_aid_classification(self):
        """Test classification of legal aid queries"""
        query = "I am poor and need free legal help for my case"
        predictions = self.classifier.predict(query)
        
        self.assertIn(LegalDomain.LEGAL_AID, predictions)
        self.assertGreater(predictions[LegalDomain.LEGAL_AID], 0.5)
    
    def test_family_law_classification(self):
        """Test classification of family law queries"""
        query = "My husband is demanding dowry and I want divorce"
        predictions = self.classifier.predict(query)
        
        self.assertIn(LegalDomain.FAMILY_LAW, predictions)
        self.assertGreater(predictions[LegalDomain.FAMILY_LAW], 0.5)
    
    def test_employment_law_classification(self):
        """Test classification of employment law queries"""
        query = "My company fired me without notice after 5 years"
        predictions = self.classifier.predict(query)
        
        self.assertIn(LegalDomain.EMPLOYMENT_LAW, predictions)
        self.assertGreater(predictions[LegalDomain.EMPLOYMENT_LAW], 0.5)
    
    def test_multi_domain_classification(self):
        """Test classification of multi-domain queries"""
        query = "I am poor woman facing divorce and workplace harassment"
        predictions = self.classifier.predict(query)
        
        # Should identify multiple relevant domains
        high_confidence_domains = [
            domain for domain, confidence in predictions.items()
            if confidence > 0.3
        ]
        self.assertGreaterEqual(len(high_confidence_domains), 2)

class TestDomainProcessors(unittest.TestCase):
    """Test individual domain processors"""
    
    def setUp(self):
        self.pipeline = MultiDomainLegalPipeline()
    
    def test_legal_aid_processor(self):
        """Test legal aid processor"""
        from src.domains.legal_aid.processor import LegalAidProcessor
        processor = LegalAidProcessor()
        
        query = "I earn Rs. 20000 per month and need legal help"
        facts = processor.extract_facts(query)
        
        self.assertIn("person(user).", facts)
        self.assertTrue(any("monthly_income" in fact for fact in facts))
        
        analysis = processor.analyze_legal_position(facts)
        self.assertIn("eligible_for_legal_aid", analysis)
    
    def test_family_law_processor(self):
        """Test family law processor"""
        from src.domains.family_law.processor import FamilyLawProcessor
        processor = FamilyLawProcessor()
        
        query = "My husband is cruel and I want divorce under Hindu law"
        facts = processor.extract_facts(query)
        analysis = processor.analyze_legal_position(facts)
        
        self.assertTrue(any("cruelty" in fact for fact in facts))
        self.assertIn("divorce_grounds_valid", analysis)
    
    def test_employment_law_processor(self):
        """Test employment law processor"""
        from src.domains.employment_law.processor import EmploymentLawProcessor
        processor = EmploymentLawProcessor()
        
        query = "Company fired me without notice after 5 years service"
        facts = processor.extract_facts(query)
        analysis = processor.analyze_legal_position(facts)
        
        self.assertTrue(any("termination" in fact for fact in facts))
        self.assertIn("wrongful_termination_case", analysis)

class TestMultiDomainPipeline(unittest.TestCase):
    """Test the complete multi-domain pipeline"""
    
    def setUp(self):
        self.pipeline = MultiDomainLegalPipeline()
    
    def test_single_domain_query(self):
        """Test processing of single-domain query"""
        query = "My company terminated me without proper notice"
        result = self.pipeline.process_legal_query(query)
        
        self.assertIn('domain_classification', result)
        self.assertIn('relevant_domains', result)
        self.assertIn('domain_results', result)
        self.assertIn('recommendations', result)
        
        # Should identify employment law as primary domain
        self.assertIn('employment_law', result['relevant_domains'])
    
    def test_multi_domain_query(self):
        """Test processing of multi-domain query"""
        query = "I am poor woman whose husband left and facing workplace harassment"
        result = self.pipeline.process_legal_query(query)
        
        # Should identify multiple domains
        self.assertGreaterEqual(len(result['relevant_domains']), 2)
        
        # Should have cross-domain analysis
        self.assertIn('cross_domain_issues', result['unified_analysis'])
    
    def test_recommendations_generation(self):
        """Test generation of actionable recommendations"""
        query = "I bought defective mobile phone worth Rs. 50000"
        result = self.pipeline.process_legal_query(query)
        
        recommendations = result['recommendations']
        self.assertIn('immediate_actions', recommendations)
        self.assertIn('legal_procedures', recommendations)
        
        # Should have some recommendations
        total_recommendations = sum(len(actions) for actions in recommendations.values())
        self.assertGreater(total_recommendations, 0)

class TestMainApplication(unittest.TestCase):
    """Test the main application interface"""
    
    def setUp(self):
        self.app = MultiDomainLegalAI()
    
    def test_application_initialization(self):
        """Test that application initializes correctly"""
        self.assertIsNotNone(self.app.pipeline)
        self.assertIsNotNone(self.app.config)
        
        status = self.app.get_system_status()
        self.assertEqual(status['system_version'], '1.0.0')
        self.assertEqual(status['pipeline_status']['total_domains'], 5)
    
    def test_query_processing(self):
        """Test query processing through main application"""
        query = "I need legal aid for family dispute"
        result = self.app.process_query(query)
        
        self.assertNotIn('error', result)
        self.assertIn('system_info', result)
        self.assertEqual(result['system_info']['status'], 'success')
    
    def test_domain_information(self):
        """Test retrieval of domain information"""
        domain_info = self.app.get_domain_information()
        
        self.assertEqual(domain_info['total_domains'], 5)
        self.assertIn('domains', domain_info)
        
        # Check that all expected domains are present
        domain_names = set(domain_info['domains'].keys())
        expected_domains = {
            'legal_aid', 'family_law', 'consumer_protection',
            'fundamental_rights', 'employment_law'
        }
        self.assertEqual(domain_names, expected_domains)

class TestSampleQueries(unittest.TestCase):
    """Test system with sample queries from data file"""
    
    def setUp(self):
        self.app = MultiDomainLegalAI()
        
        # Load sample queries
        sample_file = Path(__file__).parent.parent / 'data' / 'sample_queries.json'
        if sample_file.exists():
            with open(sample_file, 'r') as f:
                self.samples = json.load(f)
        else:
            self.samples = {}
    
    def test_legal_aid_samples(self):
        """Test legal aid sample queries"""
        if 'legal_aid_queries' not in self.samples:
            self.skipTest("No legal aid samples available")
        
        for sample in self.samples['legal_aid_queries'][:2]:  # Test first 2
            with self.subTest(query=sample['query'][:50]):
                result = self.app.process_query(sample['query'])
                self.assertNotIn('error', result)
                self.assertIn('legal_aid', result['relevant_domains'])
    
    def test_family_law_samples(self):
        """Test family law sample queries"""
        if 'family_law_queries' not in self.samples:
            self.skipTest("No family law samples available")
        
        for sample in self.samples['family_law_queries'][:2]:  # Test first 2
            with self.subTest(query=sample['query'][:50]):
                result = self.app.process_query(sample['query'])
                self.assertNotIn('error', result)
                self.assertIn('family_law', result['relevant_domains'])
    
    def test_multi_domain_samples(self):
        """Test multi-domain sample queries"""
        if 'multi_domain_queries' not in self.samples:
            self.skipTest("No multi-domain samples available")
        
        for sample in self.samples['multi_domain_queries']:
            with self.subTest(query=sample['query'][:50]):
                result = self.app.process_query(sample['query'])
                self.assertNotIn('error', result)
                self.assertGreaterEqual(len(result['relevant_domains']), 2)

def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDomainRegistry,
        TestTextPreprocessor,
        TestDomainClassifier,
        TestDomainProcessors,
        TestMultiDomainPipeline,
        TestMainApplication,
        TestSampleQueries
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    if result.errors:
        print("\\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
