"""
Multi-Domain Legal AI System - Main Application.

This is the main entry point for the multi-domain legal AI system that handles
legal queries across 5 major domains of Indian law.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.multi_domain_pipeline import MultiDomainLegalPipeline
from knowledge_base.multi_domain_rules import MULTI_DOMAIN_LEGAL_RULES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_ai.log'),
        logging.StreamHandler()
    ]
)

class MultiDomainLegalAI:
    """
    Main application class for multi-domain legal AI system.
    
    Provides a complete interface for processing legal queries across:
    - Legal Aid and Access to Justice
    - Family Law and Personal Status
    - Consumer Protection and Rights  
    - Fundamental Rights and Constitutional Law
    - Employment Law and Labor Rights
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the multi-domain legal AI system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize the pipeline
        confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.pipeline = MultiDomainLegalPipeline(confidence_threshold=confidence_threshold)
        
        # Initialize Prolog knowledge base (placeholder)
        self.knowledge_base = MULTI_DOMAIN_LEGAL_RULES
        
        self.logger.info("Multi-Domain Legal AI System initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'confidence_threshold': 0.3,
            'max_query_length': 5000,
            'enable_cross_domain_analysis': True,
            'log_level': 'INFO'
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def process_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a legal query and return comprehensive analysis.
        
        Args:
            query: Legal query in natural language
            user_context: Optional user context (location, demographics, etc.)
            
        Returns:
            Complete analysis including recommendations and legal advice
        """
        try:
            # Validate query
            if not query or not query.strip():
                return {'error': 'Empty query provided', 'status': 'failed'}
            
            if len(query) > self.config['max_query_length']:
                return {
                    'error': f'Query too long. Maximum length: {self.config["max_query_length"]}',
                    'status': 'failed'
                }
            
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Process through pipeline
            result = self.pipeline.process_legal_query(query, user_context)
            
            # Add system metadata
            result['system_info'] = {
                'version': '1.0.0',
                'domains_available': 5,
                'confidence_threshold': self.config['confidence_threshold'],
                'timestamp': str(Path().resolve()),
                'status': 'success'
            }
            
            self.logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'query': query,
                'status': 'failed',
                'system_info': {
                    'version': '1.0.0',
                    'status': 'error'
                }
            }
    
    def get_domain_information(self) -> Dict[str, Any]:
        """
        Get information about available legal domains.
        
        Returns:
            Dictionary with domain information and capabilities
        """
        return self.pipeline.get_domain_summary()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and health information.
        
        Returns:
            System status information
        """
        pipeline_status = self.pipeline.get_pipeline_status()
        
        return {
            'system_version': '1.0.0',
            'pipeline_status': pipeline_status,
            'configuration': self.config,
            'knowledge_base_loaded': bool(self.knowledge_base),
            'total_legal_rules': len(self.knowledge_base.split('\n')) if self.knowledge_base else 0
        }
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """
        Update system configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update confidence threshold if provided
            if 'confidence_threshold' in new_config:
                threshold = new_config['confidence_threshold']
                self.pipeline.update_confidence_threshold(threshold)
                self.config['confidence_threshold'] = threshold
            
            # Update other config values
            for key, value in new_config.items():
                if key in self.config:
                    self.config[key] = value
            
            self.logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            return False

def main():
    """
    Main function for command-line interface.
    """
    print("\\n" + "="*80)
    print("    MULTI-DOMAIN LEGAL AI SYSTEM")
    print("    Comprehensive Legal Analysis Across 5 Domains")
    print("="*80)
    
    # Initialize the system
    try:
        legal_ai = MultiDomainLegalAI()
        print("\\n✅ System initialized successfully!")
        
        # Show system status
        status = legal_ai.get_system_status()
        print(f"\\n📊 System Status:")
        print(f"   • Pipeline Status: {'✅ Ready' if status['pipeline_status']['preprocessor_ready'] else '❌ Not Ready'}")
        print(f"   • Active Domains: {status['pipeline_status']['total_domains']}")
        print(f"   • Confidence Threshold: {status['configuration']['confidence_threshold']}")
        
        # Show available domains
        domain_info = legal_ai.get_domain_information()
        print(f"\\n🏛️  Available Legal Domains:")
        for domain, info in domain_info['domains'].items():
            print(f"   • {info['description']}: {len(info['applicable_acts'])} acts")
        
    except Exception as e:
        print(f"\\n❌ Failed to initialize system: {str(e)}")
        return
    
    # Interactive query processing
    print("\\n" + "-"*80)
    print("🔍 INTERACTIVE QUERY PROCESSING")
    print("   Enter your legal queries below (type 'quit' to exit)")
    print("-"*80)
    
    while True:
        try:
            query = input("\\n💬 Legal Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\\n👋 Thank you for using Multi-Domain Legal AI System!")
                break
            
            if not query:
                continue
            
            print("\\n🔄 Processing query...")
            result = legal_ai.process_query(query)
            
            if result.get('status') == 'failed':
                print(f"\\n❌ Error: {result.get('error')}")
                continue
            
            # Display results
            print("\\n" + "="*60)
            print("📋 ANALYSIS RESULTS")
            print("="*60)
            
            # Domain classification
            print(f"\\n🎯 Domain Classification:")
            for domain, confidence in result['domain_classification'].items():
                status = "✅" if domain.value in result['relevant_domains'] else "⚪"
                print(f"   {status} {domain.value.replace('_', ' ').title()}: {confidence:.2f}")
            
            # Primary analysis
            if result['unified_analysis']['primary_domain']:
                primary = result['unified_analysis']['primary_domain'].replace('_', ' ').title()
                print(f"\\n🏆 Primary Domain: {primary}")
            
            # Recommendations
            if result['recommendations']:
                print(f"\\n💡 Recommendations:")
                
                if result['recommendations']['immediate_actions']:
                    print(f"   🚨 Immediate Actions:")
                    for action in result['recommendations']['immediate_actions'][:3]:
                        print(f"      • {action}")
                
                if result['recommendations']['legal_procedures']:
                    print(f"   ⚖️  Legal Procedures:")
                    for procedure in result['recommendations']['legal_procedures'][:3]:
                        print(f"      • {procedure}")
                
                if result['recommendations']['consultation_needed']:
                    print(f"   👨‍💼 Professional Consultation:")
                    for consultation in result['recommendations']['consultation_needed']:
                        print(f"      • {consultation}")
            
            # Cross-domain issues
            if result['unified_analysis']['cross_domain_issues']:
                print(f"\\n🔗 Cross-Domain Issues:")
                for issue in result['unified_analysis']['cross_domain_issues']:
                    print(f"   • {issue}")
            
            # Risk assessment
            risk_level = result['unified_analysis']['risk_assessment']
            risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(risk_level, '⚪')
            print(f"\\n{risk_emoji} Risk Assessment: {risk_level.upper()}")
            
            print("\\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\\n\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\\n❌ Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
