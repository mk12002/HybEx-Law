"""
Setup script for Multi-Domain Legal AI System.

This script handles the complete setup including dependency installation,
directory creation, and initial configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling"""
    print(f"\\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'models/trained',
        'data/processed',
        'knowledge_base/generated',
        'results'
    ]
    
    print("\\n🔄 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\\n🔄 Installing Python dependencies...")
    
    # Core dependencies
    core_packages = [
        'scikit-learn>=1.0.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'nltk>=3.7',
        'spacy>=3.4.0'
    ]
    
    # Install core packages
    for package in core_packages:
        if not run_command(f'pip install "{package}"', f"Installing {package}"):
            return False
    
    # Download spaCy model
    if not run_command('python -m spacy download en_core_web_sm', "Downloading spaCy English model"):
        print("⚠️  spaCy model download failed - text processing may be limited")
    
    # Download NLTK data
    print("\\n🔄 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
    
    return True

def create_config_file():
    """Create default configuration file"""
    config = {
        "confidence_threshold": 0.3,
        "max_query_length": 5000,
        "enable_cross_domain_analysis": True,
        "log_level": "INFO",
        "model_paths": {
            "domain_classifier": "models/trained/domain_classifier.pkl",
            "vectorizer": "models/trained/tfidf_vectorizer.pkl"
        }
    }
    
    config_path = Path('config.json')
    if not config_path.exists():
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Created default configuration file")
    else:
        print("✅ Configuration file already exists")
    
    return True

def verify_installation():
    """Verify that installation is working"""
    print("\\n🔄 Verifying installation...")
    
    try:
        # Test basic imports
        import sklearn
        import numpy
        import pandas
        import nltk
        print("✅ All core packages imported successfully")
        
        # Test spaCy
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️  spaCy model not available - some features may be limited")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("="*80)
    print("    MULTI-DOMAIN LEGAL AI SYSTEM - SETUP")
    print("    Setting up comprehensive legal analysis system")
    print("="*80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create configuration
    if not create_config_file():
        print("❌ Failed to create configuration")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    print("\\n" + "="*80)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\\n🚀 Next steps:")
    print("   1. Run 'python main.py' to start the interactive system")
    print("   2. Run 'python tests/test_multi_domain_system.py' to run tests")
    print("   3. Check 'README.md' for detailed usage instructions")
    print("\\n📋 System capabilities:")
    print("   • Legal Aid and Access to Justice")
    print("   • Family Law and Personal Status")
    print("   • Consumer Protection and Rights")
    print("   • Fundamental Rights and Constitutional Law")
    print("   • Employment Law and Labor Rights")
    print("\\n📁 Project structure:")
    print("   • src/ - Core system components")
    print("   • data/ - Sample queries and test data")
    print("   • knowledge_base/ - Legal rules and facts")
    print("   • tests/ - Comprehensive test suite")
    print("   • logs/ - System logs and debugging info")

if __name__ == "__main__":
    main()
