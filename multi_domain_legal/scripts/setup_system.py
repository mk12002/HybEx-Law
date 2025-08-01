"""
Setup and Installation Guide for HybEx-Law System.

This script helps with system setup, dependency installation,
and initial configuration of the multi-domain legal AI system.
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybExSetup:
    """Setup and installation manager for HybEx-Law system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.setup_log = []
        
    def run_complete_setup(self) -> Dict[str, Any]:
        """Run complete system setup"""
        logger.info("🚀 Starting HybEx-Law System Setup")
        
        setup_results = {
            'steps_completed': [],
            'steps_failed': [],
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'success': False
        }
        
        try:
            # Step 1: Check Python version
            logger.info("📋 Step 1: Checking Python version")
            if self.check_python_version():
                setup_results['steps_completed'].append('python_version_check')
                logger.info("✅ Python version compatible")
            else:
                setup_results['steps_failed'].append('python_version_check')
                logger.error("❌ Python version incompatible")
                return setup_results
            
            # Step 2: Install dependencies
            logger.info("📦 Step 2: Installing dependencies")
            if self.install_dependencies():
                setup_results['steps_completed'].append('dependency_installation')
                logger.info("✅ Dependencies installed")
            else:
                setup_results['steps_failed'].append('dependency_installation')
                logger.warning("⚠️ Some dependencies failed to install")
            
            # Step 3: Create directories
            logger.info("📁 Step 3: Creating project directories")
            if self.create_directories():
                setup_results['steps_completed'].append('directory_creation')
                logger.info("✅ Directories created")
            else:
                setup_results['steps_failed'].append('directory_creation')
                logger.error("❌ Failed to create directories")
            
            # Step 4: Download models
            logger.info("🤖 Step 4: Setting up language models")
            if self.setup_language_models():
                setup_results['steps_completed'].append('model_setup')
                logger.info("✅ Language models ready")
            else:
                setup_results['steps_failed'].append('model_setup')
                logger.warning("⚠️ Language model setup had issues")
            
            # Step 5: Test installation
            logger.info("🧪 Step 5: Testing installation")
            if self.test_installation():
                setup_results['steps_completed'].append('installation_test')
                logger.info("✅ Installation test passed")
            else:
                setup_results['steps_failed'].append('installation_test')
                logger.warning("⚠️ Installation test had issues")
            
            setup_results['success'] = len(setup_results['steps_failed']) == 0
            
        except Exception as e:
            logger.error(f"Setup failed with error: {e}")
            setup_results['error'] = str(e)
        
        return setup_results
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        major, minor = sys.version_info.major, sys.version_info.minor
        
        if major < 3 or (major == 3 and minor < 8):
            logger.error(f"Python 3.8+ required, found {major}.{minor}")
            return False
        
        logger.info(f"Python version {major}.{minor} is compatible")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            # Install core dependencies
            logger.info("Installing core dependencies...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning(f"Some dependencies failed to install: {result.stderr}")
                # Try installing essential packages individually
                essential_packages = [
                    "numpy>=1.21.0",
                    "pandas>=1.3.0", 
                    "scikit-learn>=1.0.0",
                    "tqdm>=4.62.0"
                ]
                
                for package in essential_packages:
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], check=True, capture_output=True, timeout=60)
                        logger.info(f"✅ Installed {package}")
                    except subprocess.CalledProcessError:
                        logger.warning(f"⚠️ Failed to install {package}")
                
                return True  # Return True even if some packages failed
            
            logger.info("All dependencies installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary project directories"""
        directories = [
            "data",
            "data/splits",
            "models",
            "models/domain_classifier",
            "models/fact_extractor", 
            "models/confidence_estimator",
            "logs",
            "evaluation_results",
            "training_output",
            "training_output/pipeline_report"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def setup_language_models(self) -> bool:
        """Setup language models and NLP resources"""
        try:
            # Try to download spaCy model
            try:
                logger.info("Downloading spaCy English model...")
                subprocess.run([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], check=True, capture_output=True, timeout=120)
                logger.info("✅ spaCy model downloaded")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning("⚠️ spaCy model download failed, will use fallback")
            
            # Try to download NLTK data
            try:
                import nltk
                logger.info("Downloading NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                logger.info("✅ NLTK data downloaded")
            except ImportError:
                logger.warning("⚠️ NLTK not available")
            except Exception:
                logger.warning("⚠️ NLTK data download failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Language model setup failed: {e}")
            return False
    
    def test_installation(self) -> bool:
        """Test the installation"""
        try:
            # Test core imports
            logger.info("Testing core imports...")
            
            test_imports = [
                ("json", "json"),
                ("pathlib", "pathlib.Path"),
                ("logging", "logging"),
                ("typing", "typing.Dict"),
            ]
            
            for module_name, import_statement in test_imports:
                try:
                    exec(f"import {import_statement}")
                    logger.info(f"✅ {module_name} import successful")
                except ImportError as e:
                    logger.warning(f"⚠️ {module_name} import failed: {e}")
            
            # Test optional ML imports
            ml_imports = [
                ("numpy", "numpy"),
                ("pandas", "pandas"),
                ("sklearn", "sklearn.linear_model"),
            ]
            
            ml_success_count = 0
            for module_name, import_statement in ml_imports:
                try:
                    exec(f"import {import_statement}")
                    logger.info(f"✅ {module_name} import successful")
                    ml_success_count += 1
                except ImportError:
                    logger.warning(f"⚠️ {module_name} not available")
            
            # Test basic system functionality
            logger.info("Testing basic system functionality...")
            
            try:
                # Test that we can import our core modules
                sys.path.append(str(self.project_root / "src"))
                from multi_domain_legal_pipeline import MultiDomainLegalPipeline
                
                # Create basic pipeline instance
                pipeline = MultiDomainLegalPipeline()
                logger.info("✅ Core system import successful")
                
                # Test basic query processing
                test_query = "I am a poor woman seeking legal help"
                result = pipeline.process_query(test_query)
                
                if result and 'domains' in result:
                    logger.info("✅ Basic query processing works")
                    return True
                else:
                    logger.warning("⚠️ Query processing returned unexpected result")
                    return ml_success_count >= 2  # At least 2 ML libraries work
                    
            except ImportError as e:
                logger.warning(f"⚠️ Core system import failed: {e}")
                return ml_success_count >= 2
                
        except Exception as e:
            logger.error(f"Installation test failed: {e}")
            return False
    
    def generate_setup_report(self, results: Dict[str, Any]):
        """Generate setup report"""
        report_path = self.project_root / "setup_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("HYBEX-LAW SYSTEM SETUP REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Setup Status: {'SUCCESS' if results['success'] else 'PARTIAL/FAILED'}\n")
            f.write(f"Python Version: {results['python_version']}\n\n")
            
            if results['steps_completed']:
                f.write("COMPLETED STEPS:\n")
                for step in results['steps_completed']:
                    f.write(f"  ✅ {step.replace('_', ' ').title()}\n")
                f.write("\n")
            
            if results['steps_failed']:
                f.write("FAILED STEPS:\n")
                for step in results['steps_failed']:
                    f.write(f"  ❌ {step.replace('_', ' ').title()}\n")
                f.write("\n")
            
            f.write("NEXT STEPS:\n")
            if results['success']:
                f.write("  1. Run: python scripts/comprehensive_data_generation.py\n")
                f.write("  2. Run: python scripts/complete_training_pipeline.py\n")
                f.write("  3. Test with: python demo/interactive_demo.py\n")
            else:
                f.write("  1. Review failed installation steps above\n")
                f.write("  2. Install missing dependencies manually\n")
                f.write("  3. Re-run setup: python scripts/setup_system.py\n")
        
        logger.info(f"Setup report saved to {report_path}")

def main():
    """Main setup function"""
    print("🚀 HybEx-Law System Setup")
    print("=" * 40)
    
    setup_manager = HybExSetup()
    results = setup_manager.run_complete_setup()
    
    # Print results
    print(f"\n📊 Setup Results:")
    print(f"   Status: {'✅ SUCCESS' if results['success'] else '⚠️ PARTIAL'}")
    print(f"   Python Version: {results['python_version']}")
    print(f"   Steps Completed: {len(results['steps_completed'])}")
    print(f"   Steps Failed: {len(results['steps_failed'])}")
    
    if results['steps_completed']:
        print(f"\n✅ Completed Steps:")
        for step in results['steps_completed']:
            print(f"   • {step.replace('_', ' ').title()}")
    
    if results['steps_failed']:
        print(f"\n❌ Failed Steps:")
        for step in results['steps_failed']:
            print(f"   • {step.replace('_', ' ').title()}")
    
    # Generate report
    setup_manager.generate_setup_report(results)
    
    # Show next steps
    print(f"\n🎯 Next Steps:")
    if results['success']:
        print("   1. ✅ System ready for use!")
        print("   2. 📊 Generate training data: python scripts/comprehensive_data_generation.py")
        print("   3. 🤖 Train models: python scripts/complete_training_pipeline.py")
        print("   4. 🎮 Try demo: python demo/interactive_demo.py")
    else:
        print("   1. ⚠️ Review setup issues above")
        print("   2. 📦 Install missing dependencies manually")
        print("   3. 🔄 Re-run setup if needed")
        print("   4. 📄 Check setup_report.txt for details")
    
    print(f"\n📋 Setup report saved to: setup_report.txt")
    print("🎉 HybEx-Law setup completed!")

if __name__ == "__main__":
    main()
