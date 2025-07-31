"""
Setup and installation script for HybEx-Law project.

This script helps set up the development environment and install dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and return success status."""
    if description:
        print(f"🔧 {description}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("Please install Python 3.8 or higher")
        return False


def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )


def download_spacy_model():
    """Download spaCy English model."""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )


def check_swi_prolog():
    """Check if SWI-Prolog is installed."""
    try:
        result = subprocess.run(
            "swipl --version", 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("✅ SWI-Prolog is installed")
        print(f"Version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ SWI-Prolog not found")
        print("Please install SWI-Prolog from: https://www.swi-prolog.org/download/stable")
        print("Make sure it's added to your system PATH")
        return False


def test_imports():
    """Test if key imports work."""
    print("🧪 Testing imports...")
    
    test_modules = [
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("nltk", "NLTK"),
    ]
    
    success = True
    for module, name in test_modules:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
        except ImportError:
            print(f"❌ {name} import failed")
            success = False
    
    # Test PySwip separately (optional)
    try:
        import pyswip
        print("✅ PySwip import successful")
    except ImportError:
        print("⚠️  PySwip import failed (will use mock engine)")
        print("For full functionality, install: pip install pyswip")
    
    return success


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/annotations",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def main():
    """Main setup routine."""
    print("🏛️  HybEx-Law Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing Dependencies...")
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        return False
    
    # Download spaCy model
    print("\n🔤 Setting up NLP models...")
    if not download_spacy_model():
        print("⚠️  Failed to download spaCy model (you can install it manually later)")
    
    # Check SWI-Prolog
    print("\n🧠 Checking Prolog installation...")
    swi_prolog_available = check_swi_prolog()
    
    # Test imports
    print("\n🧪 Testing installation...")
    if not test_imports():
        print("❌ Some imports failed. Please check the installation.")
        return False
    
    # Final status
    print("\n" + "=" * 50)
    if swi_prolog_available:
        print("🎉 Setup completed successfully!")
        print("You can now run the full HybEx-Law system.")
    else:
        print("⚠️  Setup mostly completed!")
        print("Install SWI-Prolog for full functionality.")
    
    print("\nNext steps:")
    print("1. Run: python examples.py")
    print("2. Open: notebooks/getting_started.ipynb")
    print("3. Try: python main.py --query 'Your legal question here'")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
