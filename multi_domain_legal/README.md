# HybEx-Law: Hybrid Neural-Symbolic Legal AI System

## Directory Structure

### Core System Components
- `hybex_system/` - Main system modules
  - `main.py` - Primary system interface
  - `prolog_engine.py` - Symbolic reasoning engine  
  - `neural_models.py` - Neural network components
  - `data_processor.py` - Data preprocessing
  - `evaluator.py` - Model evaluation
  - `trainer.py` - Model training
  - `config.py` - System configuration

### Knowledge Base
- `knowledge_base/` - Legal knowledge and rules
  - `multi_domain_rules.py` - Main rule definitions
  - `foundational_rules_clean.pl` - Core Prolog predicates
  - `legal_aid_clean_v2.pl` - Legal aid specific rules

### Models & Data
- `models/hybex_system/` - Trained neural models
  - `domain_classifier/` - Domain classification model
  - `eligibility_predictor/` - Eligibility prediction model
- `data/` - Training and evaluation datasets

### Supporting
- `scripts/` - Utility scripts
- `logs/` - System logs
- `results/` - Evaluation results
- `requirements.txt` - Python dependencies

## Usage

```python
from hybex_system.main import HybExLawSystem

# Initialize system
system = HybExLawSystem()

# Predict legal eligibility
result = system.predict_legal_eligibility("I need legal aid for custody case")
print(f"Eligible: {result['final_decision']['eligible']}")
```

## Features
- Hybrid neural-symbolic reasoning
- Multi-domain legal knowledge
- Real-time Prolog inference  
- Transformer-based neural models
- Comprehensive evaluation framework
