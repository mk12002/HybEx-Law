# HybEx-Law: Hybrid Neural-Symbolic Legal AI System

A comprehensive legal AI system for determining legal aid eligibility under the Legal Services Authorities Act, 1987. This system combines neural language models with symbolic reasoning for robust legal decision-making.

## 🏛️ System Overview

HybEx-Law is a production-ready hybrid neural-symbolic system designed for Indian legal aid eligibility determination. The system processes natural language legal queries and applies both machine learning and rule-based reasoning to provide accurate eligibility assessments.

### Key Features

- **Hybrid Architecture**: Combines DistilBERT neural models with Prolog symbolic reasoning
- **Multi-Domain Coverage**: Handles 5 legal domains (legal aid, family law, consumer protection, employment law, fundamental rights)
- **Comprehensive Training**: 10-15 epochs per model with early stopping and gradient clipping
- **Advanced Entity Extraction**: Income, social category, case type, and location recognition
- **Legal Compliance**: Implements Legal Services Authorities Act, 1987 eligibility criteria
- **Production-Ready**: Extensive logging, model checkpointing, and error handling

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi_domain_legal

# Install dependencies
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn spacy
python -m spacy download en_core_web_sm

# Optional: Install SWI-Prolog for symbolic reasoning
# Windows: Download from https://www.swi-prolog.org/download/stable
# Linux: sudo apt-get install swi-prolog
# macOS: brew install swi-prolog
```

### Basic Usage

```python
from hybex_system import create_system

# Initialize the system
system = create_system()

# Check system status
status = system.get_system_status()
print(status)

# Train the complete system
results = system.train_complete_system("data/")

# Evaluate trained models
evaluation = system.evaluate_system()
```

### Command Line Interface

```bash
# Train the complete system
python -m hybex_system.main train --data-dir data/

# Evaluate trained models
python -m hybex_system.main evaluate

# Check system status
python -m hybex_system.main status

# Preprocess data only
python -m hybex_system.main preprocess --data-dir data/
```

## 📊 System Architecture

### Neural Components

1. **Domain Classifier**: Multi-label classification for legal domains
2. **Entity Extractor**: BIO tagging for legal entities (income, social category, case type)
3. **Eligibility Predictor**: Binary classification for legal aid eligibility

### Symbolic Components

1. **Prolog Reasoning Engine**: Rule-based legal reasoning
2. **Legal Rule Knowledge Base**: Income thresholds and categorical eligibility rules
3. **Hybrid Decision Fusion**: Combines neural predictions with symbolic reasoning

### Training Configuration

- **Models**: DistilBERT-based architecture
- **Epochs**: 10-15 per model with early stopping
- **Learning Rates**: 1e-5 to 5e-6 (component-specific)
- **Batch Size**: 16-32 (configurable)
- **Gradient Clipping**: 1.0 norm
- **Early Stopping**: 3-5 patience epochs

## 📁 Project Structure

```
hybex_system/
├── __init__.py              # Package initialization
├── config.py               # System configuration
├── data_processor.py       # Data preprocessing pipeline
├── neural_models.py        # Neural network architectures
├── prolog_engine.py        # Symbolic reasoning engine
├── trainer.py              # Training orchestrator
├── evaluator.py            # Model evaluation framework
└── main.py                 # Main system interface

data/                       # Training data directory
├── legal_aid_samples.json
├── family_law_samples.json
├── consumer_protection_samples.json
├── employment_law_samples.json
└── fundamental_rights_samples.json

models/                     # Trained model storage
├── domain_classifier/
├── eligibility_predictor/
└── entity_extractor/

results/                    # Training and evaluation results
├── processed_data/
├── training_plots/
├── evaluation_plots/
└── logs/
```

## 🔧 Configuration

The system uses a comprehensive configuration system with the following key parameters:

```python
# Training Configuration
TRAINING_CONFIG = {
    'epochs': {
        'domain_classification': 12,
        'entity_extraction': 15,
        'eligibility_prediction': 10
    },
    'learning_rates': {
        'domain_classification': 2e-5,
        'entity_extraction': 1e-5,
        'eligibility_prediction': 5e-6
    },
    'batch_size': 16,
    'early_stopping_patience': 5,
    'gradient_clip_norm': 1.0
}

# Model Configuration
MODEL_CONFIG = {
    'base_model': 'distilbert-base-uncased',
    'max_length': 512,
    'dropout_rate': 0.3,
    'num_attention_heads': 8
}
```

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Neural Models**: Accuracy, F1-score, Precision, Recall, ROC-AUC
- **Prolog Reasoning**: Rule-based accuracy, reasoning consistency
- **End-to-End**: Hybrid system performance and component agreement
- **Visualizations**: Training curves, confusion matrices, ROC curves

## 🎯 Legal Domain Coverage

### 1. Legal Aid (General)
- Income-based eligibility (₹2 lakhs threshold)
- Categorical eligibility (SC/ST/BPL)
- Case type restrictions

### 2. Family Law
- Divorce, custody, maintenance cases
- Domestic violence protection
- Marriage-related disputes

### 3. Consumer Protection
- Product defects and warranty issues
- Unfair trade practices
- Consumer complaint resolution

### 4. Employment Law
- Wrongful termination cases
- Workplace harassment
- Labor law violations

### 5. Fundamental Rights
- Constitutional rights violations
- Human rights protection
- Civil liberties cases

## ⚖️ Legal Compliance

The system implements eligibility criteria from:

- **Legal Services Authorities Act, 1987**
- **Income Thresholds**: Category-specific limits
- **Categorical Eligibility**: SC/ST/BPL automatic qualification
- **Case Type Restrictions**: Excluded commercial disputes

## 🔍 Data Processing

### Entity Extraction

- **Income Recognition**: ₹, lakhs, crores with normalization
- **Social Categories**: SC/ST/OBC/EWS/General/BPL/APL
- **Case Types**: Criminal, civil, family, consumer, employment
- **Locations**: States, cities, districts

### Data Validation

- **Required Fields**: Query, domains, expected eligibility
- **Data Quality**: Length validation, format checking
- **Stratified Splits**: 70% train, 15% validation, 15% test

## 🧠 Model Training

### Training Pipeline

1. **Data Preprocessing**: Entity extraction and validation
2. **Neural Training**: Multi-model training with monitoring
3. **Prolog Integration**: Rule-based reasoning setup
4. **Comprehensive Evaluation**: End-to-end performance testing

### Monitoring Features

- **Real-time Logging**: Training progress and metrics
- **Model Checkpointing**: Best model state saving
- **Visualization**: Training curves and performance plots
- **Early Stopping**: Automatic training termination

## 📊 Evaluation Framework

### Component Evaluation

- **Domain Classifier**: Multi-label classification metrics
- **Eligibility Predictor**: Binary classification with thresholds
- **Prolog Engine**: Rule-based reasoning accuracy

### System-Level Metrics

- **Consistency**: Neural-symbolic agreement rates
- **Hybrid Advantage**: Ensemble vs individual performance
- **Error Analysis**: Failure pattern identification

## 🔧 Advanced Features

### Robust Training

- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Linear warmup and decay
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best performing models

### Comprehensive Logging

- **Training Logs**: Per-epoch metrics and progress
- **Error Tracking**: Detailed error reporting
- **Performance Monitoring**: Real-time training statistics
- **Result Persistence**: Automatic result saving

### Production Ready

- **Error Handling**: Comprehensive exception management
- **Resource Cleanup**: Automatic temporary file cleanup
- **Configuration Management**: Flexible parameter tuning
- **Batch Processing**: Efficient large-scale inference

## 🛠️ Development

### Custom Configuration

```python
from hybex_system import HybExConfig

# Custom configuration
config = HybExConfig()
config.TRAINING_CONFIG['epochs']['domain_classification'] = 20
config.MODEL_CONFIG['base_model'] = 'bert-base-uncased'

# Create system with custom config
system = HybExLawSystem()
system.config = config
```

### Adding New Legal Domains

1. Update `ENTITY_CONFIG['domains']` in config.py
2. Add domain-specific training data
3. Update Prolog rules for new domain logic
4. Retrain the domain classifier

### Extending Entity Types

1. Add new entity patterns in `EntityExtractor`
2. Update BIO tagging scheme
3. Modify Prolog fact generation
4. Retrain entity extraction model

## 📋 Requirements

### Core Dependencies

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+

### Optional Dependencies

- SWI-Prolog (for symbolic reasoning)
- spaCy (for enhanced NER)
- matplotlib/seaborn (for visualizations)

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM, GPU with 6GB+ VRAM
- **Training Time**: 2-4 hours (GPU), 8-12 hours (CPU)

## 🎯 Use Cases

### Legal Aid Organizations

- Automated eligibility screening
- Case prioritization and routing
- Resource allocation optimization

### Legal Practitioners

- Initial case assessment
- Client eligibility determination
- Legal advice automation

### Government Services

- Public service delivery
- Policy compliance monitoring
- Legal aid administration

## 🤝 Contributing

We welcome contributions to improve HybEx-Law:

1. **Bug Reports**: Use GitHub issues for bug reports
2. **Feature Requests**: Propose new features and enhancements
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve documentation and examples

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Legal Services Authorities Act, 1987 for legal framework
- Hugging Face Transformers for neural model architecture
- SWI-Prolog community for symbolic reasoning tools
- Indian legal aid organizations for domain expertise

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: See README and inline documentation
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions for general questions

---

**HybEx-Law v1.0.0** - Hybrid Neural-Symbolic Legal AI System
*Making legal aid accessible through AI technology*
