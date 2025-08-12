# HybEx-Law: Hybrid Neural-Symbolic Legal AI System

HybEx-Law is an advanced legal AI system that combines neural (deep learning) and symbolic (Prolog-based) reasoning for multi-domain legal aid eligibility and legal query analysis. It is designed for transparency, explainability, and extensibility across multiple legal domains.

---

## Table of Contents

- [System Overview](#system-overview)
- [Directory Structure](#directory-structure)
- [Data & Knowledge Sources](#data--knowledge-sources)
- [Models](#models)
- [Prolog Symbolic System](#prolog-symbolic-system)
- [How to Run](#how-to-run)
- [API Usage Example](#api-usage-example)
- [Requirements](#requirements)
- [Evaluation & Results](#evaluation--results)
- [Logging & Outputs](#logging--outputs)
- [Extending the System](#extending-the-system)

---

## System Overview

HybEx-Law integrates:
- **Neural Models**: Transformer-based models for domain classification and eligibility prediction.
- **Symbolic Reasoning**: Prolog engine for rule-based legal reasoning, using a curated knowledge base.
- **Hybrid Fusion**: Combines neural and symbolic outputs for robust, explainable decisions.
- **Comprehensive Scraping**: Automated legal data collection from authoritative Indian legal sources.

---

## Directory Structure

```
hybex_system/         # Main system modules (core logic)
knowledge_base/       # Prolog rules and Python rule definitions
models/               # Trained neural models
data/                 # Datasets, scraped data, and stats
logs/                 # System and component logs
results/              # Evaluation results and plots
scripts/              # Utility and scraping scripts
requirements.txt      # Python dependencies
README.md             # This file
```

**Key modules:**
- `main.py`: System entry point and API
- `prolog_engine.py`: Prolog-based symbolic reasoning
- `neural_models.py`: Transformer-based neural models
- `legal_scraper.py`: Scrapes legal data from NALSA, DoJ, IndiaCode, etc.
- `trainer.py`, `evaluator.py`: Training and evaluation orchestration

---

## Data & Knowledge Sources

- **Scraped Sites**:  
  - [NALSA](https://nalsa.gov.in/legal-aid-schemes) (National Legal Services Authority)
  - [Department of Justice](https://doj.gov.in/legal-aid-schemes, https://doj.gov.in/initiatives)
  - [India Code](https://www.indiacode.nic.in/) (Acts, e.g., Legal Services Authorities Act, 1987)
- **Knowledge Base**:  
  - Prolog files in `knowledge_base/` (e.g., `foundational_rules_clean.pl`, `legal_aid_clean_v2.pl`)
  - Python rules in `multi_domain_rules.py`
- **Data**:  
  - JSON datasets for training, validation, and testing
  - SQLite database (`data/legal_knowledge.db`) for scraped legal rules

---

## Models

- **Domain Classifier**: Multi-label transformer model (e.g., BERT) for legal domain detection.
- **Eligibility Predictor**: Binary classifier for legal aid eligibility.
- **Datasets**: Located in `data/`, with splits for training, validation, and testing.

---

## Prolog Symbolic System

- Uses SWI-Prolog for rule-based legal reasoning.
- Loads rules from curated `.pl` files and Python-generated rules.
- Handles domain-specific and cross-domain legal logic.
- Provides detailed explanations, rule traces, and legal citations for each decision.

---

## How to Run

### 1. Install Requirements

```sh
pip install -r requirements.txt
```
> **Note:** SWI-Prolog must be installed and available in your system PATH.

### 2. Scrape/Update Legal Knowledge

```python
from hybex_system.main import HybExLawSystem
system = HybExLawSystem()
system.update_legal_knowledge()  # Scrapes and updates the knowledge base
```

### 3. Preprocess Data

```python
system.preprocess_data("data/")
```

### 4. Train the System

```python
system.train_complete_system("data/")
```

### 5. Evaluate the System

```python
system.evaluate_system()
```

### 6. Predict Legal Eligibility (API Example)

```python
result = system.predict_legal_eligibility("I need legal aid for a custody case")
print(result['final_decision'])
```

---

## API Usage Example

```python
from hybex_system.main import HybExLawSystem

system = HybExLawSystem()
result = system.predict_legal_eligibility("I need legal aid for a custody case")
print(f"Eligible: {result['final_decision']['eligible']}")
print(f"Explanation: {result['final_decision']['explanation']}")
```

---

## Requirements

- Python 3.8+
- SWI-Prolog (for symbolic reasoning)
- See `requirements.txt` for Python dependencies (transformers, torch, pyswip, spacy, etc.)

---

## Evaluation & Results

- Evaluation scripts generate metrics (accuracy, F1, confusion matrices) for both neural and symbolic components.
- Results and plots are saved in the `results/` directory.

---

## Logging & Outputs

- All logs are saved in the `logs/` directory.
- Scraping, training, evaluation, and Prolog reasoning have dedicated log files.

---

## Extending the System

- **Add new legal domains**: Update Prolog rules and retrain neural models.
- **Add new data sources**: Extend `legal_scraper.py` or add new scripts in `scripts/`.
- **Improve models**: Swap transformer backbones or tune hyperparameters in `config.py`.

---

For further details, see the code and docstrings in each module. This README provides a high-level overview and step-by-step instructions for running and extending the HybEx-Law system.
