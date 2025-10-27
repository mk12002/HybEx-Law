# HybEx-Law: Complete Project Context & Architecture

**Last Updated:** October 27, 2025  
**Repository:** HybEx-Law (Owner: mk12002, Branch: main)

---

## 🎯 Project Overview

**HybEx-Law** is a sophisticated hybrid neural-symbolic legal AI system designed to determine **legal aid eligibility** under the **Legal Services Authorities Act, 1987** in India. The system combines:

1. **Neural Networks** (BERT-based models) for natural language understanding
2. **Symbolic Reasoning** (Prolog rules) for legal logic
3. **Graph Neural Networks** (GAT-based) for knowledge graph reasoning

### Core Functionality

**Input:**
```
"I am a woman earning 15000 rupees monthly facing domestic violence"
```

**Output:**
```json
{
  "eligible": true,
  "confidence": 0.95,
  "method": "hybrid",
  "primary_reason": "Eligible as a woman applicant",
  "applicable_rules": ["women_categorical_eligibility", "domestic_violence_case_coverage"]
}
```

---

## 🏗️ System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      HybEx-Law System                            │
│                         (main.py)                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
│  Data Layer   │  │ Neural Layer │  │  Symbolic Layer  │
│               │  │              │  │                  │
│ - Processor   │  │ - BERT       │  │ - Prolog Engine  │
│ - Scraper     │  │ - GNN (GAT)  │  │ - Knowledge Graph│
│ - Augmenter   │  │ - Classifier │  │ - 162 Rules      │
└───────────────┘  └──────────────┘  └──────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  ┌─────────────────┐
                  │ Hybrid Predictor│
                  │  (Fusion Layer) │
                  └─────────────────┘
```

---

## 📁 File Structure & Component Mapping

### Core System Files (`hybex_system/`)

#### 1. **`main.py`** (1002 lines)
- **Purpose:** Main system interface and orchestrator
- **Key Classes:**
  - `HybExLawSystem`: Central system class
- **Key Features:**
  - Component initialization (trainer, evaluator, data processor)
  - Lazy-loading of models (domain classifier, eligibility predictor)
  - Status checking and system configuration
  - CLI interface for training, evaluation, and prediction

#### 2. **`config.py`** (381 lines)
- **Purpose:** Centralized configuration management
- **Key Components:**
  - `HybExConfig` class
- **Configuration Sections:**
  - Model configs (BERT, domain classifier, eligibility predictor, enhanced BERT)
  - Data configs (train/val/test splits, preprocessing parameters)
  - Entity configs (income thresholds, social categories, case types)
  - Prolog configs (reasoning parameters, fusion weights)
  - GNN configs (graph neural network parameters)
  - Fusion configs (hybrid decision-making thresholds)

**Key Income Thresholds:**
```python
'income_thresholds': {
    'general': 300000,      # ₹3 lakhs (NALSA 2024)
    'sc_st': 800000,        # ₹8 lakhs (Enhanced for SC/ST)
    'obc': 600000,          # ₹6 lakhs
    'bpl': 0,               # No income limit for BPL
    'ews': 800000           # ₹8 lakhs (EWS category)
}
```

#### 3. **`data_processor.py`** (1017 lines)
- **Purpose:** Data preprocessing and feature extraction
- **Key Classes:**
  - `EnhancedFeatureExtractor`: Semantic feature extraction
  - `DataPreprocessor`: Main data processing pipeline
- **Features Extracted:**
  - Financial indicators (income, debt, savings)
  - Urgency signals (high/medium/low)
  - Implicit vulnerabilities (health, social, family, age, disaster)
  - Domain keywords (consumer, employment, family, rights)
  - Named entities (persons, organizations, locations, dates)
  - Sentiment analysis

#### 4. **`neural_models.py`** (1325 lines)
- **Purpose:** Neural network architectures
- **Key Classes:**
  - `DomainClassifier`: Multi-label domain classification
  - `EligibilityPredictor`: Binary eligibility prediction
  - `EnhancedLegalBERT`: Multi-task learning (domain + eligibility)
  - `LegalDataset`: PyTorch dataset for legal text
  - `ModelTrainer`: Training orchestrator
  - `ModelMetrics`: Evaluation metrics dataclass

**Model Architectures:**
- Base Model: `nlpaueb/legal-bert-base-uncased`
- Max Length: 256 tokens (optimized for GPU memory)
- Batch Size: 2-4 (reduced for memory efficiency)
- Dropout: 0.3
- Learning Rates: 1e-5 to 5e-5

#### 5. **`prolog_engine.py`** (2283 lines)
- **Purpose:** Symbolic reasoning with Prolog
- **Key Classes:**
  - `PrologEngine`: Main reasoning engine
  - `LegalReasoning`: Result structure
  - `PrologQuery`: Query structure
- **Features:**
  - SWI-Prolog integration with subprocess fallback
  - 162 comprehensive legal rules across 5 domains
  - Fallback reasoning when Prolog unavailable
  - Domain-specific rule loading
  - Confidence scoring for symbolic predictions

**Legal Domains Covered:**
1. Legal Aid (7 rules)
2. Family Law (17 rules)
3. Consumer Protection (12 rules)
4. Employment Law (47 rules)
5. Fundamental Rights (69 rules)

#### 6. **`knowledge_graph_engine.py`** (833 lines)
- **Purpose:** Graph Neural Network for knowledge reasoning
- **Key Classes:**
  - `LegalGAT`: Graph Attention Network model
  - `KnowledgeGraphEngine`: Graph construction and inference
- **Features:**
  - GAT-based architecture (Graph Attention Networks)
  - Entity-to-predicate mapping (30+ entities)
  - Feature normalization for income, age, social category
  - Graph construction from Prolog rules
  - Training and inference on legal knowledge graph

**GNN Architecture:**
- Hidden Channels: 128
- Attention Heads: 8
- Dropout: 0.3
- Layers: 2 GAT layers + readout layer

#### 7. **`hybrid_predictor.py`** (1081 lines)
- **Purpose:** Intelligent fusion of neural and symbolic predictions
- **Key Classes:**
  - `IntelligentHybridPredictor`: Main hybrid decision maker
  - `ConfidenceCalibrator`: Calibration using isotonic regression
  - `HybridPrediction`: Result structure
- **Fusion Strategies:**
  - High-confidence consensus
  - Prolog-neural agreement
  - Weighted ensemble fallback
  - Uncertainty quantification
  - Dynamic method selection

**Ensemble Weights:**
```python
'prolog_weight': 0.15,
'bert_weight': 0.40,
'gnn_weight': 0.35,
'domain_weight': 0.05,
'enhanced_bert_weight': 0.05
```

#### 8. **`trainer.py`** (1742 lines)
- **Purpose:** Complete training pipeline orchestration
- **Key Classes:**
  - `TrainingOrchestrator`: Main training coordinator
- **Training Stages:**
  1. Data validation and preprocessing
  2. Neural model training (domain classifier, eligibility predictor)
  3. GNN model training
  4. Enhanced BERT training (multi-task)
  5. Prolog rule validation
  6. Integrated evaluation
- **Features:**
  - Multi-stage pipeline with progress tracking
  - GPU memory management
  - Early stopping and checkpointing
  - Comprehensive logging

#### 9. **`evaluator.py`** (2031 lines)
- **Purpose:** Comprehensive model evaluation
- **Key Classes:**
  - `ModelEvaluator`: Main evaluation orchestrator
  - `EvaluationResults`: Results dataclass
  - `AblationDataset`: Dataset for ablation studies
- **Evaluation Components:**
  - Neural model evaluation (accuracy, F1, precision, recall)
  - Prolog reasoning evaluation
  - GNN model evaluation
  - Hybrid system evaluation
  - Component agreement analysis
  - Error analysis with examples

#### 10. **`advanced_evaluator.py`** (753 lines)
- **Purpose:** Advanced evaluation metrics and visualizations
- **Key Classes:**
  - `AdvancedEvaluator`: Comprehensive evaluation suite
- **Evaluation Features:**
  - Stratified analysis (by income, category, vulnerability)
  - Confidence calibration curves
  - Expected Calibration Error (ECE)
  - Fairness metrics
  - Error pattern detection
  - Production-ready monitoring

#### 11. **`master_scraper.py`** (1188 lines)
- **Purpose:** Legal knowledge scraping from Indian government websites
- **Key Classes:**
  - `MasterLegalScraper`: Unified scraping system
  - `LegalWebsite`: Website metadata structure
  - `ExtractedLegalContent`: Scraped content structure
- **Priority Websites:**
  1. India Code (indiacode.nic.in) - Central Acts Repository
  2. NALSA (nalsa.gov.in) - Legal aid schemes and thresholds
  3. E-Gazette India (egazette.gov.in) - Notifications and amendments
  4. PRS India (prsindia.org) - Legislative research
  5. Supreme Court (sci.gov.in) - Judgments and case law
- **Features:**
  - Multi-threaded scraping
  - Rate limiting and SSL handling
  - SQLite database storage
  - YAML knowledge base integration
  - Content deduplication

---

## 🧠 Model Details

### Neural Models

#### 1. **Domain Classifier**
- **Task:** Multi-label classification
- **Architecture:** BERT + Linear classifier
- **Output:** 5 legal domains (legal_aid, family_law, consumer_protection, employment_law, fundamental_rights)
- **Loss:** BCEWithLogitsLoss
- **Metrics:** Macro F1, Accuracy

#### 2. **Eligibility Predictor**
- **Task:** Binary classification
- **Architecture:** BERT + Linear classifier
- **Output:** Eligible (1) or Not Eligible (0)
- **Loss:** BCEWithLogitsLoss
- **Metrics:** Binary F1, Accuracy, Precision, Recall

#### 3. **Enhanced Legal BERT**
- **Task:** Multi-task learning
- **Architecture:** BERT + 2 classification heads
- **Outputs:** 
  - Domain classification (5 classes)
  - Eligibility prediction (2 classes)
- **Loss Weights:**
  - Eligibility: 0.7
  - Domain: 0.2
  - Confidence: 0.1

#### 4. **Knowledge Graph GNN (GAT)**
- **Task:** Node classification on legal knowledge graph
- **Architecture:** 2-layer Graph Attention Network
- **Features:**
  - Structural features (one-hot node encoding)
  - Entity features (income, age, social category)
- **Output:** Eligibility prediction with graph reasoning

### Symbolic Reasoning

#### Prolog Engine
- **Rules:** 162 comprehensive legal rules
- **Reasoning:** Backward chaining with Prolog
- **Fallback:** Python-based rule evaluation when Prolog unavailable
- **Confidence:** Rule-based confidence scoring

---

## 📊 Data Pipeline

### Data Sources
1. **Scraped Data:** Legal websites (NALSA, India Code, etc.)
2. **Generated Data:** Synthetic training data with realistic scenarios
3. **Validated Data:** Manually reviewed test cases

### Data Split
- **Training:** 70% (14K samples)
- **Validation:** 15% (3.5K samples)
- **Test:** 15% (3.5K samples)

### Data Structure
```json
{
  "sample_id": "unique_id",
  "query": "Natural language legal query",
  "domains": ["legal_aid", "family_law"],
  "expected_eligibility": true,
  "extracted_entities": {
    "income": 15000,
    "social_category": "general",
    "age": 35,
    "gender": "female",
    "case_type": "domestic_violence"
  }
}
```

---

## 🔄 Training Workflow

```
1. Data Preprocessing
   ├─ Load raw data
   ├─ Extract entities and features
   ├─ Validate and clean
   └─ Split into train/val/test

2. Neural Model Training
   ├─ Domain Classifier
   ├─ Eligibility Predictor
   └─ Enhanced Legal BERT

3. GNN Training
   ├─ Build knowledge graph
   ├─ Create case graphs
   └─ Train GAT model

4. Prolog Rule Loading
   └─ Load 162 rules from knowledge base

5. Hybrid System Integration
   ├─ Confidence calibration
   └─ Ensemble weight tuning

6. Evaluation
   ├─ Neural metrics
   ├─ Prolog metrics
   ├─ Hybrid metrics
   └─ Error analysis
```

---

## 🎯 Prediction Workflow

```
User Query
    ↓
1. Entity Extraction (data_processor)
    ↓
2. Parallel Predictions:
   ├─ BERT Eligibility Predictor → Confidence A
   ├─ Prolog Engine → Confidence B
   ├─ GNN Model → Confidence C
   ├─ Domain Classifier → Confidence D
   └─ Enhanced BERT → Confidence E
    ↓
3. Confidence Calibration
    ↓
4. Hybrid Decision (hybrid_predictor)
   ├─ High-confidence consensus?
   ├─ Prolog-neural agreement?
   └─ Weighted ensemble
    ↓
5. Final Prediction
   ├─ Eligible/Not Eligible
   ├─ Confidence score
   ├─ Method used
   ├─ Reasoning explanation
   └─ Uncertainty quantification
```

---

## 🚀 Usage Examples

### Training
```bash
python -m hybex_system.main train --data-dir data/organized
```

### Evaluation
```bash
python -m hybex_system.main evaluate --test-data data/test_split.json
```

### Single Prediction
```python
from hybex_system import HybExLawSystem

system = HybExLawSystem()
result = system.predict({
    "query": "I am a woman earning 15000 rupees monthly facing domestic violence",
    "extracted_entities": {
        "income": 15000,
        "gender": "female",
        "case_type": "domestic_violence"
    }
})
print(f"Eligible: {result.eligible}, Confidence: {result.confidence}")
```

### Batch Prediction
```python
system = HybExLawSystem()
test_samples = load_test_data("data/test_split.json")
results = system.batch_predict(test_samples)
```

---

## 📈 Performance Metrics

### Expected Performance (based on architecture)
- **Neural Models:** 85-90% accuracy
- **Prolog Engine:** 75-80% accuracy (rule coverage dependent)
- **GNN Model:** 80-85% accuracy
- **Hybrid System:** 90-95% accuracy (ensemble benefit)

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Matthews Correlation Coefficient (MCC)
- Expected Calibration Error (ECE)
- Confusion Matrix
- Stratified metrics (by income, category, vulnerability)

---

## 🛠️ Technical Stack

### Core Libraries
- **PyTorch:** Neural network training
- **Transformers (Hugging Face):** BERT models
- **PyTorch Geometric:** Graph neural networks
- **SWI-Prolog:** Symbolic reasoning
- **spaCy:** NLP and entity extraction
- **scikit-learn:** Metrics and calibration
- **pandas, numpy:** Data manipulation
- **matplotlib, seaborn:** Visualization

### Model Details
- **Base Model:** `nlpaueb/legal-bert-base-uncased`
- **Hidden Size:** 768 (BERT)
- **GNN Hidden:** 128
- **Attention Heads:** 8 (GAT)

---

## 📝 Key Insights

### Strengths
1. **Hybrid Approach:** Combines neural flexibility with symbolic interpretability
2. **Multi-Domain:** Covers 5 legal domains with 162 rules
3. **Confidence Calibration:** Isotonic regression for reliable confidence scores
4. **Stratified Evaluation:** Fairness analysis across demographics
5. **Production-Ready:** Comprehensive logging, error handling, and monitoring

### Challenges
1. **GPU Memory:** Large BERT models require optimization
2. **Prolog Availability:** Fallback reasoning when Prolog not installed
3. **Data Imbalance:** SC/ST categories have different eligibility rates
4. **Rule Coverage:** Prolog rules may not cover all edge cases
5. **Ensemble Tuning:** Balancing weights between systems

### Future Enhancements
1. **Active Learning:** Identify uncertain cases for human review
2. **Explainability:** Attention visualization and rule tracing
3. **Multilingual:** Support for regional Indian languages
4. **Real-time Updates:** Dynamic threshold updates from NALSA
5. **API Deployment:** REST API for production use

---

## 🔐 Critical Configuration

### Paths
- **Base Directory:** `d:\kyuga\Projects\VIT\NLP Project HybEx-Law\multi_domain_legal`
- **Knowledge Base:** `knowledge_base/knowledge_base.pl`
- **Models:** `models/hybex_system/`
- **Data:** `data/`
- **Results:** `results/`

### Model Files
- Domain Classifier: `models/hybex_system/domain_classifier/model.pt`
- Eligibility Predictor: `models/hybex_system/eligibility_predictor/model.pt`
- Enhanced BERT: `models/hybex_system/enhanced_legal_bert/enhanced_legal_bert_best.pt`
- GNN Model: `models/hybex_system/gnn_model/gnn_model.pt`

---

## 📞 Key Contact Points

- **Repository Owner:** mk12002
- **Current Branch:** main
- **Last Update:** October 27, 2025

---

## 🎓 Academic Context

This is a research project for **VIT** (Vellore Institute of Technology) focused on:
- **NLP for Legal AI**
- **Hybrid Neural-Symbolic Reasoning**
- **Fairness in Legal Decision-Making**
- **Knowledge Graph Applications in Law**

---

**End of Context Summary**
