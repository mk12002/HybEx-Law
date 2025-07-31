# HybEx-Law: A Hybrid NLP Framework for Verifying Eligibility for Legal Aid Services

A neuro-symbolic system that allows people to describe their legal situation in plain language and receive reliable, explainable determinations of their eligibility for legal aid services.

## Project Overview

This project combines neural NLP components with symbolic reasoning (Prolog) to:
- Extract legal facts from natural language queries
- Apply formal legal rules for eligibility determination
- Provide explainable decisions about legal aid qualification

## Architecture

### Two-Stage Hybrid Pipeline:
1. **Stage 1**: Coarse-grained classifier to identify relevant entities
2. **Stage 2**: High-precision extractors for specific legal facts
3. **Prolog Engine**: Formal rule application and decision making

## Project Structure

```
├── src/
│   ├── nlp_pipeline/          # Main NLP processing components
│   ├── extractors/            # Stage 2 specialized extractors
│   ├── prolog_engine/         # Prolog integration and knowledge base
│   ├── evaluation/            # Evaluation framework
│   └── utils/                 # Utility functions
├── data/
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Processed datasets
│   └── annotations/           # Annotated training data
├── knowledge_base/
│   └── legal_aid_eligibility.pl  # Prolog rules
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install SWI-Prolog:**
   - Download from: https://www.swi-prolog.org/download/stable
   - Ensure it's in your system PATH

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the example:**
   ```bash
   python src/main.py --query "I lost my job and have no income. My landlord is evicting me. I am from general category."
   ```

## Features

- **Natural Language Processing**: Extract legal facts from conversational queries
- **Symbolic Reasoning**: Apply formal legal rules using Prolog
- **Explainable AI**: Provide clear explanations for eligibility decisions
- **Evaluation Framework**: Comprehensive metrics for pipeline assessment
- **Modular Design**: Easy to extend with new extractors and rules

## Research Questions

1. How can a hybrid NLP pipeline accurately extract legal facts from conversational queries?
2. Does HybEx-Law outperform single LLM approaches in accuracy and reliability?
3. How effective is the Prolog verdict as a metric for evaluating NLP pipeline performance?

## Contributing

This project is part of ongoing research in legal AI and access to justice. Contributions are welcome!

## License

MIT License - see LICENSE file for details.
