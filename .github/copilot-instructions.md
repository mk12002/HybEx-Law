<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# HybEx-Law Project Instructions

This is a hybrid NLP framework for legal aid eligibility verification that combines neural language processing with symbolic reasoning using Prolog.

## Project Context
- **Domain**: Legal technology and access to justice
- **Architecture**: Two-stage hybrid NLP pipeline + Prolog reasoning engine
- **Goal**: Extract legal facts from natural language and determine legal aid eligibility

## Key Components
1. **Stage 1 Classifier**: Multi-label classifier to identify present entities (income, case_type, social_status)
2. **Stage 2 Extractors**: High-precision extractors for specific legal facts
3. **Prolog Engine**: Formal rule application based on Legal Services Authorities Act, 1987

## Code Style Guidelines
- Use clear, descriptive variable names that reflect legal concepts
- Add docstrings explaining the legal significance of functions
- Include type hints for better code clarity
- Follow PEP 8 conventions
- Add comments explaining complex legal logic

## Legal Domain Considerations
- Income thresholds and financial eligibility criteria
- Categorical eligibility (SC/ST, women, children, industrial workers)
- Case type classifications and exclusions
- Location-specific rules and variations

## Technical Preferences
- Use scikit-learn for traditional ML components
- Leverage spaCy for NLP preprocessing
- Implement robust error handling for Prolog integration
- Create comprehensive evaluation metrics
- Maintain explainability in all components

## Testing Requirements
- Unit tests for each extractor component
- Integration tests for full pipeline
- Evaluation scripts with multiple baselines
- Edge case handling for legal scenarios
