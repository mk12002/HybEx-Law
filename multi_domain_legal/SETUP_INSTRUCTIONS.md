# HybEx-Law System Setup Instructions

## Critical Prerequisites

### 1. Install SWI-Prolog (REQUIRED)
The system cannot function without SWI-Prolog for symbolic reasoning.

**Windows Installation:**
1. Download SWI-Prolog from: https://www.swi-prolog.org/download/stable
2. Install using the Windows installer
3. Add SWI-Prolog to your system PATH
4. Verify installation: `swipl --version`

**Alternative (if you have Chocolatey):**
```bash
choco install swi-prolog
```

### 2. Install Python Dependencies
```bash
pip install pyswip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Execution Order

### Phase 1: Environment Setup
```bash
# Verify all dependencies
python -c "
import torch, transformers, spacy, pyswip
print('All dependencies available!')
"
```

### Phase 2: Data Generation
```bash
python scripts/comprehensive_data_generation.py
```

### Phase 3: Knowledge Update
```bash
python -m hybex_system.main update_knowledge
```

### Phase 4: Data Preprocessing
```bash
python -m hybex_system.main preprocess --data-dir data/
```

### Phase 5: System Training
```bash
python -m hybex_system.main train --data-dir data/
```

### Phase 6: Evaluation
```bash
python -m hybex_system.main evaluate
python -m hybex_system.main status
```

### Phase 7: Testing
```bash
python -m pytest comprehensive_system_test.py -v
python -m hybex_system.main predict --query "Test query here"
```

## Troubleshooting

### SWI-Prolog Issues
- Ensure SWI-Prolog is in PATH
- Restart terminal after installation
- Try: `where swipl` (Windows) to verify installation

### Memory Issues
- Reduce batch sizes in config.py if encountering OOM errors
- Monitor GPU memory usage during training

### Data Issues
- Ensure JSON data files are properly formatted
- Check data validation logs in logs/ directory
