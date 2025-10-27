# HybEx-Law Evaluation Commands

Complete guide for running evaluations and ablation studies in the HybEx-Law system.

---

## 📋 Table of Contents
1. [Quick Start Commands](#quick-start-commands)
2. [Standard Evaluation](#standard-evaluation)
3. [Ablation Study](#ablation-study)
4. [Advanced Options](#advanced-options)
5. [Output Files](#output-files)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start Commands

### **Standard Hybrid Evaluation**
```bash
# PowerShell (Windows)
cd "d:\kyuga\Projects\VIT\NLP Project HybEx-Law\multi_domain_legal"
python -m hybex_system.main evaluate_hybrid

# Or with explicit Python path
python -m hybex_system.main evaluate_hybrid --test-data data/test_split.json
```

### **Hybrid Evaluation + Ablation Study** ⭐
```bash
# PowerShell (Windows) - RECOMMENDED FOR FULL ANALYSIS
cd "d:\kyuga\Projects\VIT\NLP Project HybEx-Law\multi_domain_legal"
python -m hybex_system.main evaluate_hybrid --ablation

# With custom test data
python -m hybex_system.main evaluate_hybrid --test-data data/val_split.json --ablation
```

---

## 📊 Standard Evaluation

### Basic Command
```bash
python -m hybex_system.main evaluate_hybrid
```

**What it does:**
- Evaluates the complete hybrid system (Prolog + GNN + BERT)
- Tests ensemble decision fusion
- Generates performance metrics (Accuracy, Precision, Recall, F1)
- Creates method-wise breakdown (Prolog, GNN, BERT, Ensemble, Fallback)
- Saves results to `results/evaluation_results/`

### With Custom Test Data
```bash
python -m hybex_system.main evaluate_hybrid --test-data path/to/your/test_data.json
```

### With Custom Config
```bash
python -m hybex_system.main evaluate_hybrid --config path/to/custom_config.json
```

---

## 🔬 Ablation Study

### **What is Ablation Study?**
Ablation study systematically removes components to identify their contribution to overall performance. It tests **ALL 17 model combinations** to find the optimal ensemble.

### **Full Command with All Options**
```bash
python -m hybex_system.main evaluate_hybrid \
    --test-data data/test_split.json \
    --ablation \
    --config config/custom_config.json
```

### **PowerShell (Windows) - Multi-line**
```powershell
python -m hybex_system.main evaluate_hybrid `
    --test-data data/test_split.json `
    --ablation
```

### **Model Combinations Tested (17 Total)**

#### **Singles (5)**
1. `01_prolog` - Prolog reasoning only
2. `02_gnn` - Knowledge Graph Neural Network only
3. `03_domain` - Domain classifier only
4. `04_eligibility` - Eligibility predictor only
5. `05_enhanced_bert` - Enhanced Legal BERT only

#### **Pairs (6)**
6. `06_domain_eligibility` - Domain + Eligibility
7. `07_domain_bert` - Domain + Enhanced BERT
8. `08_eligibility_bert` - Eligibility + Enhanced BERT
9. `09_gnn_domain` - GNN + Domain
10. `10_gnn_eligibility` - GNN + Eligibility
11. `11_gnn_bert` - GNN + Enhanced BERT

#### **With Prolog (3)**
12. `12_prolog_domain` - Prolog + Domain
13. `13_prolog_eligibility` - Prolog + Eligibility
14. `14_prolog_gnn` - Prolog + GNN

#### **Ensembles (3)**
15. `15_all_neural` - All neural models (Domain + Eligibility + BERT + GNN)
16. `16_hybrid_best` - Best hybrid (Prolog + Eligibility + GNN)
17. `17_full_ensemble` - ALL models (Prolog + Domain + Eligibility + GNN + BERT)

---

## ⚙️ Advanced Options

### **Custom Configuration File**
```bash
python -m hybex_system.main evaluate_hybrid --config my_config.json
```

Example `my_config.json`:
```json
{
    "MODEL_CONFIG": {
        "batch_size": 16,
        "max_length": 512
    },
    "EVAL_CONFIG": {
        "max_hybrid_samples": 1000,
        "max_gnn_samples": 1000
    }
}
```

### **Using Validation Split Instead of Test**
```bash
python -m hybex_system.main evaluate_hybrid --test-data data/val_split.json --ablation
```

---

## 📁 Output Files

### **Standard Evaluation Output**
```
results/
├── evaluation_results/
│   └── hybrid_results_20251028_143025.json
└── advanced_evaluation/
    ├── evaluation_report.json
    ├── evaluation_report.md
    └── visualizations/
        ├── confusion_matrix.png
        ├── calibration_curve.png
        ├── stratified_income_level.png
        └── method_comparison.png
```

### **Ablation Study Output**
```
results/
└── ablation_study/
    ├── ablation_results_20251028_143530.csv     # Summary table
    ├── ablation_results_20251028_143530.json    # Full results
    ├── ablation_complete_20251028_143530.csv    # Complete with rankings
    └── ablation_report_20251028_143530.txt      # Human-readable report
```

### **CSV Output Format**
```csv
Combination,Models,Accuracy,Precision,Recall,F1,Samples
01_prolog,prolog,85.23%,87.45%,83.12%,0.8524,1500
17_full_ensemble,prolog + domain_classifier + eligibility_predictor + gnn + enhanced_legal_bert,94.67%,96.12%,93.45%,0.9477,1500
...
```

---

## 🔍 Checking Results

### **View Results**
```bash
# View ablation CSV (sorted by performance)
cat results/ablation_study/ablation_complete_*.csv

# View ablation report (human-readable)
cat results/ablation_study/ablation_report_*.txt

# View hybrid evaluation JSON
cat results/evaluation_results/hybrid_results_*.json
```

### **PowerShell (Windows)**
```powershell
# View latest ablation results
Get-Content (Get-ChildItem results\ablation_study\ablation_report_*.txt | Sort-Object LastWriteTime -Descending | Select-Object -First 1)

# View latest hybrid results
Get-Content (Get-ChildItem results\evaluation_results\hybrid_results_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1) | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## 🛠️ Troubleshooting

### **Issue: Models Not Found**
```bash
# Check if models exist
ls models/hybex_system/domain_classifier/model.pt
ls models/hybex_system/eligibility_predictor/model.pt
ls models/hybex_system/gnn_model/model.pt

# If missing, train first:
python -m hybex_system.main train --data-dir data/
```

### **Issue: Out of Memory (GPU)**
Edit config to reduce batch size:
```json
{
    "MODEL_CONFIG": {
        "batch_size": 8  // Reduce from 16
    }
}
```

Or use CPU only:
```bash
# Set environment variable
$env:CUDA_VISIBLE_DEVICES=""
python -m hybex_system.main evaluate_hybrid --ablation
```

### **Issue: Test Data Not Found**
```bash
# Check data directory
ls data/test_split.json
ls data/val_split.json

# Use validation split if test missing
python -m hybex_system.main evaluate_hybrid --test-data data/val_split.json --ablation
```

### **Issue: Module Import Error**
```bash
# Ensure you're in the correct directory
cd "d:\kyuga\Projects\VIT\NLP Project HybEx-Law\multi_domain_legal"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall dependencies if needed
pip install -r requirements.txt
```

---

## 📈 Example Complete Workflow

### **Full Evaluation Pipeline**

```bash
# 1. Navigate to project directory
cd "d:\kyuga\Projects\VIT\NLP Project HybEx-Law\multi_domain_legal"

# 2. Verify models exist
ls models/hybex_system/*/model.pt

# 3. Run standard evaluation first (quick check)
python -m hybex_system.main evaluate_hybrid

# 4. If successful, run full ablation study
python -m hybex_system.main evaluate_hybrid --ablation

# 5. Check results
cat results/ablation_study/ablation_report_*.txt | Select-Object -Last 1

# 6. View visualizations (if generated)
explorer results/advanced_evaluation/
```

---

## 📝 Performance Expectations

### **Execution Time**
- **Standard Evaluation**: ~5-10 minutes (1500 samples)
- **Ablation Study**: ~30-60 minutes (17 combinations × 1500 samples)

### **Memory Requirements**
- **CPU**: ~8GB RAM
- **GPU**: ~6GB VRAM (recommended)
- **Disk**: ~2GB for results and logs

### **Expected Accuracy Range**
- **Prolog Only**: 82-87%
- **GNN Only**: 85-90%
- **BERT Models**: 88-93%
- **Hybrid Ensemble**: 92-96%
- **Full Ensemble**: 94-97%

---

## 🎯 Recommended Commands

### **For Quick Testing**
```bash
python -m hybex_system.main evaluate_hybrid
```

### **For Research/Paper** ⭐
```bash
python -m hybex_system.main evaluate_hybrid --ablation
```

### **For Production Validation**
```bash
python -m hybex_system.main evaluate_hybrid --test-data data/test_split.json
```

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Training Guide**: See `TRAINING_GUIDE.md`
- **API Reference**: See `docs/API.md`
- **Configuration Options**: See `hybex_system/config.py`

---

## 🆘 Support

If you encounter issues:
1. Check logs in `logs/` directory
2. Verify all dependencies: `pip list | grep -E "torch|transformers|sklearn"`
3. Ensure all models are trained
4. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Generated**: October 28, 2025  
**System**: HybEx-Law v1.0  
**Author**: VIT NLP Project Team
