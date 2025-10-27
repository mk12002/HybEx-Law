# 🚀 HybEx-Law Quick Command Reference

## Essential Commands

### **Run Evaluation with Ablation Study** (RECOMMENDED)
```bash
cd "d:\kyuga\Projects\VIT\NLP Project HybEx-Law\multi_domain_legal"
python -m hybex_system.main evaluate_hybrid --ablation
```

### **Run Standard Evaluation Only**
```bash
python -m hybex_system.main evaluate_hybrid
```

### **With Custom Test Data**
```bash
python -m hybex_system.main evaluate_hybrid --test-data data/val_split.json --ablation
```

---

## What Each Command Does

| Command | What It Tests | Output Location | Time |
|---------|--------------|-----------------|------|
| `evaluate_hybrid` | Hybrid system (Prolog+GNN+BERT) | `results/evaluation_results/` | ~5-10 min |
| `evaluate_hybrid --ablation` | **ALL 17 combinations** | `results/ablation_study/` | ~30-60 min |

---

## Output Files

### Ablation Study Results
```
results/ablation_study/
├── ablation_results_TIMESTAMP.csv    ← Open this for quick view
├── ablation_report_TIMESTAMP.txt     ← Human-readable report
└── ablation_complete_TIMESTAMP.csv   ← Full ranked results
```

### View Results (PowerShell)
```powershell
# View latest report
cat (ls results\ablation_study\ablation_report_*.txt)[-1]

# View CSV in Excel
start (ls results\ablation_study\ablation_complete_*.csv)[-1]
```

---

## 17 Model Combinations Tested

✅ **Singles**: Prolog, GNN, Domain, Eligibility, Enhanced BERT (5)  
✅ **Pairs**: Various 2-model combos (6)  
✅ **With Prolog**: Prolog + other models (3)  
✅ **Ensembles**: Multi-model combinations (3)  

**Total**: 17 combinations tested automatically

---

## Troubleshooting One-Liners

```bash
# Check models exist
ls models/hybex_system/*/model.pt

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU if GPU fails
$env:CUDA_VISIBLE_DEVICES=""
python -m hybex_system.main evaluate_hybrid --ablation

# View logs
cat logs/model_evaluation.log
```

---

## Expected Performance

- **Prolog Only**: ~85%
- **GNN Only**: ~88%
- **BERT Only**: ~91%
- **Hybrid Ensemble**: ~94%
- **Full Ensemble**: ~96% ⭐

---

📖 **Full Documentation**: `EVALUATION_COMMANDS.md`
