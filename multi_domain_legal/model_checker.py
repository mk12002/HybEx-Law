import torch
from hybex_system.neural_models import DomainClassifier, EligibilityPredictor
from hybex_system.config import HybExConfig

config = HybExConfig()
# Verify Domain Classifier
dc_model = DomainClassifier(config)
dc_path = config.MODELS_DIR / "domain_classifier" / "model.pt"
if dc_path.exists():
    dc_model.load_state_dict(torch.load(dc_path, map_location='cpu'))
    print("Domain Classifier: Loaded successfully, parameters:", sum(p.numel() for p in dc_model.parameters()))
else:
    print("Domain Classifier model not found at", dc_path)
# Verify Eligibility Predictor
ep_model = EligibilityPredictor(config)
ep_path = config.MODELS_DIR / "eligibility_predictor" / "model.pt"
if ep_path.exists():
    ep_model.load_state_dict(torch.load(ep_path, map_location='cpu'))
    print("Eligibility Predictor: Loaded successfully, parameters:", sum(p.numel() for p in ep_model.parameters()))
else:
    print("Eligibility Predictor model not found at", ep_path)