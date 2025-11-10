"""
HybEx Advanced Hybrid Prediction System
Intelligently combines Prolog, GNN, and BERT with learned confidence calibration

[FIXED - 11/09/2025]
- Rewrote __init__ for robust, CPU-safe model loading to fix "model not loaded" errors.
- Fixed TypeError in _predict_with_bert_safe and predict_with_domain_classifier_safe
  by calling models with positional arguments (e.g., model(ids, mask)) instead of
  keyword arguments (e.g., model(**inputs)).
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
# Import all required models
from .neural_models import (
    EligibilityPredictor, DomainClassifier, EnhancedLegalBERT
)
from .config import HybExConfig

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

class ConfidenceCalibrator:
    """
    Calibrates confidence scores from both BERT and Prolog.
    Uses Platt scaling / isotonic regression for better probability estimates.
    """
    
    def __init__(self):
        self.bert_calibrator = None
        self.prolog_calibrator = None
        self.gnn_calibrator = None
        self.domain_calibrator = None
        self.enhanced_calibrator = None
        self.calibrated = False
        
    def fit(self, bert_scores, prolog_scores, gnn_scores, true_labels, domain_scores=None, enhanced_scores=None):
        """
        Fit calibrators on validation set.
        
        Args:
            bert_scores: List of (confidence, prediction) tuples from BERT
            prolog_scores: List of (confidence, prediction) tuples from Prolog
            gnn_scores: List of (confidence, prediction) tuples from GNN
            domain_scores: List of (confidence, prediction) tuples from DomainClassifier
            enhanced_scores: List of (confidence, prediction) tuples from EnhancedBERT
            true_labels: Ground truth labels
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Prepare data
            bert_confidences = np.array([s[0] for s in bert_scores if s[0] is not None])
            prolog_confidences = np.array([s[0] for s in prolog_scores if s[0] is not None])
            gnn_confidences = np.array([s[0] for s in gnn_scores if s[0] is not None])
            
            labels = np.array(true_labels)
            
            # Fit isotonic regression (non-parametric, monotonic)
            if len(bert_confidences) > 10:
                self.bert_calibrator = IsotonicRegression(out_of_bounds='clip')
                self.bert_calibrator.fit(bert_confidences, labels[:len(bert_confidences)])
            
            if len(prolog_confidences) > 10:
                self.prolog_calibrator = IsotonicRegression(out_of_bounds='clip')
                self.prolog_calibrator.fit(prolog_confidences, labels[:len(prolog_confidences)])
            
            if len(gnn_confidences) > 10:
                self.gnn_calibrator = IsotonicRegression(out_of_bounds='clip')
                self.gnn_calibrator.fit(gnn_confidences, labels[:len(gnn_confidences)])
                
            if domain_scores:
                domain_confidences = np.array([s[0] for s in domain_scores if s is not None and s[0] is not None])
                if len(domain_confidences) > 10:
                    self.domain_calibrator = IsotonicRegression(out_of_bounds='clip')
                    self.domain_calibrator.fit(domain_confidences, labels[:len(domain_confidences)])

            if enhanced_scores:
                enhanced_confidences = np.array([s[0] for s in enhanced_scores if s is not None and s[0] is not None])
                if len(enhanced_confidences) > 10:
                    self.enhanced_calibrator = IsotonicRegression(out_of_bounds='clip')
                    self.enhanced_calibrator.fit(enhanced_confidences, labels[:len(enhanced_confidences)])
            
            self.calibrated = True
            logger.info("✅ Confidence calibrators fitted")
            
        except ImportError:
            logger.warning("⚠️  scikit-learn not available, skipping calibration")
        except Exception as e:
            logger.warning(f"⚠️  Calibration fitting failed: {e}")
        
    def calibrate_bert(self, confidence):
        """Calibrate BERT confidence score."""
        if not self.calibrated or self.bert_calibrator is None:
            return confidence
        try:
            return float(self.bert_calibrator.predict([confidence])[0])
        except:
            return confidence
    
    def calibrate_prolog(self, confidence):
        """Calibrate Prolog confidence score."""
        if not self.calibrated or self.prolog_calibrator is None:
            return confidence
        try:
            return float(self.prolog_calibrator.predict([confidence])[0])
        except:
            return confidence
    
    def calibrate_gnn(self, confidence):
        """Calibrate GNN confidence score."""
        if not self.calibrated or self.gnn_calibrator is None:
            return confidence
        try:
            return float(self.gnn_calibrator.predict([confidence])[0])
        except:
            return confidence

    def calibrate_domain(self, confidence):
        """Calibrate Domain Classifier confidence score."""
        if not self.calibrated or self.domain_calibrator is None:
            return confidence
        try:
            return float(self.domain_calibrator.predict([confidence])[0])
        except:
            return confidence

    def calibrate_enhanced(self, confidence):
        """Calibrate EnhancedBERT confidence score."""
        if not self.calibrated or self.enhanced_calibrator is None:
            return confidence
        try:
            return float(self.enhanced_calibrator.predict([confidence])[0])
        except:
            return confidence


# ============================================================================
# PREDICTION RESULTS
# ============================================================================

@dataclass
class HybridPrediction:
    """Enhanced container for hybrid prediction results"""
    case_id: str
    eligible: bool
    confidence: float
    method_used: str
    prolog_result: Optional[Dict] = None
    gnn_result: Optional[Dict] = None
    bert_result: Optional[Dict] = None
    reasoning: str = ""
    uncertainty: float = 0.0
    requires_review: bool = False
    calibrated_confidences: Dict[str, float] = field(default_factory=dict)
    decision_rationale: str = ""
    prolog_debug_facts: Optional[List[str]] = None


# ============================================================================
# ROBUST MODEL LOADING HELPERS
# ============================================================================

def _torch_load_cpu_safe(path):
    """
    Always load to CPU if CUDA is not available or if device is 'cpu'.
    Handles checkpoints that include CUDA device metadata.
    """
    try:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_loc = torch.device(device_str) if device_str == 'cuda' else 'cpu'
        ckpt = torch.load(path, map_location=map_loc)
        return ckpt
    except Exception as e:
        try:
            ckpt = torch.load(path, map_location='cpu')
            return ckpt
        except Exception as e2:
            logger.error(f"Failed to load checkpoint {path}: {e}; fallback error: {e2}")
            raise

def _ensure_model_file(path: Path, name: str):
    """Check if model file exists and return friendly boolean."""
    if not path.exists():
        logger.warning(f"Model file for {name} not found at: {path}")
        return False
    return True

def load_model_safely(model, path, device):
    """
    Safe model loader with map_location, strict=False fallback, and detailed logging.
    """
    try:
        ckpt = _torch_load_cpu_safe(path)
        
        if isinstance(ckpt, dict):
            state = ckpt.get('model_state_dict', ckpt)
        else:
            state = ckpt
        
        try:
            model.load_state_dict(state, strict=True)
            logger.info(f"Loaded model from {path} (strict=True)")
        except RuntimeError as e:
            logger.warning(f"Strict load failed for {path}: {e}. Trying non-strict load.")
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded model from {path} (strict=False)")
            
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Model load failed for {path}: {e}")
        return None

# ============================================================================
# MAIN PREDICTOR CLASS
# ============================================================================

class IntelligentHybridPredictor:
    """
    Advanced hybrid predictor with:
    1. Learned ensemble weights
    2. Confidence calibration
    3. Dynamic method selection
    4. Uncertainty quantification
    """
    
    def __init__(self, prolog_engine, gnn_model, bert_model, config: HybExConfig, force_cpu: bool = False):
        """
        [FIXED 11/09/2025]
        Rewritten to robustly load all neural models, as streamlit_app
        initializes this class directly without passing pre-loaded models.
        """
        self.prolog_engine = prolog_engine
        self.prolog = prolog_engine
        self.gnn = gnn_model
        self.bert = bert_model  # This is usually None when called from streamlit
        self.config = config
        
        cuda_available = torch.cuda.is_available() and (not force_cpu)
        self.device = torch.device('cuda' if cuda_available else 'cpu')
        
        if force_cpu and torch.cuda.is_available():
            logger.info("Force CPU mode enabled (CUDA available but not using it)")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['base_model'])
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("HybridPredictor loaded its own tokenizer.")
        except Exception as e:
            logger.error(f"HybridPredictor failed to load tokenizer: {e}")
            self.tokenizer = None

        # --- [START] ROBUST MODEL LOADING ---
        # This logic runs because streamlit_app passes bert_model=None
        
        self.eligibility_model = None
        self.domain_classifier = None
        self.enhanced_bert = None

        logger.info("IntelligentHybridPredictor loading models manually...")

        # Load Eligibility Predictor
        try:
            elig_path = self.config.MODELS_DIR / 'eligibility_predictor' / 'model.pt'
            if _ensure_model_file(elig_path, "EligibilityPredictor"):
                model_instance = EligibilityPredictor(self.config)
                self.eligibility_model = load_model_safely(model_instance, elig_path, self.device)
                if self.eligibility_model:
                     logger.info("✅ HybridPredictor loaded EligibilityPredictor.")
            else:
                logger.error("EligibilityPredictor model.pt not found.")
        except Exception as e:
            logger.error(f"Failed to load EligibilityPredictor: {e}", exc_info=True)

        # Load Domain Classifier
        try:
            domain_path = self.config.MODELS_DIR / 'domain_classifier' / 'model.pt'
            if _ensure_model_file(domain_path, "DomainClassifier"):
                model_instance = DomainClassifier(self.config)
                self.domain_classifier = load_model_safely(model_instance, domain_path, self.device)
                if self.domain_classifier:
                    logger.info("✅ HybridPredictor loaded DomainClassifier.")
            else:
                logger.error("DomainClassifier model.pt not found.")
        except Exception as e:
            logger.error(f"Failed to load DomainClassifier: {e}", exc_info=True)

        # Load EnhancedLegalBERT
        try:
            enhanced_path = self.config.ENHANCED_BERT_MODEL_PATH
            if _ensure_model_file(enhanced_path, "EnhancedLegalBERT"):
                model_instance = EnhancedLegalBERT(self.config)
                # Note: EnhancedBERT path points to the .pt file directly
                self.enhanced_bert = load_model_safely(model_instance, enhanced_path, self.device)
                if self.enhanced_bert:
                    logger.info("✅ HybridPredictor loaded EnhancedLegalBERT.")
            else:
                logger.error("EnhancedLegalBERT .pt file not found.")
        except Exception as e:
            logger.error(f"Failed to load EnhancedLegalBERT: {e}", exc_info=True)
        
        # --- [END] ROBUST MODEL LOADING ---
        
        # Confidence calibrator
        self.calibrator = ConfidenceCalibrator()
        
        # Learned ensemble parameters - Fix 4: Prolog should dominate (legal rules are deterministic)
        self.ensemble_params = {
            'prolog_threshold': 0.75,
            'gnn_threshold': 0.70,
            'bert_threshold': 0.75,
            'prolog_weight': 0.50,  # INCREASED from 0.15 - Legal rules should dominate
            'bert_weight': 0.25,     # DECREASED from 0.40
            'gnn_weight': 0.15,      # DECREASED from 0.35
            'domain_weight': 0.05,
            'enhanced_bert_weight': 0.05,
            'uncertainty_threshold': 0.60,
            'conflict_penalty': 0.10  # INCREASED from 0.05 - Penalize disagreements more
        }
        
        self.method_history = {
            'bert': {'correct': 0, 'total': 0},
            'prolog': {'correct': 0, 'total': 0},
            'gnn': {'correct': 0, 'total': 0},
            'ensemble': {'correct': 0, 'total': 0}
        }
        
        logger.info("✅ Intelligent Hybrid Predictor initialized with confidence calibration")
    
    def predict(self, case_data: Dict[str, Any]) -> HybridPrediction:
        """
        Make prediction with full reasoning trace and uncertainty quantification.
        
        Returns:
            HybridPrediction with confidence calibration, uncertainty, and review flags
        """
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            entities = case_data.get('extracted_entities', {})
            
            # 1. Get Domain Prediction FIRST
            domain_res = self.predict_with_domain_classifier_safe(case_data)
            predicted_domain_index = 0 # Default to 'legal_aid' (index 0)
            
            if domain_res and domain_res.bert_result and 'domain_probs' in domain_res.bert_result:
                probs = domain_res.bert_result['domain_probs']
                if isinstance(probs, list):
                    probs = np.array(probs)
                if probs.ndim > 1:
                    probs = probs.flatten() # Ensure 1D array
                if probs.size > 0:
                    predicted_domain_index = np.argmax(probs)
            
            # 2. Get predictions from other systems
            bert_result = self._predict_with_bert_safe(case_data)
            
            # C: Build Prolog facts and ensure proper dict conversion
            prolog_debug_facts = None
            prolog_result = None
            prolog_result_dict = None
            try:
                # Ensure the Prolog engine receives structured entities
                if self.prolog_engine and hasattr(self.prolog_engine, 'build_facts_from_entities'):
                    prolog_debug_facts = self.prolog_engine.build_facts_from_entities(entities)
                    logger.debug(f"Prolog facts for {case_id}: {prolog_debug_facts}")

                # Call prolog predictor passing extracted entities (so rules are applied per-query)
                prolog_result = self._predict_with_prolog_safe({**case_data, 'extracted_entities': entities})

                # C: If prolog_result is a HybridPrediction, extract its prolog_result field
                if prolog_result and hasattr(prolog_result, 'prolog_result'):
                    prolog_obj = prolog_result.prolog_result
                    # Convert to dict for JSON safety
                    if hasattr(prolog_obj, 'to_dict'):
                        prolog_result_dict = prolog_obj.to_dict()
                    elif hasattr(prolog_obj, '__dict__'):
                        from dataclasses import is_dataclass, asdict
                        if is_dataclass(prolog_obj):
                            prolog_result_dict = asdict(prolog_obj)
                        else:
                            prolog_result_dict = prolog_obj.__dict__
                    else:
                        prolog_result_dict = prolog_obj
            except Exception as e:
                logger.warning(f"Prolog prediction failed for {case_id}: {e}")
                prolog_result = None
                prolog_result_dict = None
            
            gnn_result = self._predict_with_gnn_safe(case_data)
            
            # 3. Get EnhancedBERT prediction, PASSING IN THE DOMAIN
            enhanced_res = self.predict_with_enhanced_bert_safe(case_data, int(predicted_domain_index))
            
            # If all failed, use fallback
            if not any([bert_result, prolog_result, gnn_result, domain_res, enhanced_res]):
                return self._fallback_prediction(case_data)
            
            # Extract raw confidences
            bert_conf = bert_result.confidence if bert_result else 0.5
            prolog_conf = prolog_result.confidence if prolog_result else 0.5
            gnn_conf = gnn_result.confidence if gnn_result else 0.5
            domain_conf = domain_res.confidence if domain_res else 0.5
            enhanced_conf = enhanced_res.confidence if enhanced_res else 0.5
            
            # Calibrate confidences
            bert_conf_cal = self.calibrator.calibrate_bert(bert_conf)
            prolog_conf_cal = self.calibrator.calibrate_prolog(prolog_conf)
            gnn_conf_cal = self.calibrator.calibrate_gnn(gnn_conf)
            domain_conf_cal = self.calibrator.calibrate_domain(domain_conf)
            enhanced_conf_cal = self.calibrator.calibrate_enhanced(enhanced_conf)
            
            # Make hybrid decision with intelligent routing
            decision = self._make_hybrid_decision(
                bert_result, prolog_result, gnn_result,
                bert_conf_cal, prolog_conf_cal, gnn_conf_cal,
                entities,
                domain_res,      # Pass the result
                enhanced_res,    # Pass the result
                domain_conf_cal, # Pass calibrated confidence
                enhanced_conf_cal # Pass calibrated confidence
            )
            
            # Calculate uncertainty (epistemic + aleatoric)
            uncertainty = self._calculate_uncertainty(
                bert_result, prolog_result, gnn_result, entities
            )
            
            # Determine if human review needed
            requires_review = self._needs_review(
                decision['confidence'], uncertainty, entities
            )
            
            # D: Helper to convert any object to JSON-safe dict
            def _obj_to_safe(o):
                if o is None:
                    return None
                # Already a simple type
                if isinstance(o, (str, int, float, bool, list, dict)):
                    return o
                # Has to_dict method
                if hasattr(o, 'to_dict'):
                    return o.to_dict()
                # Is a dataclass
                from dataclasses import is_dataclass, asdict
                if is_dataclass(o):
                    return asdict(o)
                # Has __dict__
                if hasattr(o, '__dict__'):
                    return o.__dict__
                # Fallback: convert to string
                return str(o)
            
            return HybridPrediction(
                case_id=case_id,
                eligible=decision['eligible'],
                confidence=decision['confidence'],
                method_used=decision['method'],
                bert_result=_obj_to_safe(bert_result.bert_result if bert_result else None),
                prolog_result=_obj_to_safe(prolog_result_dict if prolog_result_dict else None),
                gnn_result=_obj_to_safe(gnn_result.gnn_result if gnn_result else None),
                reasoning=decision['reasoning'],
                uncertainty=uncertainty,
                requires_review=requires_review,
                calibrated_confidences={
                    'bert': bert_conf_cal,
                    'prolog': prolog_conf_cal,
                    'gnn': gnn_conf_cal
                },
                decision_rationale=decision.get('rationale', ''),
                prolog_debug_facts=prolog_debug_facts
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {case_data.get('sample_id')}: {e}")
            return self._fallback_prediction(case_data)
    
    def predict_from_text(self, text: str, ask_for_clarification: bool = False):
        """
        High-level convenience method that accepts raw user text and runs:
         - entity extraction (via DataPreprocessor)
         - per-component predictions (BERT/GNN/Prolog)
         - calibration & ensemble decision (existing predict)
         - returns HybridPrediction dataclass/dict
        If required entities are missing and ask_for_clarification==True, returns
        a structured object asking the frontend for clarification.
        """
        # 1) extract entities using repo preprocessor if present
        try:
            # prefer package-relative import
            from .data_processor import DataPreprocessor
            dp = DataPreprocessor(self.config)
            if hasattr(dp, 'extract_entities'):
                entities = dp.extract_entities(text)
            elif hasattr(dp, 'preprocess_text'):
                pre = dp.preprocess_text(text)
                entities = pre.get('extracted_entities', {})
            else:
                # last resort
                pre = dp.preprocess_case({'query': text})
                entities = pre.get('extracted_entities', {})
        except Exception:
            # fallback minimal extractor (replicate same heuristics)
            entities = {}
            import re
            m = re.search(r'([\d,]+)\s*(?:rupees|rs|₹)?', text, flags=re.I)
            if m:
                try:
                    entities['income'] = int(m.group(1).replace(',', ''))
                except:
                    pass
            m = re.search(r'(\d{2})\s*(?:years?)', text, flags=re.I)
            if m:
                try:
                    entities['age'] = int(m.group(1))
                except:
                    pass
            if re.search(r'\b(woman|female|she|her)\b', text, flags=re.I):
                entities['gender'] = 'female'
            elif re.search(r'\b(man|male|he|him)\b', text, flags=re.I):
                entities['gender'] = 'male'
            entities['mentions_eviction'] = bool(re.search(r'\bevict|eviction\b', text, flags=re.I))

        # 2) build minimal case dict the predictor.predict expects
        case = {
            'sample_id': 'TEXT_' + str(abs(hash(text)) % (10**8)),
            'query': text,
            'extracted_entities': entities
        }

        # 3) check for required fields for deterministic rules (optional)
        # e.g., if income is missing but might be needed and ask_for_clarification True -> return clarification request
        if ask_for_clarification:
            required_for_rules = ['income', 'gender', 'age']
            missing = [k for k in required_for_rules if k not in entities]
            # only ask if all are missing (reduce prompts)
            if len(missing) >= 2:
                return {
                    'type': 'clarify',
                    'missing': missing,
                    'message': f'Please clarify the following details for a reliable eligibility check: {missing}'
                }

        # 4) call existing predictor entrypoint
        return self.predict(case)

    def _make_hybrid_decision(self, bert_res, prolog_res, gnn_res,
                             bert_conf_cal, prolog_conf_cal, gnn_conf_cal, entities,
                             domain_res, enhanced_res, 
                             domain_conf_cal, enhanced_conf_cal):
        """
        UPDATED: Intelligent hybrid decision with ALL 5 models
        """
        
        # Strategy 1: Check for HIGH-CONFIDENCE AGREEMENT
        predictions = []
        if bert_res:
            predictions.append(('bert', bert_res.eligible, bert_conf_cal))
        if prolog_res:
            predictions.append(('prolog', prolog_res.eligible, prolog_conf_cal))
        if gnn_res:
            predictions.append(('gnn', gnn_res.eligible, gnn_conf_cal))
        
        if domain_res:
            predictions.append(('domain_classifier', domain_res.eligible, domain_conf_cal))
        if enhanced_res:
            predictions.append(('enhanced_bert', enhanced_res.eligible, enhanced_conf_cal))

        if len(predictions) >= 3:
            eligibilities = [p[1] for p in predictions]
            confidences = [p[2] for p in predictions]
            
            if len(set(eligibilities)) == 1 and np.mean(confidences) > 0.85:
                boosted_conf = min(0.98, np.mean(confidences) * 1.12)
                return {
                    'eligible': eligibilities[0],
                    'confidence': boosted_conf,
                    'method': 'strong_consensus',
                    'reasoning': f"All {len(predictions)} models strongly agree (avg conf={np.mean(confidences):.2f})",
                    'rationale': "Unanimous decision with high confidence"
                }
            
            if prolog_res and prolog_conf_cal > 0.80:
                neural_agree = sum(1 for p in predictions if p[0] != 'prolog' and p[1] == prolog_res.eligible)
                if neural_agree >= 2:
                    boosted_conf = min(0.95, (prolog_conf_cal + np.mean([p[2] for p in predictions if p[0] != 'prolog'])) / 2)
                    return {
                        'eligible': prolog_res.eligible,
                        'confidence': boosted_conf,
                        'method': 'prolog_neural_consensus',
                        'reasoning': f"Prolog ({prolog_conf_cal:.2f}) + {neural_agree} neural models agree",
                        'rationale': "Legal rules confirmed by neural analysis"
                    }
        
        # Strategy 2: Check for conflict
        if len(predictions) >= 2:
            eligibilities = [p[1] for p in predictions]
            if len(set(eligibilities)) > 1:  # Disagreement
                votes_eligible = sum(1 for e in eligibilities if e)
                votes_not = len(eligibilities) - votes_eligible
                conf_eligible = np.mean([p[2] for p in predictions if p[1] == True]) if votes_eligible > 0 else 0.0
                conf_not = np.mean([p[2] for p in predictions if p[1] == False]) if votes_not > 0 else 0.0
                
                if prolog_res:
                    if prolog_res.eligible:
                        votes_eligible += 1
                    else:
                        votes_not += 1
                
                if votes_eligible > votes_not:
                    final_conf = conf_eligible * 0.95
                    return {
                        'eligible': True,
                        'confidence': final_conf,
                        'method': 'majority_vote_eligible',
                        'reasoning': f"Majority: {votes_eligible:.0f} eligible vs {votes_not:.0f} not",
                        'rationale': "Democratic voting"
                    }
                else:
                    final_conf = conf_not * 0.95
                    return {
                        'eligible': False,
                        'confidence': final_conf,
                        'method': 'majority_vote_not_eligible',
                        'reasoning': f"Majority: {votes_not:.0f} not eligible vs {votes_eligible:.0f}",
                        'rationale': "Democratic voting"
                    }
        
        # Strategy 3: Systems agree
        if len(predictions) >= 2:
            eligibilities = [p[1] for p in predictions] # Re-check (might be redundant but safe)
            if len(set(eligibilities)) == 1:
                avg_conf = np.mean([p[2] for p in predictions])
                boosted_conf = min(0.98, avg_conf * 1.15)
                return {
                    'eligible': eligibilities[0],
                    'confidence': boosted_conf,
                    'method': 'ensemble',
                    'reasoning': f"All {len(predictions)} systems agree ({eligibilities[0]}): avg_conf={avg_conf:.2f}",
                    'rationale': "Strong consensus"
                }
        
        # Strategy 4: Default weighted ensemble
        return self._weighted_ensemble(
            bert_res, prolog_res, gnn_res, domain_res, enhanced_res,
            bert_conf_cal, prolog_conf_cal, gnn_conf_cal,
            domain_conf_cal, enhanced_conf_cal
        )
    
    def _weighted_ensemble(self, bert_res, prolog_res, gnn_res, domain_res, enhanced_res,
                          bert_conf_cal, prolog_conf_cal, gnn_conf_cal,
                          domain_conf_cal, enhanced_conf_cal,
                          conflict_penalty=0.0):
        """
        UPDATED: Weighted ensemble with ALL 5 models and adaptive weights.
        """
        
        eligible_score = 0.0
        total_weight = 0.0
        components = []

        if bert_res:
            base_weight = self.ensemble_params.get('bert_weight', 0.40)
            conf_multiplier = 0.5 + bert_conf_cal
            conf_weight = base_weight * conf_multiplier
            score = bert_conf_cal if bert_res.eligible else (1 - bert_conf_cal)
            eligible_score += conf_weight * score
            total_weight += conf_weight
            components.append(f"BERT:{score:.2f}×{conf_weight:.2f}")

        if prolog_res:
            base_weight = self.ensemble_params.get('prolog_weight', 0.15)
            conf_multiplier = 0.5 + prolog_conf_cal
            conf_weight = base_weight * conf_multiplier
            score = prolog_conf_cal if prolog_res.eligible else (1 - prolog_conf_cal)
            eligible_score += conf_weight * score
            total_weight += conf_weight
            components.append(f"Prolog:{score:.2f}×{conf_weight:.2f}")

        if gnn_res:
            base_weight = self.ensemble_params.get('gnn_weight', 0.35)
            conf_multiplier = 0.5 + gnn_conf_cal
            conf_weight = base_weight * conf_multiplier
            score = gnn_conf_cal if gnn_res.eligible else (1 - gnn_conf_cal)
            eligible_score += conf_weight * score
            total_weight += conf_weight
            components.append(f"GNN:{score:.2f}×{conf_weight:.2f}")

        if domain_res:
            base_weight = self.ensemble_params.get('domain_weight', 0.05)
            conf_multiplier = 0.5 + domain_conf_cal
            conf_weight = base_weight * conf_multiplier
            score = domain_conf_cal if domain_res.eligible else (1 - domain_conf_cal)
            eligible_score += conf_weight * score
            total_weight += conf_weight
            components.append(f"Domain:{score:.2f}×{conf_weight:.2f}")

        if enhanced_res:
            base_weight = self.ensemble_params.get('enhanced_bert_weight', 0.05)
            conf_multiplier = 0.5 + enhanced_conf_cal
            conf_weight = base_weight * conf_multiplier
            score = enhanced_conf_cal if enhanced_res.eligible else (1 - enhanced_conf_cal)
            eligible_score += conf_weight * score
            total_weight += conf_weight
            components.append(f"Enhanced:{score:.2f}×{conf_weight:.2f}")

        if total_weight > 0:
            eligible_score /= total_weight

        final_confidence = eligible_score * (1 - conflict_penalty)
        eligible = eligible_score > 0.5

        return {
            'eligible': eligible,
            'confidence': final_confidence,
            'method': 'weighted_ensemble',
            'reasoning': f"Weighted: {', '.join(components)}",
            'rationale': f"Adaptive weighting {'(penalty applied)' if conflict_penalty > 0 else ''}"
        }
    
    def _calculate_uncertainty(self, bert_res, prolog_res, gnn_res, entities):
        """
        Calculate prediction uncertainty.
        """
        
        # Component 1: Disagreement between methods
        predictions = []
        if bert_res:
            predictions.append(bert_res.eligible)
        if prolog_res:
            predictions.append(prolog_res.eligible)
        if gnn_res:
            predictions.append(gnn_res.eligible)
        
        if len(predictions) >= 2:
            disagreement = 1.0 - (predictions.count(predictions[0]) / len(predictions))
        else:
            disagreement = 0.5  # Unknown if only one model
        
        # Component 2: Low confidence from any method
        confidences = []
        if bert_res:
            confidences.append(bert_res.confidence)
        if prolog_res:
            confidences.append(prolog_res.confidence)
        if gnn_res:
            confidences.append(gnn_res.confidence)
        
        low_confidence = 1.0 - max(confidences) if confidences else 0.5
        
        # Component 3: Near income threshold (for legal aid)
        near_boundary = self._check_near_boundary(entities)
        
        # Combine uncertainties (weighted average)
        total_uncertainty = (
            0.4 * disagreement +
            0.4 * low_confidence +
            0.2 * near_boundary
        )
        
        return min(1.0, total_uncertainty)
    
    def _check_near_boundary(self, entities):
        """Check if case is near decision boundary."""
        if 'income' not in entities and 'annual_income' not in entities:
            return 0.0
        
        income = entities.get('income', 0) or entities.get('annual_income', 0)
        if income == 0:
            return 0.0
        
        # Convert monthly to annual if needed
        annual_income = income * 12 if income < 100000 else income
        
        category = entities.get('social_category', 'general').lower()
        
        # Income thresholds per LSA Act 1987
        thresholds = {
            'general': 300000,
            'obc': 600000,
            'sc': 800000,
            'st': 800000,
            'ews': 800000,
            'bpl': float('inf')
        }
        
        threshold = thresholds.get(category, 300000)
        
        if threshold == float('inf'):
            return 0.0
        
        # Calculate distance from threshold (normalized)
        distance = abs(annual_income - threshold) / threshold
        
        # Near if within 15% of threshold
        if distance < 0.15:
            return 1.0 - (distance / 0.15)  # 1.0 at threshold, 0.0 at 15% away
        
        return 0.0
    
    def _needs_review(self, confidence, uncertainty, entities):
        """Determine if case needs human review."""
        
        # Review if low confidence
        if confidence < self.ensemble_params['uncertainty_threshold']:
            return True
        
        # Review if high uncertainty
        if uncertainty > 0.65:
            return True
        
        # Review if high-income near threshold
        if self._check_near_boundary(entities) > 0.7:
            return True
        
        # Review if vulnerable person with borderline income
        has_vulnerable = any([
            entities.get('is_disabled'),
            entities.get('is_senior_citizen'),
            entities.get('is_widow'),
            entities.get('is_single_parent'),
            entities.get('is_transgender'),
            entities.get('gender') == 'female'
        ])
        
        if has_vulnerable:
            income = entities.get('income', 0) or entities.get('annual_income', 0)
            if income > 20000:  # Above typical thresholds
                return True
        
        return False
    
    def _classify_case_type(self, entities: Dict) -> str:
        """Classify case type"""
        category = entities.get('social_category', 'general')
        has_income = 'income' in entities or 'annual_income' in entities
        
        has_vulnerable = any([
            entities.get('is_disabled'),
            entities.get('is_senior_citizen'),
            entities.get('is_widow'),
            entities.get('is_single_parent'),
            entities.get('is_transgender'),
            entities.get('gender') == 'female'
        ])
        
        if category in ['sc', 'st', 'obc', 'bpl'] or has_vulnerable:
            return 'deterministic'
        elif has_income or len(entities) > 2:
            return 'structured'
        else:
            return 'text_only'
    
    def _predict_with_prolog_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """FIX: Use correct PrologEngine method"""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            
            # FIX: Use batch_legal_analysis with single case
            results = self.prolog.batch_legal_analysis([case_data])
            
            if results and len(results) > 0:
                prolog_reasoning = results[0]
                
                # D: Convert LegalReasoning to dict for JSON serialization
                from dataclasses import asdict, is_dataclass
                if is_dataclass(prolog_reasoning):
                    prolog_dict = asdict(prolog_reasoning)
                elif hasattr(prolog_reasoning, 'to_dict'):
                    prolog_dict = prolog_reasoning.to_dict()
                elif hasattr(prolog_reasoning, '__dict__'):
                    prolog_dict = prolog_reasoning.__dict__
                else:
                    prolog_dict = {'reasoning': str(prolog_reasoning)}
                
                return HybridPrediction(
                    case_id=case_id,
                    eligible=prolog_reasoning.eligible,
                    confidence=prolog_reasoning.confidence,
                    method_used='prolog',
                    prolog_result=prolog_dict,  # D: Use dict instead of object
                    reasoning=prolog_reasoning.primary_reason
                )
            return None
            
        except Exception as e:
            logger.warning(f"Prolog failed for {case_data.get('sample_id')}: {str(e)[:100]}")
            return None
    
    def _predict_with_gnn_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """FIX: Use the GNN's own prediction method for correct inference."""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            entities = case_data.get('extracted_entities', {})
            
            # self.gnn IS the knowledge graph engine
            if self.gnn is None:
                return None
            
            # Ensure the GNN model is loaded within the engine
            if self.gnn.model is None:
                self.gnn.load_model(str(self.config.GNN_MODEL_PATH))

            # Use the engine's built-in prediction method
            # This correctly uses the GAT.forward(), global_mean_pool, and readout layer
            prediction, probabilities = self.gnn.predict_eligibility(
                entities, 
                return_probabilities=True
            )
            
            confidence = probabilities[prediction].item()
            
            return HybridPrediction(
                case_id=case_id,
                eligible=bool(prediction),
                confidence=confidence,
                method_used='gnn',
                gnn_result={'probs': probabilities.cpu().numpy().tolist()},
                reasoning=f"GNN: {len(entities)} features"
            )
            
        except Exception as e:
            logger.warning(f"GNN failed for {case_data.get('sample_id')}: {str(e)[:100]}")
            return None
    
    def _predict_with_bert_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """
        [FIXED 11/09/2025]
        - Use self.tokenizer and self.eligibility_model
        - Call model with positional arguments: model(ids, mask)
        """
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            
            if not query or self.tokenizer is None:
                logger.warning(f"BERT failed for {case_id}: Missing query or tokenizer.")
                return None
            if self.eligibility_model is None:
                logger.warning(f"BERT failed for {case_id}: Eligibility model not loaded.")
                return None
            
            inputs = self.tokenizer(
                query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                self.eligibility_model.eval()
                
                # --- [START] THE FIX ---
                # Call with positional arguments, not kwargs
                outputs = self.eligibility_model(input_ids, attention_mask, return_dict=True)
                # --- [END] THE FIX ---
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get(list(outputs.keys())[0]))
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits

                if logits.dim() == 1:
                    probs = torch.sigmoid(logits)
                    pred_class = (probs > 0.5).long().item()
                    confidence = probs.item() if pred_class == 1 else (1 - probs.item())
                elif logits.dim() == 2:
                    if logits.size(1) == 1:
                        logits = logits.squeeze(1)
                        probs = torch.sigmoid(logits)
                        pred_class = (probs > 0.5).long().item()
                        confidence = probs.item() if pred_class == 1 else (1 - probs.item())
                    else:
                        probs = torch.softmax(logits, dim=1)
                        pred_class = torch.argmax(probs, dim=1).item()
                        confidence = probs[0, pred_class].item()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                return HybridPrediction(
                    case_id=case_id,
                    eligible=bool(pred_class),
                    confidence=confidence,
                    method_used='bert',
                    bert_result={'probs': probs.cpu().numpy().tolist() if hasattr(probs, 'cpu') else probs},
                    reasoning=f"BERT: {len(query)} chars"
                )
            
        except Exception as e:
            logger.error(f"BERT prediction failed for {case_data.get('sample_id')}: {e}", exc_info=True)
            return None
    
    def predict_with_domain_classifier_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """
        [FIXED 11/09/2025]
        - Use self.tokenizer and self.domain_classifier
        - Call model with positional arguments: model(ids, mask)
        """
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            
            if not query or self.tokenizer is None:
                logger.warning(f"Domain classifier failed for {case_id}: Missing query or tokenizer.")
                return None
            if self.domain_classifier is None:
                logger.warning(f"Domain classifier failed for {case_id}: Model not loaded.")
                return None
            
            inputs = self.tokenizer(
                query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                self.domain_classifier.eval()
                
                # --- [START] THE FIX ---
                # Call with positional arguments, not kwargs
                logits = self.domain_classifier(
                    input_ids, 
                    attention_mask,
                    return_dict=True
                )['logits']
                # --- [END] THE FIX ---
                
                probs = torch.sigmoid(logits)
                legal_aid_idx = 0
                confidence = float(probs[0, legal_aid_idx])
                eligible = confidence >= 0.5
            
            return HybridPrediction(
                case_id=case_id,
                eligible=eligible,
                confidence=confidence,
                method_used='domain_classifier',
                bert_result={'domain_probs': probs.cpu().numpy().tolist()},
                reasoning=f"Domain classifier confidence: {confidence:.3f}"
            )
        except Exception as e:
            logger.error(f"Domain classifier failed: {e}", exc_info=True)
            return None

    def predict_with_enhanced_bert_safe(self, case_data: Dict, domain_index: int = 0) -> Optional[HybridPrediction]:
        """
        [FIXED 11/09/2025]
        - Use self.tokenizer and self.enhanced_bert
        - Call signature (kwargs) is correct for this model, but check for None.
        """
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            
            if not query or self.tokenizer is None:
                logger.warning(f"EnhancedBERT failed for {case_id}: Missing query or tokenizer.")
                return None
            if self.enhanced_bert is None:
                logger.warning(f"EnhancedBERT failed for {case_id}: Model not loaded.")
                return None
            
            inputs = self.tokenizer(
                query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            domain_tensor = torch.tensor([domain_index], dtype=torch.long).to(self.device)

            with torch.no_grad():
                self.enhanced_bert.eval()
                
                # This model IS designed to accept kwargs, so this call is correct.
                outputs = self.enhanced_bert(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'],
                    domains=domain_tensor,
                    return_dict=True,
                    return_confidence=True
                )
                
                eligibility_logits = outputs['eligibility_logits']
                domain_logits = outputs.get('domain_logits', None)
                
                if eligibility_logits.dim() == 1:
                    prob = torch.sigmoid(eligibility_logits).item()
                else:
                    probs = torch.softmax(eligibility_logits, dim=1)
                    prob = probs[0, 1].item()
                
                eligible = prob >= 0.5
            
            return HybridPrediction(
                case_id=case_id,
                eligible=eligible,
                confidence=prob,
                method_used='enhanced_bert',
                bert_result={
                    'eligibility_prob': prob,
                    'domain_logits': domain_logits.cpu().numpy().tolist() if domain_logits is not None else None
                },
                reasoning=f"EnhancedBERT (multi-task) confidence: {prob:.3f}"
            )
        except Exception as e:
            logger.error(f"EnhancedBERT failed: {e}", exc_info=True)
            return None
    
    def _predict_with_ensemble_safe(self, case_data: Dict) -> HybridPrediction:
        """Safe ensemble - GUARANTEED to return HybridPrediction"""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            
            # Try all three
            prolog_pred = self._predict_with_prolog_safe(case_data)
            gnn_pred = self._predict_with_gnn_safe(case_data)
            bert_pred = self._predict_with_bert_safe(case_data)
            
            # If all failed, use fallback
            if not any([prolog_pred, gnn_pred, bert_pred]):
                return self._fallback_prediction(case_data)
            
            # Weighted voting
            eligible_score = 0.0
            total_weight = 0.0
            
            if prolog_pred:
                weight = self.ensemble_params.get('prolog_weight', 0.33)
                score = prolog_pred.confidence if prolog_pred.eligible else (1 - prolog_pred.confidence)
                eligible_score += weight * score
                total_weight += weight
            
            if gnn_pred:
                weight = self.ensemble_params.get('gnn_weight', 0.33)
                score = gnn_pred.confidence if gnn_pred.eligible else (1 - gnn_pred.confidence)
                eligible_score += weight * score
                total_weight += weight
            
            if bert_pred:
                weight = self.ensemble_params.get('bert_weight', 0.34)
                score = bert_pred.confidence if bert_pred.eligible else (1 - bert_pred.confidence)
                eligible_score += weight * score
                total_weight += weight
            
            if total_weight > 0:
                eligible_score /= total_weight
            
            final_eligible = eligible_score > 0.5
            final_confidence = eligible_score if final_eligible else (1 - eligible_score)
            
            parts = []
            if prolog_pred:
                parts.append(f"P={prolog_pred.eligible}/{prolog_pred.confidence:.2f}")
            if gnn_pred:
                parts.append(f"G={gnn_pred.eligible}/{gnn_pred.confidence:.2f}")
            if bert_pred:
                parts.append(f"B={bert_pred.eligible}/{bert_pred.confidence:.2f}")
            
            return HybridPrediction(
                case_id=case_id,
                eligible=final_eligible,
                confidence=final_confidence,
                method_used='ensemble',
                prolog_result=prolog_pred.prolog_result if prolog_pred else None,
                gnn_result=gnn_pred.gnn_result if gnn_pred else None,
                bert_result=bert_pred.bert_result if bert_pred else None,
                reasoning=f"Ensemble: {', '.join(parts)}"
            )
            
        except Exception as e:
            logger.error(f"Ensemble failed for {case_data.get('sample_id')}: {e}")
            return self._fallback_prediction(case_data)
    
    def _get_domain_specific_next_steps(self, eligible: bool, domain: str, case_type: str, category: str) -> List[str]:
        """
        Generate comprehensive, domain-specific next steps
        Different for eligible vs ineligible cases
        """
        
        # CRITICAL: Tax law cases are NOT covered under legal aid
        if domain == 'taxlaw':
            # Extract income if available (default to 0)
            income_lakhs = 0.0
            # This method doesn't have access to entities, so we'll show generic tax guidance
            return self._get_tax_law_next_steps(income_lakhs)
        
        if eligible:
            # ===== ELIGIBLE CASES - DETAILED ACTIONABLE STEPS =====
            
            base_steps = [
                "## 📋 Step 1: Gather Required Documents",
                "Collect the following documents before applying:",
                "- ✅ **Income Proof**: Salary slips (last 3 months) OR Income certificate from Tehsildar",
                "- ✅ **Identity Proof**: Aadhaar card, Voter ID, or Passport",
                "- ✅ **Residence Proof**: Electricity bill, rent agreement, or ration card",
            ]
            
            # Add category-specific documents
            if category in ['SC', 'ST']:
                base_steps.append("- ✅ **Caste Certificate**: Valid SC/ST certificate from competent authority")
            elif category == 'OBC':
                base_steps.append("- ✅ **OBC Certificate**: Non-creamy layer certificate (if applicable)")
            elif category == 'BPL':
                base_steps.append("- ✅ **BPL Card**: Valid Below Poverty Line card")
            
            base_steps.extend([
                "",
                "## 🏛️ Step 2: Locate Your Legal Services Authority",
                "Visit your nearest Legal Services Authority:",
            ])
            
            # Domain-specific authority guidance
            if domain == 'familylaw':
                base_steps.extend([
                    "- **Family Court**: Most family law cases are handled here",
                    "- **District Legal Services Authority (DLSA)**: Located at District Court complex",
                    "- **State Legal Services Authority (SLSA)**: For complex cases or appeals",
                    "- 📞 **Women's Helpline**: 1091 for immediate domestic violence support"
                ])
            elif domain == 'consumerprotection':
                base_steps.extend([
                    "- **Consumer Forum**: District/State/National based on claim value",
                    "- **District Consumer Disputes Redressal Commission**: For claims < ₹1 crore",
                    "- 📞 **National Consumer Helpline**: 1800-11-4000"
                ])
            elif domain == 'employmentlaw':
                base_steps.extend([
                    "- **Labor Court**: For termination/wage disputes",
                    "- **Labor Commissioner Office**: For conciliation",
                    "- **District Legal Services Authority (DLSA)**: For legal aid",
                    "- 📞 **Labor Helpline**: 155-214 (varies by state)"
                ])
            elif domain == 'criminallaw':
                base_steps.extend([
                    "- **District Court**: For criminal cases",
                    "- **Legal Aid Cell at Police Station**: Available 24/7",
                    "- **District Legal Services Authority (DLSA)**: For free lawyer",
                    "- 📞 **Police Helpline**: 100 | **Women Helpline**: 1091"
                ])
            else:
                base_steps.extend([
                    "- **District Legal Services Authority (DLSA)**: At District Court",
                    "- **Taluk Legal Services Committee (TLSC)**: At Taluk level",
                    "- **State Legal Services Authority (SLSA)**: For complex cases",
                    "- 🌐 **Find Nearest**: Visit https://nalsa.gov.in"
                ])
            
            base_steps.extend([
                "",
                "## 📝 Step 3: Submit Legal Aid Application",
                "Fill out the legal aid application form with:",
                "- Your personal details and case information",
                "- Income and category details",
                "- Nature of legal problem",
                "- Documents supporting your eligibility",
                "",
                "💡 **Tip**: Applications are FREE. Do not pay anyone for legal aid services.",
            ])
            
            # Domain-specific filing guidance
            if domain == 'familylaw':
                base_steps.extend([
                    "",
                    "## ⚖️ Step 4: File Your Case (Family Law)",
                    "- **Divorce/Maintenance**: File petition in Family Court",
                    "- **Domestic Violence**: File under Protection of Women from Domestic Violence Act, 2005",
                    "- **Child Custody**: File guardianship petition",
                    "- **Timeline**: Legal aid lawyer assigned within 15-30 days",
                ])
            elif domain == 'consumerprotection':
                base_steps.extend([
                    "",
                    "## 🛒 Step 4: File Consumer Complaint",
                    "- **Online Filing**: Available at https://edaakhil.nic.in",
                    "- **Offline Filing**: Visit Consumer Forum with copies of:",
                    "  - Purchase invoice/receipt",
                    "  - Warranty/guarantee card",
                    "  - Correspondence with seller/manufacturer",
                    "- **Fee**: Minimal court fee (₹100-500 depending on claim)",
                    "- **Timeline**: Hearing within 21 days of filing",
                ])
            elif domain == 'employmentlaw':
                base_steps.extend([
                    "",
                    "## 💼 Step 4: File Labor Complaint",
                    "- **Conciliation**: First attempt settlement with Labor Commissioner",
                    "- **Labor Court**: If conciliation fails, file case in Labor Court",
                    "- **Documents Required**: Appointment letter, termination notice, salary slips",
                    "- **Timeline**: Conciliation within 45 days, court case 6-12 months",
                ])
            elif domain == 'criminallaw':
                base_steps.extend([
                    "",
                    "## 🚨 Step 4: Legal Proceedings (Criminal)",
                    "- **FIR Already Filed**: Approach Legal Aid Cell for defense lawyer",
                    "- **Need to File FIR**: Police station or online (state-specific portals)",
                    "- **Bail Application**: Legal aid lawyer will file bail application",
                    "- **Timeline**: Lawyer assigned within 24 hours for custody cases",
                ])
            else:
                base_steps.extend([
                    "",
                    "## ⚖️ Step 4: Proceed with Legal Case",
                    "- Your assigned lawyer will guide you through court procedures",
                    "- Attend all hearings (your lawyer will inform you of dates)",
                    "- Keep all original documents safe, submit only copies",
                    "- **Timeline**: Case processing varies by court (3-18 months typically)",
                ])
            
            base_steps.extend([
                "",
                "## ⏰ Step 5: Priority Processing (if urgent)",
                "Request expedited processing if:",
                "- 🚨 **Emergency**: Imminent eviction, arrest, or threat to life",
                "- 👶 **Child Involved**: Child custody or maintenance",
                "- 🩹 **Medical Emergency**: Health crisis requiring immediate legal intervention",
                "- 📅 **Court Deadline**: Nearby court hearing or legal deadline",
                "",
                "📞 **24/7 Emergency Legal Aid Helpline**: 15100 (NALSA)",
            ])
            
            # Additional domain-specific resources
            if domain == 'familylaw':
                base_steps.extend([
                    "",
                    "## 📞 Additional Support for Family Law",
                    "- **Women's Helpline**: 1091 (24/7)",
                    "- **Domestic Violence Helpline**: 181",
                    "- **Child Helpline**: 1098",
                    "- **NCW (National Commission for Women)**: 011-26944740"
                ])
            elif domain == 'criminallaw':
                base_steps.extend([
                    "",
                    "## 📞 Additional Support for Criminal Cases",
                    "- **Police Helpline**: 100",
                    "- **Women Helpline**: 1091",
                    "- **Senior Citizen Helpline**: 14567",
                    "- **Cyber Crime Helpline**: 1930"
                ])
            
            return base_steps
        
        else:
            # ===== NOT ELIGIBLE CASES - ALTERNATIVE OPTIONS =====
            
            not_eligible_steps = [
                "## ❌ You Do Not Qualify for Free Legal Aid",
                f"Based on the Legal Services Authorities Act, 1987, Section 12, your case does not meet eligibility criteria for free legal aid.",
                "",
                "**Common reasons for ineligibility:**",
                "- Annual income exceeds the threshold for your category",
                "- Wealth indicators (business ownership, multiple properties) present",
                "- Case type not covered under legal aid provisions",
                "",
                "---",
                "",
                "## 🔄 Alternative Legal Support Options",
                "",
                "### 1. 💼 Pro Bono Services",
                "Many lawyers offer free or reduced-fee services:",
                "- **Contact Local Bar Association**: Ask for pro bono lawyers list",
                "- **Law Firms**: Many have CSR initiatives offering free consultations",
                "- **NGOs**: Legal aid NGOs may help based on case merit",
                "- 🌐 **Directory**: https://probono-india.in",
                "",
                "### 2. 🎓 Law School Legal Clinics",
                "Free legal advice from law students (supervised by professors):",
                "- **National Law Universities (NLUs)**: Offer free legal clinics",
                "- **Government Law Colleges**: Weekend legal aid camps",
                "- **Legal Literacy Centers**: Basic legal guidance",
                "",
                "### 3. 🤝 Specialized NGOs",
            ]
            
            # Domain-specific NGO recommendations
            if domain == 'familylaw':
                not_eligible_steps.extend([
                    "**Family Law NGOs:**",
                    "- **Lawyers Collective Women's Rights Initiative**",
                    "- **Majlis Legal Centre** (women's rights)",
                    "- **Centre for Social Justice** (family matters)",
                ])
            elif domain == 'consumerprotection':
                not_eligible_steps.extend([
                    "**Consumer Rights NGOs:**",
                    "- **Consumer Guidance Society of India**",
                    "- **Voluntary Organization in Interest of Consumer Education (VOICE)**",
                    "- **Consumer Education and Research Centre**",
                ])
            elif domain == 'employmentlaw':
                not_eligible_steps.extend([
                    "**Labor Rights NGOs:**",
                    "- **Centre for Labour Research and Action**",
                    "- **National Campaign Committee for Unorganised Sector Workers**",
                    "- **Aajeevika Bureau** (worker rights)",
                ])
            else:
                not_eligible_steps.extend([
                    "**General Legal NGOs:**",
                    "- **Human Rights Law Network**",
                    "- **India Legal Aid and Advice Board**",
                    "- **Legal Services India** (online guidance)",
                ])
            
            not_eligible_steps.extend([
                "",
                "### 4. 💳 Payment Plans & Loans",
                "Affordable legal fee options:",
                "- **Installment Plans**: Many lawyers accept monthly payments",
                "- **Fixed Fee Packages**: For specific case types (₹10,000-50,000)",
                "- **Legal Insurance**: Some insurance plans cover legal fees",
                "- **Litigation Financing**: Third-party funding for strong cases",
                "",
                "### 5. 📞 Free Legal Helplines & Online Guidance",
                "Get basic legal advice for free:",
                "- **Tele-Law Services**: 9in1 Helpline (NALSA) - Basic guidance",
                "- **MyGov Helpdesk**: https://mygov.in (government queries)",
                "- **India Code Portal**: https://indiacode.nic.in (read laws)",
                "- **Legal Services India**: Online legal information",
                "",
                "### 6. 🏛️ Court Self-Help Centers",
                "Many courts have self-help desks:",
                "- **District Court Self-Help Desk**: Free form filling assistance",
                "- **eCourts Services Portal**: https://ecourts.gov.in",
                "- **Case Status Tracking**: Monitor your case online",
                "",
                "### 7. 💡 Mediation & Alternative Dispute Resolution (ADR)",
                "Cheaper than court litigation:",
                "- **Lok Adalat (People's Court)**: Settle disputes amicably",
                "- **Mediation Centers**: At District Courts (nominal fee)",
                "- **Arbitration**: For contract disputes",
                "- **Negotiation**: Try direct settlement before court",
                "",
                "---",
                "",
                "## 📋 If You Believe This Assessment Is Incorrect",
                "You can request a manual review:",
                "",
                "**Required documents for review:**",
                "1. **Income Proof**: Last 3 months salary slips OR Income certificate",
                "2. **Category Certificate**: SC/ST/OBC certificate (if applicable)",
                "3. **Asset Declaration**: Details of property, business, investments",
                "4. **Special Circumstances**: Medical bills, disaster certificate, etc.",
                "",
                "**Where to apply:**",
                "- Visit your **District Legal Services Authority (DLSA)**",
                "- Submit written application with documents",
                "- Secretary will review and make final decision",
                "- Appeal to **State Legal Services Authority (SLSA)** if rejected",
                "",
                "🌐 **Find Your DLSA**: https://nalsa.gov.in → State → District",
                "",
                "---",
                "",
                "## ⚠️ Important Notes",
                "- ❌ **Do NOT pay agents/middlemen** for legal aid - it's FREE",
                "- ✅ **Eligibility can change** if your income/situation changes",
                "- 📞 **Report fraud**: If someone demands money for legal aid services",
                "- 🔒 **Your data is safe**: Legal aid applications are confidential",
            ])
            
            return not_eligible_steps
    
    def _fallback_prediction(self, case_data: Dict) -> HybridPrediction:
        """Conservative fallback when all models fail - requires manual review"""
        return HybridPrediction(
            case_id=case_data.get('sample_id', 'unknown'),
            eligible=False,              # Conservative: do not grant eligibility without analysis
            confidence=0.35,             # Low confidence to trigger review
            method_used='fallback',
            reasoning="All models failed - fallback conservative: manual review required",
            requires_review=True,        # Flag for manual review
            uncertainty=0.85             # High uncertainty
        )
    
    def fit_calibrator(self, validation_data: List[Dict]):
        """
        Fit confidence calibrators on validation set.
        
        Args:
            validation_data: List of cases with 'expected_eligibility' labels
        """
        logger.info("Fitting confidence calibrators...")
        
        bert_scores = []
        prolog_scores = []
        gnn_scores = []
        true_labels = []
        
        for case in validation_data:
            try:
                # Get predictions
                bert_res = self._predict_with_bert_safe(case)
                prolog_res = self._predict_with_prolog_safe(case)
                gnn_res = self._predict_with_gnn_safe(case)
                
                true_label = float(case.get('expected_eligibility', 0.0))
                
                if bert_res:
                    bert_scores.append((bert_res.confidence, bert_res.eligible))
                if prolog_res:
                    prolog_scores.append((prolog_res.confidence, prolog_res.eligible))
                if gnn_res:
                    gnn_scores.append((gnn_res.confidence, gnn_res.eligible))
                
                true_labels.append(true_label)
                
            except Exception as e:
                logger.warning(f"Failed to process case for calibration: {e}")
                continue
        
        # Fit calibrators
        if len(true_labels) >= 10:
            self.calibrator.fit(bert_scores, prolog_scores, gnn_scores, true_labels)
        else:
            logger.warning("⚠️  Insufficient validation data for calibration")
    
    def update_ensemble_weights(self, validation_results: List[Dict]):
        """
        Adaptively update ensemble weights based on validation performance.
        
        Args:
            validation_results: List of dicts with 'bert_prob', 'prolog_prob', 'gnn_prob', 'true_label'
        """
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                """Objective: maximize F1 score on validation set."""
                bert_weight, prolog_weight, gnn_weight = weights
                
                # Ensure weights sum to 1
                if abs(bert_weight + prolog_weight + gnn_weight - 1.0) > 0.01:
                    return 1.0  # Penalty
                
                # Simulate predictions with these weights
                correct = 0
                total = len(validation_results)
                
                for result in validation_results:
                    weighted_prob = (
                        bert_weight * result.get('bert_prob', 0.5) +
                        prolog_weight * result.get('prolog_prob', 0.5) +
                        gnn_weight * result.get('gnn_prob', 0.5)
                    )
                    pred = weighted_prob > 0.5
                    if pred == result.get('true_label', False):
                        correct += 1
                
                accuracy = correct / total if total > 0 else 0.0
                return 1.0 - accuracy  # Minimize (1 - accuracy)
            
            # Optimize weights
            initial_weights = [0.33, 0.33, 0.34]
            bounds = [(0.1, 0.7), (0.1, 0.7), (0.1, 0.7)]  # Allow 10-70% range
            constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] + w[2] - 1.0}
            
            result = minimize(
                objective, initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                self.ensemble_params['bert_weight'] = result.x[0]
                self.ensemble_params['prolog_weight'] = result.x[1]
                self.ensemble_params['gnn_weight'] = result.x[2]
                logger.info(f"✅ Updated ensemble weights: BERT={result.x[0]:.3f}, Prolog={result.x[1]:.3f}, GNN={result.x[2]:.3f}")
            else:
                logger.warning("⚠️  Weight optimization failed, keeping current weights")
                
        except ImportError:
            logger.warning("⚠️  scipy not available, skipping weight optimization")
        except Exception as e:
            logger.warning(f"⚠️  Weight optimization failed: {e}")
    
    def _get_tax_law_next_steps(self, annual_income_lakhs: float) -> List[str]:
        """Tax law alternative options (not eligible)"""
        return [
            "## ❌ Income Tax Disputes Are NOT Covered Under Legal Aid",
            "",
            f"**Your Income**: ₹{annual_income_lakhs:.1f} lakhs annually",
            "",
            "Tax matters are **commercial/financial issues** under the Income Tax Act, 1961, NOT covered by the Legal Services Authorities Act, 1987.",
            "",
            "**Why not covered:**",
            "- Legal aid applies to civil/criminal/family disputes",
            "- Tax matters require specialized tax professionals (CAs)",
            "- Income Tax Department has its own grievance mechanisms",
            "",
            "---",
            "",
            "## 💼 Recommended Actions for Tax Disputes",
            "",
            "### 1. 🧾 Hire a Chartered Accountant (CA)",
            "**Best option for tax disputes:**",
            "- File revised returns / respond to notices",
            "- Represent before Income Tax Officer",
            "- Handle assessment proceedings",
            "- **Cost**: ₹5,000-50,000 (varies by complexity)",
            "- **Find CA**: https://icai.org → 'Find a CA'",
            "",
            "### 2. 📞 Income Tax Helpdesk (FREE)",
            "**For basic queries and guidance:**",
            "- **Helpline**: 1800-180-1961 (toll-free)",
            "- **Email**: grivcell@incometax.gov.in",
            "- **e-Filing Support**: https://incometaxindiaefiling.gov.in",
            "- **Timing**: Monday-Friday, 9 AM - 6 PM",
            "",
            "### 3. 🏛️ Income Tax Ombudsman",
            "**For complaints against IT Department:**",
            "- File complaint if harassment/illegal notice",
            "- Free service, no fee",
            "- **Website**: https://incometaxindia.gov.in/pages/ombudsman.aspx",
            "- **Jurisdiction**: Appeals within ₹50 lakh disputed amount",
            "",
            "### 4. 📊 Tax Consultant / Advocate",
            "**For complex disputes:**",
            "- Appeals to Commissioner (Appeals)",
            "- Income Tax Appellate Tribunal (ITAT) cases",
            "- Writ petitions in High Court",
            "- **Cost**: ₹10,000-2,00,000+ (complex cases)",
            "",
            "### 5. 📝 E-Nivaran Portal (Online Grievance)",
            "**Submit online complaints:**",
            "- **Portal**: https://enivaran.incometax.gov.in",
            "- Track status of grievance",
            "- Response within 30 days",
            "- Escalation to higher authorities",
            "",
            "### 6. 💡 Free Tax Advice (Limited)",
            "**Some organizations offer basic help:**",
            "- **TaxSpanner**: Free consultation",
            "- **ClearTax**: Free guidance articles",
            "- **CA Association Free Camps**: Check local listings",
            "",
            "---",
            "",
            "## ⚠️ Critical Tax Compliance Tips",
            "- ❌ **Do NOT ignore tax notices** - respond within 30 days",
            "- ✅ **File returns on time** - avoid penalties (₹5,000+)",
            "- 📄 **Keep documents** for 7 years (returns, receipts, proofs)",
            "- 🔒 **Use official portals** only - avoid fraud/fake websites",
            "- ⏰ **Respond promptly** - late response reduces appeal chances",
            "",
            "## 📞 Emergency Contacts",
            "- **Tax Helpline**: 1800-180-1961",
            "- **Cyber Fraud (if scam)**: 1930",
            "- **Consumer Helpline**: 1800-11-4000",
            "",
            "🌐 **Official Income Tax Website**: https://incometaxindia.gov.in"
        ]
    
    def batch_predict(self, cases: List[Dict]) -> List[HybridPrediction]:
        """FIX: Filter out None predictions"""
        predictions = []
        
        for i, case in enumerate(cases):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(cases)} cases...")
            
            pred = self.predict(case)
            
            # FIX: Guarantee non-None
            if pred is None:
                pred = self._fallback_prediction(case)
            
            predictions.append(pred)
        
        # Stats
        method_counts = {}
        for pred in predictions:
            method_counts[pred.method_used] = method_counts.get(pred.method_used, 0) + 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Hybrid Prediction Complete: {len(predictions)} cases")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {method}: {count} ({count/len(predictions)*100:.1f}%)")
        logger.info(f"{'='*60}\n")
        
        return predictions


# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================

# Alias for backward compatibility with existing code
HybridPredictor = IntelligentHybridPredictor