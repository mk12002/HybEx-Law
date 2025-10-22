"""
HybEx Advanced Hybrid Prediction System
Intelligently combines Prolog, GNN, and BERT with learned confidence calibration
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

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
        self.calibrated = False
        
    def fit(self, bert_scores, prolog_scores, gnn_scores, true_labels):
        """
        Fit calibrators on validation set.
        
        Args:
            bert_scores: List of (confidence, prediction) tuples from BERT
            prolog_scores: List of (confidence, prediction) tuples from Prolog
            gnn_scores: List of (confidence, prediction) tuples from GNN
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

class IntelligentHybridPredictor:
    """
    Advanced hybrid predictor with:
    1. Learned ensemble weights
    2. Confidence calibration
    3. Dynamic method selection
    4. Uncertainty quantification
    """
    
    def __init__(self, prolog_engine, gnn_model, bert_model, config):
        self.prolog = prolog_engine
        self.gnn = gnn_model
        self.bert = bert_model
        self.config = config
        
        # Confidence calibrator
        self.calibrator = ConfidenceCalibrator()
        
        # Learned ensemble parameters (tune on validation set)
        self.ensemble_params = {
            'prolog_threshold': 0.90,      # Use Prolog if confidence > this
            'gnn_threshold': 0.80,          # Use GNN if confidence > this
            'bert_threshold': 0.85,         # Use BERT if confidence > this
            'prolog_weight': 0.50,          # Weight in ensemble (favor Prolog for legal rules)
            'bert_weight': 0.30,
            'gnn_weight': 0.20,
            'uncertainty_threshold': 0.70,  # Flag for review if confidence < this
            'conflict_penalty': 0.15        # Reduce confidence when systems disagree
        }
        
        # Track performance for adaptive weighting
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
            
            # Get predictions from all systems
            bert_result = self._predict_with_bert_safe(case_data)
            prolog_result = self._predict_with_prolog_safe(case_data)
            gnn_result = self._predict_with_gnn_safe(case_data)
            
            # If all failed, use fallback
            if not any([bert_result, prolog_result, gnn_result]):
                return self._fallback_prediction(case_data)
            
            # Extract raw confidences
            bert_conf = bert_result.confidence if bert_result else 0.5
            prolog_conf = prolog_result.confidence if prolog_result else 0.5
            gnn_conf = gnn_result.confidence if gnn_result else 0.5
            
            # Calibrate confidences
            bert_conf_cal = self.calibrator.calibrate_bert(bert_conf)
            prolog_conf_cal = self.calibrator.calibrate_prolog(prolog_conf)
            gnn_conf_cal = self.calibrator.calibrate_gnn(gnn_conf)
            
            # Make hybrid decision with intelligent routing
            decision = self._make_hybrid_decision(
                bert_result, prolog_result, gnn_result,
                bert_conf_cal, prolog_conf_cal, gnn_conf_cal,
                entities
            )
            
            # Calculate uncertainty (epistemic + aleatoric)
            uncertainty = self._calculate_uncertainty(
                bert_result, prolog_result, gnn_result, entities
            )
            
            # Determine if human review needed
            requires_review = self._needs_review(
                decision['confidence'], uncertainty, entities
            )
            
            return HybridPrediction(
                case_id=case_id,
                eligible=decision['eligible'],
                confidence=decision['confidence'],
                method_used=decision['method'],
                bert_result=bert_result.bert_result if bert_result else None,
                prolog_result=prolog_result.prolog_result if prolog_result else None,
                gnn_result=gnn_result.gnn_result if gnn_result else None,
                reasoning=decision['reasoning'],
                uncertainty=uncertainty,
                requires_review=requires_review,
                calibrated_confidences={
                    'bert': bert_conf_cal,
                    'prolog': prolog_conf_cal,
                    'gnn': gnn_conf_cal
                },
                decision_rationale=decision.get('rationale', '')
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {case_data.get('sample_id')}: {e}")
            return self._fallback_prediction(case_data)
    
    def _make_hybrid_decision(self, bert_res, prolog_res, gnn_res,
                             bert_conf_cal, prolog_conf_cal, gnn_conf_cal, entities):
        """
        UPDATED: Intelligent hybrid decision with ALL 5 models
        
        Models:
        1. Prolog (symbolic reasoning)
        2. GNN (graph-based)
        3. Eligibility Predictor (BERT)
        4. Domain Classifier (BERT)
        5. EnhancedLegalBERT (Multi-task BERT)
        
        Strategies:
        1. High-confidence Prolog (trust symbolic reasoning)
        2. Conflict resolution between systems
        3. Agreement boosting
        4. Weighted ensemble fallback
        """
        
        # Try to get additional predictions from domain classifier and enhanced BERT
        case_data = {'extracted_entities': entities, 'query': entities.get('query', '')}
        domain_res = self.predict_with_domain_classifier_safe(case_data)
        enhanced_res = self.predict_with_enhanced_bert_safe(case_data)
        
        # Strategy 1: High-confidence Prolog (trust legal rules)
        if prolog_res and prolog_conf_cal >= self.ensemble_params['prolog_threshold']:
            return {
                'eligible': prolog_res.eligible,
                'confidence': prolog_conf_cal,
                'method': 'prolog',
                'reasoning': f"Prolog high confidence ({prolog_conf_cal:.2f})",
                'rationale': f"Legal rules clearly indicate {'ELIGIBLE' if prolog_res.eligible else 'NOT ELIGIBLE'}"
            }
        
        # Strategy 2: Check for conflict between all systems
        predictions = []
        if bert_res:
            predictions.append(('bert', bert_res.eligible, bert_conf_cal))
        if prolog_res:
            predictions.append(('prolog', prolog_res.eligible, prolog_conf_cal))
        if gnn_res:
            predictions.append(('gnn', gnn_res.eligible, gnn_conf_cal))
        if domain_res:
            predictions.append(('domain_classifier', domain_res.eligible, domain_res.confidence))
        if enhanced_res:
            predictions.append(('enhanced_bert', enhanced_res.eligible, enhanced_res.confidence))
        
        # Check if there's disagreement
        if len(predictions) >= 2:
            eligibilities = [p[1] for p in predictions]
            if len(set(eligibilities)) > 1:  # Systems disagree
                
                # If Prolog says NOT eligible and others say eligible
                if prolog_res and not prolog_res.eligible and prolog_conf_cal > 0.75:
                    # Trust Prolog for negative cases (stricter legal interpretation)
                    return {
                        'eligible': False,
                        'confidence': prolog_conf_cal * 0.9,  # Slightly reduce due to conflict
                        'method': 'prolog',
                        'reasoning': f"Conflict resolved: Prolog NOT eligible ({prolog_conf_cal:.2f}) overrides neural models",
                        'rationale': "Legal rules take precedence for rejection cases"
                    }
                
                # If Prolog says eligible and others say NOT eligible
                elif prolog_res and prolog_res.eligible and prolog_conf_cal > 0.80:
                    # Trust Prolog for positive cases (legal entitlement)
                    return {
                        'eligible': True,
                        'confidence': prolog_conf_cal * 0.9,
                        'method': 'prolog',
                        'reasoning': f"Conflict resolved: Prolog eligible ({prolog_conf_cal:.2f}) overrides neural models",
                        'rationale': "Legal entitlement established by rules"
                    }
                
                # Can't resolve - use weighted ensemble with conflict penalty
                return self._weighted_ensemble(
                    bert_res, prolog_res, gnn_res, domain_res, enhanced_res,
                    bert_conf_cal, prolog_conf_cal, gnn_conf_cal,
                    conflict_penalty=self.ensemble_params['conflict_penalty']
                )
        
        # Strategy 3: Systems agree - boost confidence
        if len(predictions) >= 2:
            if len(set(eligibilities)) == 1:  # All agree
                avg_conf = np.mean([p[2] for p in predictions])
                
                # Boost confidence when systems agree
                boosted_conf = min(0.98, avg_conf * 1.15)
                
                return {
                    'eligible': eligibilities[0],
                    'confidence': boosted_conf,
                    'method': 'ensemble',
                    'reasoning': f"All {len(predictions)} systems agree ({eligibilities[0]}): avg_conf={avg_conf:.2f}",
                    'rationale': "Strong consensus across all models"
                }
        
        # Strategy 4: Default weighted ensemble
        return self._weighted_ensemble(
            bert_res, prolog_res, gnn_res, domain_res, enhanced_res,
            bert_conf_cal, prolog_conf_cal, gnn_conf_cal
        )
    
    def _weighted_ensemble(self, bert_res, prolog_res, gnn_res, domain_res, enhanced_res,
                          bert_conf_cal, prolog_conf_cal, gnn_conf_cal,
                          conflict_penalty=0.0):
        """
        UPDATED: Weighted ensemble with ALL 5 models and adaptive weights.
        
        Default weights (can be optimized):
        - Prolog: 0.30 (legal rules)
        - BERT (Eligibility): 0.20 (specialized model)
        - GNN: 0.15 (graph reasoning)
        - Domain Classifier: 0.15 (domain detection)
        - EnhancedBERT: 0.20 (multi-task model)
        """
        
        # Calculate weighted probability
        eligible_score = 0.0
        total_weight = 0.0
        components = []
        
        # Base 3 models (existing)
        if bert_res:
            weight = self.ensemble_params.get('bert_weight', 0.20)
            score = bert_conf_cal if bert_res.eligible else (1 - bert_conf_cal)
            eligible_score += weight * score
            total_weight += weight
            components.append(f"BERT={score:.2f}({weight})")
        
        if prolog_res:
            weight = self.ensemble_params.get('prolog_weight', 0.30)
            score = prolog_conf_cal if prolog_res.eligible else (1 - prolog_conf_cal)
            eligible_score += weight * score
            total_weight += weight
            components.append(f"Prolog={score:.2f}({weight})")
        
        if gnn_res:
            weight = self.ensemble_params.get('gnn_weight', 0.15)
            score = gnn_conf_cal if gnn_res.eligible else (1 - gnn_conf_cal)
            eligible_score += weight * score
            total_weight += weight
            components.append(f"GNN={score:.2f}({weight})")
        
        # NEW: Domain Classifier
        if domain_res:
            weight = self.ensemble_params.get('domain_weight', 0.15)
            score = domain_res.confidence if domain_res.eligible else (1 - domain_res.confidence)
            eligible_score += weight * score
            total_weight += weight
            components.append(f"Domain={score:.2f}({weight})")
        
        # NEW: EnhancedBERT (multi-task)
        if enhanced_res:
            weight = self.ensemble_params.get('enhanced_bert_weight', 0.20)
            score = enhanced_res.confidence if enhanced_res.eligible else (1 - enhanced_res.confidence)
            eligible_score += weight * score
            total_weight += weight
            components.append(f"Enhanced={score:.2f}({weight})")
        
        if total_weight > 0:
            eligible_score /= total_weight
        
        # Apply conflict penalty if provided
        final_confidence = eligible_score * (1 - conflict_penalty)
        
        # Decision
        eligible = eligible_score > 0.5
        
        return {
            'eligible': eligible,
            'confidence': final_confidence,
            'method': 'ensemble',
            'reasoning': f"Weighted ensemble: {', '.join(components)}",
            'rationale': f"Combined prediction with {'conflict penalty' if conflict_penalty > 0 else 'full confidence'}"
        }
    
    def _calculate_uncertainty(self, bert_res, prolog_res, gnn_res, entities):
        """
        Calculate prediction uncertainty.
        
        Components:
        1. Model disagreement (epistemic uncertainty)
        2. Low confidence (aleatoric uncertainty)
        3. Near decision boundary (income threshold)
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
                return HybridPrediction(
                    case_id=case_id,
                    eligible=prolog_reasoning.eligible,
                    confidence=prolog_reasoning.confidence,
                    method_used='prolog',
                    prolog_result={'reasoning': prolog_reasoning},
                    reasoning=prolog_reasoning.primary_reason
                )
            return None
            
        except Exception as e:
            logger.warning(f"Prolog failed for {case_data.get('sample_id')}: {str(e)[:100]}")
            return None
    
    def _predict_with_gnn_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """FIX: Correct GNN forward pass for 2-layer model"""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            entities = case_data.get('extracted_entities', {})
            
            kg_engine = getattr(self.gnn, 'kg_engine', None) or getattr(self.gnn, '_knowledge_graph_engine_cache', None)
            if kg_engine is None:
                return None
            
            graph = kg_engine.create_case_graph(entities, label=0.0)
            gnn_model = getattr(kg_engine, 'model', None)
            if gnn_model is None:
                return None
            
            graph = graph.to(self.gnn.device)
            
            with torch.no_grad():
                gnn_model.eval()
                x, edge_index = graph.x, graph.edge_index
                
                # FIX: Only 2 conv layers (conv1, conv2)
                h = gnn_model.conv1(x, edge_index)
                h = torch.relu(h)
                h = gnn_model.conv2(h, edge_index)
                h = torch.relu(h)
                # No conv3!
                
                # Global pooling
                graph_embedding = torch.mean(h, dim=0, keepdim=True)
                
                # Classification
                logits = gnn_model.readout(graph_embedding)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
            
            return HybridPrediction(
                case_id=case_id,
                eligible=bool(pred_class),
                confidence=confidence,
                method_used='gnn',
                gnn_result={'probs': probs.cpu().numpy().tolist()},
                reasoning=f"GNN: {len(entities)} features"
            )
            
        except Exception as e:
            logger.warning(f"GNN failed for {case_data.get('sample_id')}: {str(e)[:100]}")
            return None
    
    def _predict_with_bert_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """FIX: Handle 1D and 2D BERT outputs properly"""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            
            if not query:
                return None
            
            inputs = self.bert.tokenizer(
                query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )
            
            inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                self.bert.eligibility_model.eval()
                outputs = self.bert.eligibility_model(**inputs)
                
                # Handle dict/tensor/object outputs
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get(list(outputs.keys())[0]))
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits
                
                # FIX: Handle both 1D and 2D logits
                if logits.dim() == 1:
                    # Model outputs raw scores [batch_size] - treat as binary logits
                    # Reshape to [batch_size, 2] with [not_eligible_score, eligible_score]
                    logits = torch.stack([1 - logits, logits], dim=1)
                elif logits.dim() == 2 and logits.size(1) == 1:
                    # Model outputs [batch_size, 1] - convert to [batch_size, 2]
                    logits = torch.cat([1 - logits, logits], dim=1)
                
                # Now safe to apply softmax on dim=1
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
            
            return HybridPrediction(
                case_id=case_id,
                eligible=bool(pred_class),
                confidence=confidence,
                method_used='bert',
                bert_result={'probs': probs.cpu().numpy().tolist()},
                reasoning=f"BERT: {len(query)} chars"
            )
            
        except Exception as e:
            logger.warning(f"BERT failed for {case_data.get('sample_id')}: {str(e)[:100]}")
            return None
    
    def predict_with_domain_classifier_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """Get domain classification as eligibility proxy"""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            
            if not query:
                return None
            
            # Domain classifier
            domain_classifier = getattr(self.bert, 'domain_classifier', None)
            if domain_classifier is None:
                return None
            
            inputs = self.bert.tokenizer(
                query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                domain_classifier.eval()
                logits = domain_classifier(**inputs).logits
                probs = torch.sigmoid(logits)  # Multi-label
                
                # Use legal_aid domain probability as eligibility indicator
                legal_aid_idx = 0  # Assuming legal_aid is index 0
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
            logger.warning(f"Domain classifier failed: {str(e)[:100]}")
            return None
    
    def predict_with_enhanced_bert_safe(self, case_data: Dict) -> Optional[HybridPrediction]:
        """Get prediction from EnhancedLegalBERT (multi-task model)"""
        try:
            case_id = case_data.get('sample_id', 'unknown')
            query = case_data.get('query', '')
            
            if not query:
                return None
            
            # EnhancedLegalBERT
            enhanced_bert = getattr(self.bert, 'enhanced_bert', None)
            if enhanced_bert is None:
                return None
            
            inputs = self.bert.tokenizer(
                query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                enhanced_bert.eval()
                outputs = enhanced_bert(**inputs)
                
                # EnhancedBERT returns both eligibility and domain
                eligibility_logits = outputs['eligibility_logits']
                domain_logits = outputs.get('domain_logits', None)
                
                # Get eligibility prediction
                if eligibility_logits.dim() == 1:
                    prob = torch.sigmoid(eligibility_logits).item()
                else:
                    probs = torch.softmax(eligibility_logits, dim=1)
                    prob = probs[0, 1].item()  # Probability of eligible class
                
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
            logger.warning(f"EnhancedBERT failed: {str(e)[:100]}")
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
                weight = self.PROLOG_WEIGHT
                score = prolog_pred.confidence if prolog_pred.eligible else (1 - prolog_pred.confidence)
                eligible_score += weight * score
                total_weight += weight
            
            if gnn_pred:
                weight = self.GNN_WEIGHT
                score = gnn_pred.confidence if gnn_pred.eligible else (1 - gnn_pred.confidence)
                eligible_score += weight * score
                total_weight += weight
            
            if bert_pred:
                weight = self.BERT_WEIGHT
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
    
    def _fallback_prediction(self, case_data: Dict) -> HybridPrediction:
        """Absolute last resort - majority class"""
        return HybridPrediction(
            case_id=case_data.get('sample_id', 'unknown'),
            eligible=True,
            confidence=0.70,
            method_used='fallback',
            reasoning="All models failed - using majority class"
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
