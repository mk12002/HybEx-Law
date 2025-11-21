# streamlit_app.py
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict
from dataclasses import is_dataclass, asdict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# NEW: Import Gemini predictor
from hybex_system.predictor import GeminiEligibilityPredictor

# Import translator for multilingual support
try:
    from hybex_system.translator import MultilingualTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    st.warning("⚠️ Translator module not available. Running in English-only mode.")

# Import numpy and torch for robust serialization
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

# E: Shorten logging for Streamlit (suppress repetitive INFO logs)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("hybex_system").setLevel(logging.WARNING)
logging.getLogger("hybex_system.prolog_engine").setLevel(logging.WARNING)
logging.getLogger("hybex_system.knowledge_graph_engine").setLevel(logging.WARNING)
logging.getLogger("hybex_system.hybrid_predictor").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------- Robust JSON Serialization ----------
def _make_json_safe(obj):
    """
    Recursively convert many python objects to JSON-serializable primitives.
    Handles dataclasses, objects with __dict__, numpy arrays, torch tensors, sets, enums, decimals.
    """
    # primitives:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy types
    if np is not None:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    # torch tensors
    if torch is not None and isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return str(obj)

    # dataclass
    if is_dataclass(obj):
        try:
            return _make_json_safe(asdict(obj))
        except Exception:
            # fallback: iterate fields
            return {k: _make_json_safe(getattr(obj, k)) for k in getattr(obj, '__dataclass_fields__', {})}

    # dict-like
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(x) for x in obj]

    # object with to_dict or asdict
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _make_json_safe(obj.to_dict())

    # object with __dict__
    if hasattr(obj, "__dict__"):
        # try to avoid serializing functions/methods
        data = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            data[k] = _make_json_safe(v)
        return data

    # fallback to string
    try:
        return str(obj)
    except Exception:
        return None

# UI helper CSS (small)
st.set_page_config(page_title="HybEx-Law — Quick Eligibility", layout="wide")
st.markdown(
    """
    <style>
      .stApp { font-family: "Inter", sans-serif; }
      .result-box { padding: 12px; border-radius: 8px; background: #f7f9fc; }
      .muted { color: #6b7280; }
      .small { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_predictor():
    """
    Load HybEx-Law Hybrid Prediction System
    Combines Prolog, GNN, and BERT for intelligent eligibility analysis
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except:
            pass
    
    # Initialize predictor silently
    try:
        predictor = GeminiEligibilityPredictor(api_key=api_key)
        return predictor
    except Exception:
        # Fallback initialization - still works without API
        return GeminiEligibilityPredictor(api_key=None)


@st.cache_resource(show_spinner=False)
def load_translator():
    """Load Azure translator once"""
    if not TRANSLATOR_AVAILABLE:
        return None
    try:
        return MultilingualTranslator()
    except Exception as e:
        st.warning(f"⚠️ Translator initialization failed: {e}")
        return None


def translate(text: str, lang: str, translator=None) -> str:
    """Helper function to translate text"""
    if lang == 'en' or not translator:
        return text
    try:
        return translator.translate(text, lang, 'en')
    except Exception:
        return text  # Fallback to original text


def translate_result(result: Dict, lang: str, translator=None) -> Dict:
    """Translate prediction result to target language"""
    if lang == 'en' or not translator:
        return result
    
    try:
        # Translate key fields
        translated = result.copy()
        
        # Translate explanation
        if 'explanation' in translated and isinstance(translated['explanation'], dict):
            if 'summary' in translated['explanation']:
                translated['explanation']['summary'] = translator.translate(
                    result['explanation']['summary'], lang, 'en'
                )
            if 'final_reasoning' in translated['explanation']:
                translated['explanation']['final_reasoning'] = translator.translate(
                    result['explanation']['final_reasoning'], lang, 'en'
                )
        
        # Translate next steps
        if 'next_steps' in translated and translated['next_steps']:
            translated['next_steps'] = [
                translator.translate(step, lang, 'en') 
                for step in result['next_steps']
            ]
        
        # Translate legal reasoning
        if 'legal_reasoning' in translated:
            translated['legal_reasoning'] = translator.translate(
                result['legal_reasoning'], lang, 'en'
            )
        
        return translated
    except Exception as e:
        # If translation fails, return original
        return result


# OLD: Hybrid system predictor (commented out for reference)
# @st.cache_resource(show_spinner=False)
# def load_predictor_hybrid(force_cpu: bool = True):
#     """
#     Load IntelligentHybridPredictor once and return it.
#     Pass force_cpu=True to avoid GPU initialization (recommended for local machines).
#     """
#     try:
#         # Delay heavy imports until here
#         from hybex_system.hybrid_predictor import IntelligentHybridPredictor
#         from hybex_system.config import HybExConfig
#         from hybex_system.prolog_engine import PrologEngine
#         from hybex_system.knowledge_graph_engine import KnowledgeGraphEngine
#         from hybex_system.neural_models import EnhancedLegalBERT
#     except Exception as e:
#         raise RuntimeError(
#             f"Could not import HybEx-Law modules: {e}\n"
#             "Ensure you run this from the project root and that hybex_system is on PYTHONPATH."
#         )
#
#     cfg = HybExConfig()
#     
#     # Track which components loaded successfully
#     components_status = {
#         "prolog": {"loaded": False, "error": None},
#         "gnn": {"loaded": False, "error": None},
#         "bert": {"loaded": False, "error": None}
#     }
#     
#     # Initialize required components
#     # 1. Initialize Prolog Engine
#     prolog_engine = None
#     try:
#         prolog_engine = PrologEngine(cfg)
#         components_status["prolog"]["loaded"] = True
#     except Exception as e:
#         components_status["prolog"]["error"] = str(e)
#         st.warning(f"⚠️ Prolog engine: {str(e)[:100]}")
#     
#     # 2. Initialize GNN Model
#     gnn_model = None
#     try:
#         gnn_model = KnowledgeGraphEngine(cfg)
#         components_status["gnn"]["loaded"] = True
#     except Exception as e:
#         components_status["gnn"]["error"] = str(e)
#         st.warning(f"⚠️ GNN model: {str(e)[:100]}")
#     
#     # 3. Initialize BERT Model
#     bert_model = None
#     try:
#         bert_model = EnhancedLegalBERT(cfg)
#         components_status["bert"]["loaded"] = True
#     except Exception as e:
#         components_status["bert"]["error"] = str(e)
#         st.warning(f"⚠️ BERT model: {str(e)[:100]}")
#     
#     # Show component status
#     loaded = [k for k, v in components_status.items() if v["loaded"]]
#     if loaded:
#         st.success(f"✅ Loaded: {', '.join(loaded).upper()}")
#     else:
#         st.error("❌ No components loaded successfully. Predictions will use fallback mode.")
#     
#     # Create predictor with all components
#     try:
#         pred = IntelligentHybridPredictor(
#             prolog_engine=prolog_engine,
#             gnn_model=gnn_model,
#             bert_model=bert_model,
#             config=cfg,
#             force_cpu=force_cpu
#         )
#     except TypeError:
#         # older constructor signature — try without force_cpu
#         pred = IntelligentHybridPredictor(
#             prolog_engine=prolog_engine,
#             gnn_model=gnn_model,
#             bert_model=bert_model,
#             config=cfg
#         )
#     
#     # Store component status in predictor for display
#     pred._component_status = components_status
#     return pred


def run_query(predictor, query: str, ask_for_clarification: bool = False):
    """Run predictor on a single-line query and return standard dict output."""
    # NEW: Gemini predictor has simple predict(query) interface
    try:
        # Call Gemini predictor directly
        out = predictor.predict(query)
    except Exception as e:
        # Return a structured error dict for UI
        return {"error": True, "message": f"Prediction failed: {str(e)}", "raw_exception": repr(e)}

    # If predictor asked for clarification, return that directly
    if isinstance(out, dict) and out.get("type") == "clarify":
        return out

    # === ROBUST SERIALIZATION ===
    # If predictor returned dataclass-like object, convert it
    try:
        if hasattr(out, "__dict__") and not isinstance(out, dict):
            out_raw = out.__dict__
        else:
            out_raw = out
    except Exception:
        out_raw = out

    # Ensure we always return a plain dict
    outd = _make_json_safe(out_raw)

    # Build components_loaded for UI clarity (ensure uppercase listing)
    comps = []
    # look for typical patterns in the result
    if outd.get("prolog_result") not in (None, {}, []):
        comps.append("PROLOG")
    if outd.get("gnn_result") not in (None, {}, []):
        comps.append("GNN")
    if outd.get("bert_result") not in (None, {}, []) or (outd.get("calibrated_confidences", {}).get("bert") is not None):
        comps.append("BERT")

    outd["components_loaded"] = comps

    # Ensure per-component sections exist and are JSON-friendly
    # Prolog
    if "prolog_result" in outd:
        outd["prolog_result"] = _make_json_safe(outd["prolog_result"])

    # GNN
    if "gnn_result" in outd:
        outd["gnn_result"] = _make_json_safe(outd["gnn_result"])

    # BERT
    if "bert_result" in outd:
        outd["bert_result"] = _make_json_safe(outd["bert_result"])

    # Calibrated confidences - ensure numeric primitives
    if "calibrated_confidences" in outd:
        cc = outd["calibrated_confidences"]
        outd["calibrated_confidences"] = {k: float(v) if v is not None else None for k, v in cc.items()}

    # Provide a clear decision_rationale string
    if "decision_rationale" in outd and not isinstance(outd["decision_rationale"], str):
        outd["decision_rationale"] = _make_json_safe(outd["decision_rationale"])

    # If there is a LegalReasoning object nested somewhere (e.g. prolog_result.reasoning),
    # ensure it's converted to dict/text
    if "prolog_result" in outd:
        pr = outd["prolog_result"]
        if isinstance(pr, dict):
            # some engines store 'reasoning' as object
            if "reasoning" in pr:
                pr["reasoning"] = _make_json_safe(pr["reasoning"])

    # === ADD NEXT STEPS ===
    # Extract domain, case_type, category for next steps generation
    eligible = outd.get("eligible", False)
    domain = outd.get("domain", "legalaid")
    case_type = outd.get("case_type", "general")
    category = outd.get("category", "general")
    
    # Call predictor's comprehensive next steps generator
    if hasattr(predictor, '_get_domain_specific_next_steps'):
        try:
            next_steps = predictor._get_domain_specific_next_steps(
                eligible=eligible,
                domain=domain,
                case_type=case_type,
                category=category
            )
            outd["next_steps"] = next_steps
        except Exception as e:
            # Fallback to simple next steps
            outd["next_steps"] = [
                "Contact your nearest District Legal Services Authority",
                "Bring all relevant documents",
                "Request a manual review if needed"
            ]
    
    return outd


def main():
    # Load translator
    translator = load_translator() if TRANSLATOR_AVAILABLE else None
    
    # Initialize language in session state
    if 'language' not in st.session_state:
        st.session_state['language'] = 'en'
    
    # Helper function for shorter translation calls
    def t(text: str, lang: str) -> str:
        """Shorthand for translate"""
        return translate(text, lang, translator)
    
    # Language selector in top-right
    col_title, col_lang = st.columns([4, 1])
    
    with col_title:
        lang = st.session_state.get('language', 'en')
        st.title(f"⚖️ {t('HybEx-Law — Legal Aid Eligibility System', lang)}")
        st.write(t("Advanced hybrid AI combining Prolog symbolic reasoning, Graph Neural Networks (GNN), and BERT NLP", lang))
    
    with col_lang:
        if translator:
            # Language dropdown
            lang_options = translator.get_language_options()
            selected_lang_display = st.selectbox(
                "🌐",
                options=list(lang_options.keys()),
                index=0,  # Default: English
                label_visibility="collapsed"
            )
            selected_lang = lang_options[selected_lang_display]
            
            # Store in session state
            if st.session_state['language'] != selected_lang:
                st.session_state['language'] = selected_lang
                st.rerun()

    # Sidebar: Clean, no API key mentions
    with st.sidebar:
        lang = st.session_state.get('language', 'en')
        st.header(translate("📋 System Information", lang, translator))
        
        st.divider()
        
        # All examples in one list (with translations)
        all_examples = [
            # ELIGIBLE Cases
            (t("Low Income - General", lang), 
             t("I am 28 years old earning ₹15,000 per month. Can I get legal aid for a property dispute?", lang),
             t("General | ₹1.8L annual < ₹3L threshold", lang)),
            
            (t("SC Category - Medium Income", lang),
             t("I am a 30-year-old SC category member earning ₹40,000 monthly. Am I eligible for legal aid?", lang),
             t("SC | ₹4.8L annual < ₹8L threshold", lang)),
            
            (t("Senior Citizen", lang),
             t("I am a 65-year-old person with annual income of ₹10 lakhs. Am I eligible?", lang),
             t("Automatic | 65+ years", lang)),
            
            (t("Woman - Low Income", lang),
             t("I am a 35-year-old woman earning ₹20,000 per month. Can I get legal aid?", lang),
             t("Woman | ₹2.4L annual < ₹3L threshold", lang)),
            
            (t("Minor Student", lang),
             t("I am a 16-year-old student. My family income is ₹5 lakhs. Am I eligible?", lang),
             t("Automatic | Minor < 18 years", lang)),
            
            (t("Person with Disability", lang),
             t("I am a 40-year-old person with disability earning ₹50,000 monthly. Am I eligible?", lang),
             t("Automatic | PWD", lang)),
            
            (t("Domestic Violence Victim", lang),
             t("I am a woman facing domestic violence. My husband earns ₹60,000 monthly. Am I eligible?", lang),
             t("Priority | Woman + DV case", lang)),
            
            (t("Tenant Facing Eviction", lang),
             t("I am a tenant earning ₹12,000 monthly facing eviction. Am I eligible?", lang),
             t("Vulnerable tenant | Low income", lang)),
            
            (t("Wrongful Termination", lang),
             t("I was fired from my job without notice. My salary was ₹18,000 monthly. Am I eligible?", lang),
             t("Employment | Wrongful termination", lang)),
            
            (t("Consumer Dispute - Low Income", lang),
             t("I bought a defective phone for ₹15,000. I earn ₹20,000 monthly. Am I eligible?", lang),
             t("Consumer | Low income", lang)),
            
            (t("OBC Category - Medium Income", lang),
             t("I am OBC category earning ₹45,000 monthly. Am I eligible for legal aid?", lang),
             t("OBC | ₹5.4L annual < ₹6L threshold", lang)),
            
            # NOT ELIGIBLE Cases
            (t("High Income - General", lang),
             t("I am 35 years old earning ₹50,000 monthly. Am I eligible for legal aid?", lang),
             t("General | ₹6L annual > ₹3L threshold", lang)),
            
            (t("Very High Income", lang),
             t("I earn ₹2 lakhs per month. Can I get legal aid for a divorce case?", lang),
             t("₹24L annual >> ₹3L threshold", lang)),
            
            (t("Business Owner", lang),
             t("I run a successful business with monthly income of ₹1.5 lakhs. Can I get legal aid?", lang),
             t("Wealth indicator + High income", lang)),
            
            (t("Property Owner", lang),
             t("I own multiple properties and earn ₹4 lakhs annually. Am I eligible?", lang),
             t("Wealth indicator + Multiple assets", lang)),
            
            (t("High-Value Contract Dispute", lang),
             t("I have a contract dispute worth ₹10 lakhs. My income is ₹5 lakhs annually. Am I eligible?", lang),
             t("Contract value too high", lang)),
            
            (t("SC Category - High Income", lang),
             t("I am SC category earning ₹80,000 monthly. Am I eligible?", lang),
             t("SC | ₹9.6L annual > ₹8L threshold", lang)),
            
            (t("Small Consumer Claim", lang),
             t("I bought a defective item worth ₹5,000. My income is ₹2 lakhs. Am I eligible?", lang),
             t("Claim value too small", lang)),
            
            # EDGE Cases
            (t("Borderline Income - General", lang),
             t("I earn exactly ₹3 lakhs per year. Am I eligible for legal aid?", lang),
             t("At threshold boundary", lang)),
            
            (t("Woman - Borderline", lang),
             t("I am a woman earning ₹3.5 lakhs annually. Can I get legal aid for family dispute?", lang),
             t("Woman + Slightly above threshold", lang)),
            
            (t("Joint Family", lang),
             t("I live in a joint family. My individual income is ₹2 lakhs but family income is ₹8 lakhs. Am I eligible?", lang),
             t("Joint family considerations", lang)),
            
            (t("Refugee", lang),
             t("I am a refugee with no income. Am I eligible for legal aid?", lang),
             t("Automatic | Refugee status", lang)),
            
            (t("Medical Negligence", lang),
             t("I suffered permanent injury due to medical negligence. My income is ₹4 lakhs. Am I eligible?", lang),
             t("Serious harm - priority", lang)),
            
            (t("In Police Custody", lang),
             t("I am in police custody with no income. Am I eligible for legal aid?", lang),
             t("Automatic | In custody", lang)),
            
            (t("Flood Victim", lang),
             t("I lost everything in floods. My previous income was ₹5 lakhs. Am I eligible?", lang),
             t("Automatic | Disaster victim", lang)),
            
            (t("Caste Atrocity Victim", lang),
             t("I am SC category and victim of caste atrocity. My income is ₹6 lakhs. Am I eligible?", lang),
             t("SC + Atrocity victim - priority", lang)),
            
            (t("Student - Part-Time", lang),
             t("I am a 17-year-old student earning ₹10,000 monthly from part-time work. Am I eligible?", lang),
             t("Student + Low income", lang)),
            
            (t("Transgender Person", lang),
             t("I am a transgender person with income ₹30,000 monthly. Am I eligible?", lang),
             t("Automatic | Transgender", lang)),
        ]
        
        # Single expander with all examples
        with st.expander(t("📋 Examples", lang), expanded=False):
            st.caption(t("Click any example to auto-fill the query box", lang))
            for title, query, explanation in all_examples:
                if st.button(
                    title,
                    key=f"example_{title}",
                    help=explanation,
                    use_container_width=True
                ):
                    st.session_state['example_query'] = query
                    st.rerun()
        
        st.divider()

        
        # Legal domains covered
        with st.expander(t("🏛️ Legal Domains Covered", lang)):
            domains_list = [
                f"1. **{t('Legal Aid', lang)}** - {t('General eligibility', lang)}",
                f"2. **{t('Family Law', lang)}** - {t('Divorce, custody, maintenance', lang)}",
                f"3. **{t('Consumer Protection', lang)}** - {t('Defective goods, fraud', lang)}",
                f"4. **{t('Employment Law', lang)}** - {t('Termination, wages', lang)}",
                f"5. **{t('Fundamental Rights', lang)}** - {t('Rights violations', lang)}",
                f"6. **{t('Property Law', lang)}** - {t('Eviction, disputes', lang)}",
                f"7. **{t('Criminal Law', lang)}** - {t('FIR, custody, bail', lang)}",
                f"8. **{t('Medical Negligence', lang)}** - {t('Treatment errors', lang)}",
                f"9. **{t('Education Rights', lang)}** - {t('School, admission', lang)}",
                f"10. **{t('Disaster Relief', lang)}** - {t('Calamity victims', lang)}",
                f"11. **{t('Atrocity Cases', lang)}** - {t('Caste/tribal violence', lang)}"
            ]
            st.markdown("\n".join(domains_list))
        
        st.divider()
        st.caption("© 2025 HybEx-Law | Multi-Domain AI")

    # Load predictor silently (no UI feedback)
    predictor = load_predictor()

    # Initialize session state
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""
    if "run_query" not in st.session_state:
        st.session_state["run_query"] = False

    # Check for example queries
    lang = st.session_state.get('language', 'en')
    if 'example_query' in st.session_state:
        query = st.text_area(
            translate("Enter your eligibility query:", lang, translator),
            value=st.session_state['example_query'],
            height=100,
            placeholder=translate("Example: I am a 28-year-old woman earning ₹15,000 per month. Can I get legal aid?", lang, translator)
        )
        # Clear the example after loading
        if st.session_state['example_query']:
            del st.session_state['example_query']
    else:
        query = st.text_area(
            translate("Enter your eligibility query:", lang, translator),
            value=st.session_state.get("current_query", ""),
            height=100,
            placeholder=translate("Example: I am a 28-year-old woman earning ₹15,000 per month. Can I get legal aid?", lang, translator)
        )
    
    analyze = st.button(translate("🔍 Analyze", lang, translator), type="primary", use_container_width=False)
    
    # Update session state
    if query:
        st.session_state["current_query"] = query
    
    # Check if we should run
    should_run = analyze or st.session_state.get("run_query", False)
    if st.session_state.get("run_query", False):
        st.session_state["run_query"] = False  # Reset flag

    if not query:
        st.info("👆 Please enter a query above or click an example in the sidebar.")
        return

    if should_run:
        # Analyze query with hybrid system
        lang = st.session_state.get('language', 'en')
        with st.spinner(translate("� Processing through hybrid system...", lang, translator)):
            # If query is in another language, translate to English for processing
            if lang != 'en' and translator:
                try:
                    query_en = translator.translate(query, 'en', lang)
                except Exception:
                    query_en = query  # Fallback to original
            else:
                query_en = query
            
            result = run_query(predictor, query_en, ask_for_clarification=False)
            
            # Translate result back to selected language
            result = translate_result(result, lang, translator)

        # Handle clarify response (hybrid mode only)
        if isinstance(result, dict) and result.get("type") == "clarify":
            st.warning("The system needs a detail to answer confidently.")
            st.write("Missing fields:", result.get("missing", []))
            if result.get("message"):
                st.write(result["message"])
            return

        # Handle error
        if result.get("error"):
            st.error("Prediction failed:")
            st.write(result.get("message"))
            if st.checkbox("Show raw exception"):
                st.write(result.get("raw_exception"))
            return

        # ---------- BEGIN: Human-friendly output & trace ----------
        def _score_to_label(p: float, pos_label="Likely eligible", neg_label="Likely not eligible", neutral_thresh=0.55):
            """Turn a probability into a short verbal label"""
            if p is None:
                return "No score"
            try:
                p = float(p)
            except Exception:
                return str(p)
            if p >= 0.90:
                return f"Very likely ({p:.2f})"
            if p >= neutral_thresh:
                return f"Likely ({p:.2f})"
            if p >= (1 - neutral_thresh):
                return f"Borderline ({p:.2f})"
            return f"Unlikely ({p:.2f})"

        def build_person_friendly_explanation(res: dict):
            """
            Build a human-friendly explanation structure from predictor output.
            Returns: (headline, detailed_explanation_text, per_component_texts, simplified_trace, next_steps)
            """
            # Basic flags
            eligible = bool(res.get("eligible", False))
            conf = float(res.get("confidence", 0.0))
            method = res.get("method_used") or res.get("method") or "ensemble"
            comps = [c.upper() for c in res.get("components_loaded", [])]
            
            # Special handling for LLM mode
            if method == "llm":
                headline = f"{'✅ ELIGIBLE' if eligible else '❌ NOT ELIGIBLE'} for legal aid (LLM Analysis)"
                detail_lines = [f"Confidence: {conf*100:.1f}%"]
                
                # Add reasoning
                if res.get("reasoning"):
                    detail_lines.append(f"Reasoning: {res['reasoning']}")
                
                # Add legal citations if available
                if res.get("legal_citations"):
                    detail_lines.append(f"Legal basis: {res['legal_citations']}")
                
                # Component explanation (just LLM)
                comp_texts = [("LLM (OpenAI)", res.get("reasoning", "No reasoning provided"), conf)]
                
                # Next steps for LLM mode
                if eligible:
                    next_steps = [
                        "Verify the documents that support your eligibility (BPL card, income proofs, ID).",
                        "Contact your nearest Legal Aid Cell or Legal Services Authority and submit the application.",
                        "Bring copies of the evidence (BPL card, eviction notice, ID proof) to the appointment.",
                        "If urgent (imminent eviction), ask for emergency legal assistance / temporary injunction."
                    ]
                else:
                    # Use alternative options from LLM if available
                    next_steps = [
                        "🚫 You do not qualify for free legal aid under LSA Act 1987, Section 12.",
                        ""
                    ]
                    if res.get("alternative_options"):
                        next_steps.append(res["alternative_options"])
                    else:
                        next_steps.extend([
                            "### Alternative Legal Support Options:",
                            "📋 **Pro Bono Services**: Contact local bar associations",
                            "🏛️ **Legal Aid Clinics**: Visit law school legal clinics",
                            "🤝 **NGOs**: Reach out to specialized NGOs",
                            "💼 **Payment Plans**: Many lawyers offer installment options",
                            "📞 **Legal Helplines**: Call for free consultation"
                        ])
                
                # Simplified trace
                simp_trace = {
                    "method": "llm",
                    "model": res.get("llm_model", "unknown"),
                    "eligible": eligible,
                    "confidence": conf,
                    "extracted_entities": res.get("extracted_entities", {}),
                    "reasoning": res.get("reasoning")
                }
                
                return headline, detail_lines, comp_texts, simp_trace, next_steps, ["LLM"]

            # Per-component confidences (fall back to calibrated_confidences)
            cc = res.get("calibrated_confidences", {})
            bert_p = cc.get("bert") if cc else None
            gnn_p = cc.get("gnn") if cc else None
            prolog_p = cc.get("prolog") if cc else None

            # Per-component textual explanations
            comp_texts = []
            
            # --- [START] SNIPPET 3 of 3 ---
            # PROLOG: deterministic rules + primary_reason + citations
            if res.get("prolog_result"):
                pr = res["prolog_result"]
                primary_reason = "No specific reason found."

                # Check multiple possible structures for the reason
                if isinstance(pr, dict):
                    if "primary_reason" in pr:
                        primary_reason = pr["primary_reason"]
                    elif "reasoning" in pr:
                        primary_reason = pr["reasoning"]
                elif isinstance(pr, str):
                    primary_reason = pr

                # Clean the "Prolog analysis:" prefix if it exists
                if primary_reason.startswith("Prolog analysis:"):
                    primary_reason = primary_reason.replace("Prolog analysis:", "").strip()

                prolog_line = f"Prolog (symbolic rules): {primary_reason}"
                comp_texts.append(("PROLOG", prolog_line, prolog_p))
            # --- [END] SNIPPET 3 of 3 ---
            
            else:
                comp_texts.append(("PROLOG", "No rule-based result available.", None))

            # GNN: give class label and probability and short interpretation
            if res.get("gnn_result"):
                gnn = res["gnn_result"]
                # gnn may present probs or a single score
                probs = None
                if isinstance(gnn, dict):
                    probs = gnn.get("probs") or gnn.get("probabilities") or None
                elif isinstance(gnn, (list, tuple)):
                    probs = gnn
                if probs:
                    try:
                        # assume binary [p_not_eligible, p_eligible] or [p0,p1]
                        p_eligible = float(probs[-1])
                        label = _score_to_label(p_eligible, neutral_thresh=0.55)
                        comp_texts.append(("GNN", f"Graph-based model: {label} (p={p_eligible:.2f})", p_eligible))
                    except Exception:
                        comp_texts.append(("GNN", f"Graph-based model output: {probs}", None))
                else:
                    comp_texts.append(("GNN", "Graph-based model provided no probability.", None))
            else:
                comp_texts.append(("GNN", "GNN not available.", None))

            # BERT: semantic classifier
            if res.get("bert_result") is not None:
                br = res["bert_result"]
                # typical pattern: bert_result may be None or dict {score, logits, label}
                if isinstance(br, dict):
                    bscore = br.get("score") or br.get("prob") or br.get("p") or None
                    if bscore is not None:
                        try:
                            bscore = float(bscore)
                        except:
                            pass
                        comp_texts.append(("BERT", f"Text semantic model: {_score_to_label(bscore)} (p={bscore:.2f})", bscore))
                    else:
                        comp_texts.append(("BERT", f"Text model result provided: { _make_json_safe(br) }", None))
                else:
                    comp_texts.append(("BERT", "Text model returned non-standard result.", None))
            else:
                # If null but we have a calibrated_confidences field use that
                if bert_p is not None:
                    comp_texts.append(("BERT", f"Text semantic model: {_score_to_label(bert_p)} (p={bert_p:.2f})", bert_p))
                else:
                    comp_texts.append(("BERT", "Text model not available.", None))

            # Compose short plain-language headline
            headline = "ELIGIBLE" if eligible else "NOT ELIGIBLE"
            headline += f" — confidence {conf*100:.1f}% (method: {method})"

            # Compose human-friendly detailed explanation:
            # Priority: Prolog deterministic reason + (supporting) GNN/BERT signals
            detail_lines = []
            # show primary deterministic prolog reason if present
            if res.get("prolog_result"):
                # extract primary_reason if present
                pr = res["prolog_result"]
                reasoning = pr.get("reasoning") if isinstance(pr, dict) and "reasoning" in pr else pr
                if isinstance(reasoning, dict):
                    primary = reasoning.get("primary_reason") or reasoning.get("reason") or ""
                    if primary:
                        detail_lines.append(f"Symbolic rule result: {primary}")
            # add per-component summarised evidence
            for name, text, score in comp_texts:
                if text:
                    detail_lines.append(f"{name}: {text}")

            # Explain final fusion:
            fusion_line = ""
            if method.lower().startswith("prolog"):
                fusion_line = "Final decision derived from rule-based (legal) match."
            elif method.lower().startswith("consensus") or "consensus" in (res.get("decision_rationale","") or "").lower() or method.lower().startswith("ensemble"):
                fusion_line = "Final decision by combining rule-based and learned models (ensemble)."
            else:
                fusion_line = f"Final decision via method: {method}."

            detail_lines.append(fusion_line)

            # Next steps: different for eligible / not eligible - Fix 6: Enhanced alternative options
            if eligible:
                next_steps = [
                    "Verify the documents that support your eligibility (BPL card, income proofs, ID).",
                    "Contact your nearest Legal Aid Cell or Legal Services Authority and submit the application.",
                    "Bring copies of the evidence (BPL card, eviction notice, ID proof) to the appointment.",
                    "If urgent (imminent eviction), ask for emergency legal assistance / temporary injunction."
                ]
            else:
                # Extract income info if available for personalized guidance
                extracted_entities = res.get("extracted_entities", {})
                annual_income = extracted_entities.get("annual_income")
                
                next_steps = [
                    "🚫 You do not qualify for free legal aid under LSA Act 1987, Section 12.",
                    ""  # Blank line for readability
                ]
                
                # Income-specific guidance
                if annual_income and annual_income > 300000:
                    next_steps.append(f"💰 Your annual income (₹{annual_income:,.0f}) exceeds the eligibility threshold for your category.")
                
                # Alternative options
                next_steps.extend([
                    "### Alternative Legal Support Options:",
                    "📋 **Pro Bono Services**: Contact local bar associations for lists of lawyers offering reduced-fee or pro bono consultations",
                    "🏛️ **Legal Aid Clinics**: Visit law school legal clinics (e.g., NLU legal aid clinics) that may handle cases outside LSA criteria",
                    "🤝 **NGOs**: Reach out to specialized NGOs working in your legal domain (consumer, employment, family law, etc.)",
                    "💼 **Payment Plans**: Many lawyers offer installment payment options for clients who cannot afford upfront fees",
                    "📞 **Legal Helplines**: Call national/state legal helplines for free initial consultation and referrals",
                    "⚖️ **If you believe this assessment is incorrect**: Gather supporting documents (income certificates, BPL card, category certificate) and request manual review"
                ])

            # Simplified "Common-person" trace (only useful fields)
            simp_trace = {
                "case_id": res.get("case_id"),
                "eligible": eligible,
                "confidence": round(conf, 4),
                "method_used": method,
                "summary_reason": detail_lines[0] if detail_lines else "",
                "evidence": {
                    "prolog": _make_json_safe(res.get("prolog_result")),
                    "gnn_probs": _make_json_safe(res.get("gnn_result")),
                    "bert": _make_json_safe(res.get("bert_result")),
                    "calibrated_confidences": {k: v for k, v in (res.get("calibrated_confidences") or {}).items()}
                },
                "prolog_debug_facts": res.get("prolog_debug_facts")  # B2: Show debug facts
            }

            # Return everything needed for UI
            return headline, detail_lines, comp_texts, simp_trace, next_steps, comps

        # --- Display results with no duplication ---
        
        # Main result card
        lang = st.session_state.get('language', 'en')
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            eligible_flag = result.get("eligible", False)
            if eligible_flag:
                st.success(f"### ✅ {translate('ELIGIBLE for Free Legal Aid', lang, translator)}")
            else:
                st.error(f"### ❌ {translate('NOT ELIGIBLE for Free Legal Aid', lang, translator)}")
        
        with col2:
            st.metric(translate("Confidence", lang, translator), f"{result.get('confidence', 0):.1%}")
        
        with col3:
            method_used = result.get("method_used") or result.get("method", "ensemble")
            st.metric(translate("Method", lang, translator), method_used.replace('_', ' ').title())
        
        # Category and domain info
        if 'category' in result and 'domain' in result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**{translate('Category', lang, translator)}**: {result.get('category', 'General')}")
            with col2:
                st.caption(f"**{translate('Domain', lang, translator)}**: {result.get('domain', 'legalaid').replace('_', ' ').title()}")
            with col3:
                st.caption(f"**{translate('Case Type', lang, translator)}**: {result.get('case_type', 'General')}")
        
        st.divider()
        
        # Why this decision? (SHOW ONLY ONCE)
        with st.expander(f"🔍 **{translate('Why this decision?', lang, translator)}**", expanded=True):
            # Check if we have explanation structure
            if 'explanation' in result and isinstance(result['explanation'], dict):
                exp = result['explanation']
                if 'summary' in exp:
                    st.write(exp['summary'])
                if 'symbolic_result' in exp:
                    st.write(f"**Symbolic Result:** {exp['symbolic_result']}")
                
                if 'components' in exp:
                    st.write("**Component Analysis:**")
                    for comp_name, comp_desc in exp['components'].items():
                        st.write(f"• **{comp_name.upper()}:** {comp_desc}")
                
                if 'final_reasoning' in exp:
                    st.write(f"• {exp['final_reasoning']}")
            else:
                # Fallback to old format
                headline, detail_lines, comp_texts, _, _, _ = build_person_friendly_explanation(result)
                st.markdown(f"**{headline}**")
                for line in detail_lines:
                    st.write("•", line)
        
        # Per-component evidence (SHOW ONLY ONCE)
        if 'component_evidence' in result and result['component_evidence']:
            with st.expander("📊 **Per-component evidence**"):
                cols = st.columns(min(3, len(result['component_evidence'])))
                for idx, comp in enumerate(result['component_evidence']):
                    with cols[idx % len(cols)]:
                        st.metric(
                            label=comp.get('component', 'Unknown'),
                            value=f"{comp.get('confidence', 0):.2f}",
                            help=comp.get('description', '')
                        )
        else:
            # Fallback: extract from calibrated_confidences
            cc = result.get("calibrated_confidences", {})
            if cc:
                with st.expander("� **Per-component evidence**"):
                    cols = st.columns(min(3, len(cc)))
                    for idx, (comp_name, conf) in enumerate(cc.items()):
                        with cols[idx % len(cols)]:
                            st.metric(
                                label=comp_name.upper(),
                                value=f"{conf:.2f}"
                            )
        
        # Legal reasoning
        if 'legal_reasoning' in result or 'applicable_sections' in result or 'eligibility_factors' in result:
            with st.expander("⚖️ **Legal Reasoning**"):
                if 'legal_reasoning' in result:
                    st.write(result['legal_reasoning'])
                
                if 'applicable_sections' in result and result['applicable_sections']:
                    st.write("**Applicable Sections:**")
                    for section in result['applicable_sections']:
                        st.write(f"• {section}")
                
                if 'eligibility_factors' in result and result['eligibility_factors']:
                    st.write("**Key Factors:**")
                    for factor in result['eligibility_factors']:
                        st.write(f"• {factor}")
        
        st.divider()
        
        # Next steps (COMPREHENSIVE - SHOW ONLY ONCE)
        st.markdown(f"## 📋 {translate('Next Steps', lang, translator)}")
        
        # Use next_steps from result if available
        if 'next_steps' in result and result['next_steps']:
            for step in result['next_steps']:
                st.markdown(step)
        else:
            # Fallback to generated next steps
            _, _, _, _, next_steps, _ = build_person_friendly_explanation(result)
            for step in next_steps:
                st.markdown(step)
        
        # Warning for borderline cases
        if result.get('requires_review', False):
            st.warning("⚠️ **Note**: This case is borderline. Consider requesting manual review at your District Legal Services Authority for final determination.")

        # Collapsible simplified trace (human readable)
        with st.expander("Show simplified trace (friendly)", expanded=False):
            _, _, _, simp_trace, _, _ = build_person_friendly_explanation(result)
            st.json(simp_trace)

        # Raw full JSON still available for debugging
        with st.expander("Show full prediction trace (raw JSON)", expanded=False):
            safe_full = _make_json_safe(result)
            st.code(json.dumps(safe_full, indent=2, ensure_ascii=False), language="json")
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            safe_result = _make_json_safe(result)
            result_json = json.dumps(safe_result, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Download JSON", 
                result_json, 
                file_name="hybex_prediction.json", 
                mime="application/json",
                use_container_width=True
            )
        with col_dl2:
            # Copy button with instructions
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Use the code block above to manually copy, or click Download JSON")

        st.markdown("---")
        st.caption("⚠️ This is an automated assessment — not legal advice. Final eligibility determined by authorities.")
        # ---------- END replacement ----------

if __name__ == "__main__":
    main()
