"""
SLM Fact Extractor - Core fact extraction using fine-tuned Small Language Models.

This module implements the main SLM-based fact extractor that replaces the 
two-stage TF-IDF pipeline with a single, more powerful SLM approach.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model
import re

logger = logging.getLogger(__name__)


class SLMFactExtractor:
    """
    Small Language Model-based fact extractor for legal queries.
    
    This class handles:
    1. Loading and managing fine-tuned SLMs
    2. Converting natural language to Prolog facts
    3. Validation and post-processing of extracted facts
    4. Integration with existing pipeline components
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        model_path: Optional[str] = None,
        device: str = "auto",
        use_quantization: bool = True
    ):
        """
        Initialize the SLM fact extractor.
        
        Args:
            model_name: Base model name from Hugging Face
            model_path: Path to fine-tuned model (if available)
            device: Device to run model on ('auto', 'cuda', 'cpu')
            use_quantization: Whether to use 4-bit quantization for efficiency
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.use_quantization = use_quantization
        
        # Model components (initialized lazily)
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        # Fact validation patterns
        self.fact_patterns = self._compile_fact_patterns()
        
        # System prompt for fact extraction
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"SLM Fact Extractor initialized with model: {model_name}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                logger.warning("CUDA not available, using CPU. This will be slow.")
        
        return torch.device(device)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for fact extraction."""
        return """You are a legal fact extraction specialist for legal aid eligibility determination. 

Your task is to extract structured legal facts from natural language queries and output them in Prolog format.

Extract these facts when present:
- applicant(user).  [Always include this]
- income_monthly(user, Amount).  [Monthly income in rupees]
- case_type(user, 'Type').  [Case category]
- is_woman(user, Boolean).  [true/false]
- is_sc_st(user, Boolean).  [Scheduled Caste/Tribe - true/false]
- is_child(user, Boolean).  [Under 18 - true/false]
- is_disabled(user, Boolean).  [Physical/mental disability - true/false]
- is_senior_citizen(user, Boolean).  [Above 65 - true/false]

Case types include: domestic_violence, property_dispute, family_matter, labor_dispute, criminal_matter, accident_compensation, defamation, business_dispute.

Output only the Prolog facts, one per line. Do not include explanations."""
    
    def _compile_fact_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for fact validation."""
        patterns = {
            'applicant': re.compile(r'applicant\(user\)\.'),
            'income': re.compile(r'income_monthly\(user,\s*(\d+)\)\.'),
            'case_type': re.compile(r"case_type\(user,\s*'([^']+)'\)\."),
            'boolean_fact': re.compile(r'is_(\w+)\(user,\s*(true|false)\)\.')
        }
        return patterns
    
    def load_model(self):
        """Load the tokenizer and model."""
        if self.is_loaded:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config
        quantization_config = None
        if self.use_quantization and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        
        # Load fine-tuned weights if available
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading fine-tuned weights from: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        
        self.is_loaded = True
        logger.info("Model loaded successfully")
    
    def extract_facts(self, query: str, verbose: bool = False) -> List[str]:
        """
        Extract legal facts from a natural language query.
        
        Args:
            query: Natural language legal query
            verbose: Whether to show detailed processing
            
        Returns:
            List of Prolog facts as strings
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Prepare the prompt
        prompt = self._format_prompt(query)
        
        if verbose:
            print(f"🤖 SLM Processing: {query}")
            print(f"📝 Formatted prompt length: {len(prompt)} chars")
        
        # Generate facts
        generated_text = self._generate_response(prompt)
        
        if verbose:
            print(f"🔤 Raw SLM output: {generated_text}")
        
        # Extract and validate facts
        facts = self._parse_and_validate_facts(generated_text)
        
        processing_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Extracted facts: {facts}")
            print(f"⏱️ Processing time: {processing_time:.3f}s")
        
        return facts
    
    def _format_prompt(self, query: str) -> str:
        """Format the input query into a proper prompt."""
        prompt = f"<|system|>\n{self.system_prompt}\n\n<|user|>\nQuery: {query}\n\n<|assistant|>\n"
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from the SLM."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def _parse_and_validate_facts(self, generated_text: str) -> List[str]:
        """Parse and validate facts from generated text."""
        facts = []
        
        # Split into lines and process each
        lines = generated_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Validate against patterns
            if self._is_valid_fact(line):
                facts.append(line)
        
        # Ensure applicant fact is always present
        if not any('applicant(user)' in fact for fact in facts):
            facts.insert(0, 'applicant(user).')
        
        return facts
    
    def _is_valid_fact(self, fact: str) -> bool:
        """Validate if a string is a proper Prolog fact."""
        # Check basic structure
        if not fact.endswith('.'):
            return False
        
        # Check against known patterns
        for pattern in self.fact_patterns.values():
            if pattern.match(fact):
                return True
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'use_quantization': self.use_quantization
        }
        
        if self.is_loaded:
            info.update({
                'model_dtype': str(self.model.dtype) if hasattr(self.model, 'dtype') else 'unknown',
                'vocab_size': self.tokenizer.vocab_size,
                'max_length': getattr(self.tokenizer, 'model_max_length', 'unknown')
            })
        
        return info
    
    def benchmark_performance(self, test_queries: List[str], num_runs: int = 3) -> Dict[str, float]:
        """Benchmark model performance on test queries."""
        if not self.is_loaded:
            self.load_model()
        
        logger.info(f"Benchmarking performance on {len(test_queries)} queries, {num_runs} runs each")
        
        total_time = 0
        successful_extractions = 0
        total_facts_extracted = 0
        
        for query in test_queries:
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    facts = self.extract_facts(query, verbose=False)
                    total_time += time.time() - start_time
                    successful_extractions += 1
                    total_facts_extracted += len(facts)
                except Exception as e:
                    logger.warning(f"Failed to process query: {query[:50]}... Error: {e}")
        
        total_runs = len(test_queries) * num_runs
        avg_time = total_time / total_runs if total_runs > 0 else 0
        success_rate = successful_extractions / total_runs if total_runs > 0 else 0
        avg_facts_per_query = total_facts_extracted / successful_extractions if successful_extractions > 0 else 0
        
        return {
            'avg_processing_time': avg_time,
            'success_rate': success_rate,
            'avg_facts_per_query': avg_facts_per_query,
            'total_runs': total_runs,
            'successful_runs': successful_extractions
        }


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    extractor = SLMFactExtractor()
    
    test_query = "I am a woman facing domestic violence. My husband beats me and demands dowry. I earn 15000 rupees per month."
    
    print("🧪 Testing SLM Fact Extractor")
    print("=" * 50)
    print(f"Query: {test_query}")
    print()
    
    try:
        facts = extractor.extract_facts(test_query, verbose=True)
        print(f"\n✅ Extracted {len(facts)} facts:")
        for fact in facts:
            print(f"  • {fact}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: This requires proper dependencies and model access.")
