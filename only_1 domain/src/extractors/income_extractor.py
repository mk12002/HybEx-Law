"""
Income Extractor: High-precision extraction of income information

This extractor identifies and normalizes income-related information from legal queries,
including monthly/annual income, employment status, and financial circumstances.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IncomeInfo:
    """Structured representation of extracted income information."""
    monthly_income: Optional[int] = None
    annual_income: Optional[int] = None
    employment_status: Optional[str] = None
    income_source: Optional[str] = None


class IncomeExtractor:
    """
    Extracts income-related facts from natural language text.
    
    Handles various income expressions:
    - "I earn 25000 per month"
    - "My salary is 3 lakhs annually" 
    - "I lost my job and have no income"
    - "Unemployed for 6 months"
    """
    
    def __init__(self):
        """Initialize the income extractor with patterns and mappings."""
        
        # Regex patterns for income extraction
        self.income_patterns = [
            # Monthly income patterns
            (r'(?:earn|salary|income|make|get|paid).*?(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|₹)?\s*(?:per\s*month|monthly|/month)', 'monthly'),
            (r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|₹)?\s*(?:per\s*month|monthly|/month)', 'monthly'),
            
            # Annual income patterns  
            (r'(?:earn|salary|income|make|get|paid).*?(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|₹)?\s*(?:per\s*year|yearly|annually|/year)', 'annual'),
            (r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|₹)?\s*(?:per\s*year|yearly|annually|/year)', 'annual'),
            
            # Lakh/Crore patterns
            (r'(\d+(?:\.\d+)?)\s*lakhs?\s*(?:per\s*year|yearly|annually|/year)?', 'lakh_annual'),
            (r'(\d+(?:\.\d+)?)\s*crores?\s*(?:per\s*year|yearly|annually|/year)?', 'crore_annual'),
            (r'(\d+(?:\.\d+)?)\s*lakhs?\s*(?:per\s*month|monthly|/month)', 'lakh_monthly'),
            
            # General amount patterns (assume monthly if not specified)
            (r'(?:earn|salary|income|make|get|paid).*?(\d+(?:,\d+)*)', 'monthly_default'),
        ]
        
        # Employment status patterns
        self.employment_patterns = [
            (r'(?:lost\s*(?:my\s*)?job|unemployed|jobless|no\s*job)', 'unemployed'),
            (r'(?:fired|terminated|laid\s*off)', 'unemployed'),
            (r'(?:retired|pension)', 'retired'),
            (r'(?:student|studying)', 'student'),
            (r'(?:self\s*employed|business|shop)', 'self_employed'),
            (r'(?:employed|working|job)', 'employed'),
        ]
        
        # No income indicators
        self.no_income_patterns = [
            r'no\s*income',
            r'zero\s*income', 
            r'without\s*income',
            r'lost\s*(?:my\s*)?job',
            r'unemployed',
            r'no\s*salary'
        ]
    
    def extract(self, processed_text: str, original_text: str) -> List[str]:
        """
        Extract income facts from text.
        
        Args:
            processed_text: Preprocessed text
            original_text: Original query text
            
        Returns:
            List of Prolog facts about income
        """
        income_info = self._extract_income_info(original_text.lower())
        facts = []
        
        # Convert to Prolog facts
        if income_info.monthly_income is not None:
            facts.append(f'income_monthly(user, {income_info.monthly_income})')
        
        if income_info.annual_income is not None:
            facts.append(f'income_annual(user, {income_info.annual_income})')
        
        # If no specific income found but employment status indicates no income
        if (income_info.monthly_income is None and 
            income_info.annual_income is None and
            income_info.employment_status == 'unemployed'):
            facts.append('income_monthly(user, 0)')
        
        return facts
    
    def _extract_income_info(self, text: str) -> IncomeInfo:
        """
        Extract structured income information from text.
        
        Args:
            text: Input text (lowercased)
            
        Returns:
            IncomeInfo object with extracted data
        """
        income_info = IncomeInfo()
        
        # Check for no income indicators first
        if any(re.search(pattern, text) for pattern in self.no_income_patterns):
            income_info.monthly_income = 0
            income_info.employment_status = 'unemployed'
            return income_info
        
        # Extract employment status
        for pattern, status in self.employment_patterns:
            if re.search(pattern, text):
                income_info.employment_status = status
                break
        
        # Extract income amounts
        for pattern, income_type in self.income_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1)
                amount = self._parse_amount(amount_str)
                
                if income_type == 'monthly':
                    income_info.monthly_income = amount
                elif income_type == 'annual':
                    income_info.annual_income = amount
                elif income_type == 'lakh_annual':
                    income_info.annual_income = int(float(amount_str) * 100000)
                elif income_type == 'crore_annual':
                    income_info.annual_income = int(float(amount_str) * 10000000)
                elif income_type == 'lakh_monthly':
                    income_info.monthly_income = int(float(amount_str) * 100000)
                elif income_type == 'monthly_default':
                    income_info.monthly_income = amount
                
                break  # Use first match found
        
        # Convert between monthly and annual if only one is available
        if income_info.monthly_income and not income_info.annual_income:
            income_info.annual_income = income_info.monthly_income * 12
        elif income_info.annual_income and not income_info.monthly_income:
            income_info.monthly_income = income_info.annual_income // 12
        
        return income_info
    
    def _parse_amount(self, amount_str: str) -> int:
        """
        Parse amount string to integer.
        
        Args:
            amount_str: String representation of amount
            
        Returns:
            Parsed integer amount
        """
        # Remove commas and convert to int
        cleaned = re.sub(r'[,\s]', '', amount_str)
        try:
            return int(float(cleaned))
        except (ValueError, TypeError):
            return 0
