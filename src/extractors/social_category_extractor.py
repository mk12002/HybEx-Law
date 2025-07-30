"""
Social Category Extractor: Identifies categorical eligibility criteria

This extractor identifies social categories that may qualify for legal aid
regardless of income, such as women, children, SC/ST members, disabled persons, etc.
"""

import re
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class SocialCategoryInfo:
    """Structured representation of social category information."""
    is_woman: bool = False
    is_child: bool = False
    is_sc_st: bool = False
    is_disabled: bool = False
    is_industrial_worker: bool = False
    is_in_custody: bool = False
    is_disaster_victim: bool = False
    general_category: bool = False


class SocialCategoryExtractor:
    """
    Extracts social category information that affects legal aid eligibility.
    
    Key categories:
    - Women (always eligible regardless of income)
    - Children/Minors
    - SC/ST members
    - Disabled persons
    - Industrial workers
    - Persons in custody
    - Disaster victims
    """
    
    def __init__(self):
        """Initialize the social category extractor with patterns."""
        
        # Category detection patterns
        self.category_patterns = {
            'woman': [
                r'\b(?:woman|female|wife|mother|lady|girl)\b',
                r'\bi\s+am\s+a\s+woman\b',
                r'\bwomen\b'
            ],
            'child': [
                r'\b(?:child|minor|kid|boy|girl)\b',
                r'\b(?:under|below)\s+18\b',
                r'\byoung\b',
                r'\bstudent\b'  # Often indicates younger person
            ],
            'sc_st': [
                r'\b(?:sc|st)\b',
                r'\bscheduled\s+caste\b',
                r'\bscheduled\s+tribe\b',
                r'\bdalit\b',
                r'\btribal\b'
            ],
            'general_category': [
                r'\bgeneral\s+category\b',
                r'\bgeneral\s+caste\b',
                r'\bupper\s+caste\b'
            ],
            'obc': [
                r'\bobc\b',
                r'\bother\s+backward\s+class\b'
            ],
            'disabled': [
                r'\b(?:disabled|handicapped|blind|deaf|mute)\b',
                r'\bphysical\s+disability\b',
                r'\bmental\s+disability\b',
                r'\bspecial\s+needs\b'
            ],
            'industrial_worker': [
                r'\bindustrial\s+worker\b',
                r'\bfactory\s+worker\b',
                r'\blabor\b',
                r'\bworker\b.*\bfactory\b',
                r'\bmill\s+worker\b'
            ],
            'in_custody': [
                r'\bin\s+(?:jail|prison|custody)\b',
                r'\b(?:arrested|detained)\b',
                r'\bpolice\s+custody\b',
                r'\bbehind\s+bars\b'
            ],
            'disaster_victim': [
                r'\bdisaster\s+victim\b',
                r'\b(?:flood|earthquake|cyclone|fire)\s+victim\b',
                r'\bnatural\s+disaster\b',
                r'\bdisaster\s+affected\b'
            ]
        }
        
        # Age-related patterns for child detection
        self.age_patterns = [
            r'\b(\d+)\s+years?\s+old\b',
            r'\bage\s+(\d+)\b',
            r'\b(\d+)\s+yr\b'
        ]
    
    def extract(self, processed_text: str, original_text: str) -> List[str]:
        """
        Extract social category facts from text.
        
        Args:
            processed_text: Preprocessed text
            original_text: Original query text
            
        Returns:
            List of Prolog facts about social categories
        """
        category_info = self._extract_category_info(original_text.lower())
        facts = []
        
        # Convert to Prolog facts
        if category_info.is_woman:
            facts.append('is_woman(user, true)')
        else:
            facts.append('is_woman(user, false)')
        
        if category_info.is_child:
            facts.append('is_child(user, true)')
        else:
            facts.append('is_child(user, false)')
        
        if category_info.is_sc_st:
            facts.append('is_sc_st(user, true)')
        else:
            facts.append('is_sc_st(user, false)')
        
        if category_info.is_disabled:
            facts.append('is_disabled(user, true)')
        else:
            facts.append('is_disabled(user, false)')
        
        if category_info.is_industrial_worker:
            facts.append('is_industrial_workman(user, true)')
        else:
            facts.append('is_industrial_workman(user, false)')
        
        if category_info.is_in_custody:
            facts.append('is_in_custody(user, true)')
        else:
            facts.append('is_in_custody(user, false)')
        
        if category_info.is_disaster_victim:
            facts.append('is_disaster_victim(user, true)')
        else:
            facts.append('is_disaster_victim(user, false)')
        
        return facts
    
    def _extract_category_info(self, text: str) -> SocialCategoryInfo:
        """
        Extract social category information from text.
        
        Args:
            text: Input text (lowercased)
            
        Returns:
            SocialCategoryInfo object with extracted information
        """
        info = SocialCategoryInfo()
        
        # Check each category
        info.is_woman = self._check_category(text, 'woman')
        info.is_child = self._check_category(text, 'child') or self._check_age(text)
        info.is_sc_st = self._check_category(text, 'sc_st')
        info.is_disabled = self._check_category(text, 'disabled')
        info.is_industrial_worker = self._check_category(text, 'industrial_worker')
        info.is_in_custody = self._check_category(text, 'in_custody')
        info.is_disaster_victim = self._check_category(text, 'disaster_victim')
        info.general_category = self._check_category(text, 'general_category')
        
        return info
    
    def _check_category(self, text: str, category: str) -> bool:
        """
        Check if a specific category is mentioned in text.
        
        Args:
            text: Input text
            category: Category to check for
            
        Returns:
            True if category is detected
        """
        if category not in self.category_patterns:
            return False
        
        patterns = self.category_patterns[category]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _check_age(self, text: str) -> bool:
        """
        Check if age indicates a minor (under 18).
        
        Args:
            text: Input text
            
        Returns:
            True if age indicates a minor
        """
        for pattern in self.age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    if age < 18:
                        return True
                except (ValueError, IndexError):
                    continue
        
        return False
    
    def get_detected_categories(self, text: str) -> Dict[str, bool]:
        """
        Get all detected categories for debugging.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping category names to detection status
        """
        info = self._extract_category_info(text.lower())
        
        return {
            'woman': info.is_woman,
            'child': info.is_child,
            'sc_st': info.is_sc_st,
            'disabled': info.is_disabled,
            'industrial_worker': info.is_industrial_worker,
            'in_custody': info.is_in_custody,
            'disaster_victim': info.is_disaster_victim,
            'general_category': info.general_category
        }
