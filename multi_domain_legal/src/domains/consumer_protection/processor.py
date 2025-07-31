"""
Consumer Protection Domain Processor.

This module handles consumer protection queries including product defects,
service deficiencies, unfair trade practices, and related consumer rights.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class ConsumerProtectionProcessor:
    """
    Processor for consumer protection domain queries.
    
    Handles consumer complaints, product defects, service deficiencies,
    and other consumer protection matters under various consumer laws.
    """
    
    def __init__(self):
        self.consumer_forums = self._setup_consumer_forums()
        self.product_categories = self._setup_product_categories()
        self.service_categories = self._setup_service_categories()
        self.complaint_types = self._setup_complaint_types()
        self.limitation_periods = self._setup_limitation_periods()
        self.remedies = self._setup_remedies()
    
    def _setup_consumer_forums(self) -> Dict[str, Dict[str, Any]]:
        """Setup consumer forum jurisdiction based on claim value"""
        return {
            'district': {
                'max_value': 2000000,  # 20 lakhs
                'name': 'District Consumer Disputes Redressal Commission',
                'appeal_to': 'state'
            },
            'state': {
                'max_value': 10000000,  # 1 crore
                'name': 'State Consumer Disputes Redressal Commission',
                'appeal_to': 'national'
            },
            'national': {
                'max_value': float('inf'),  # No upper limit
                'name': 'National Consumer Disputes Redressal Commission',
                'appeal_to': 'supreme_court'
            }
        }
    
    def _setup_product_categories(self) -> Dict[str, List[str]]:
        """Setup product categories with keywords"""
        return {
            'electronics': [
                'mobile', 'phone', 'smartphone', 'laptop', 'computer', 'tv', 'television',
                'refrigerator', 'washing machine', 'ac', 'air conditioner', 'microwave'
            ],
            'automobiles': [
                'car', 'bike', 'motorcycle', 'scooter', 'vehicle', 'auto', 'truck'
            ],
            'home_appliances': [
                'fridge', 'cooler', 'heater', 'fan', 'mixer', 'grinder', 'iron'
            ],
            'clothing': [
                'clothes', 'shirt', 'dress', 'shoes', 'garment', 'fabric', 'textile'
            ],
            'food_products': [
                'food', 'grocery', 'packaged food', 'dairy', 'oil', 'spices', 'snacks'
            ],
            'medicines': [
                'medicine', 'drugs', 'pharmaceutical', 'tablet', 'syrup', 'injection'
            ],
            'cosmetics': [
                'cosmetics', 'beauty products', 'cream', 'shampoo', 'soap', 'perfume'
            ]
        }
    
    def _setup_service_categories(self) -> Dict[str, List[str]]:
        """Setup service categories with keywords"""
        return {
            'real_estate': [
                'builder', 'apartment', 'flat', 'house', 'property', 'construction',
                'possession', 'rera', 'real estate'
            ],
            'banking': [
                'bank', 'loan', 'credit card', 'atm', 'banking', 'finance', 'emi'
            ],
            'insurance': [
                'insurance', 'policy', 'claim', 'premium', 'life insurance', 'health insurance'
            ],
            'telecom': [
                'mobile service', 'internet', 'broadband', 'sim card', 'network', 'telecom'
            ],
            'airlines': [
                'flight', 'airline', 'airport', 'booking', 'travel', 'aviation'
            ],
            'hospitality': [
                'hotel', 'restaurant', 'resort', 'food service', 'catering', 'hospitality'
            ],
            'e_commerce': [
                'online', 'e-commerce', 'shopping', 'delivery', 'marketplace', 'website'
            ],
            'healthcare': [
                'hospital', 'doctor', 'medical', 'treatment', 'healthcare', 'clinic'
            ],
            'education': [
                'school', 'college', 'university', 'coaching', 'education', 'training'
            ]
        }
    
    def _setup_complaint_types(self) -> Dict[str, List[str]]:
        """Setup complaint types with indicators"""
        return {
            'defective_product': [
                'defective', 'faulty', 'broken', 'not working', 'damaged', 'manufacturing defect'
            ],
            'service_deficiency': [
                'poor service', 'bad service', 'service deficiency', 'unsatisfactory service',
                'delay in service', 'incomplete service'
            ],
            'unfair_trade_practice': [
                'fraud', 'cheating', 'misleading', 'false advertisement', 'overcharging',
                'unfair practice', 'deceptive practice'
            ],
            'warranty_issue': [
                'warranty', 'guarantee', 'warranty claim', 'warranty period', 'free service'
            ],
            'billing_dispute': [
                'wrong bill', 'overcharged', 'billing error', 'excess amount', 'duplicate charge'
            ],
            'delivery_issue': [
                'not delivered', 'delivery delay', 'wrong delivery', 'damaged delivery',
                'partial delivery'
            ]
        }
    
    def _setup_limitation_periods(self) -> Dict[str, int]:
        """Setup limitation periods for filing complaints (in years)"""
        return {
            'product_defect': 2,
            'service_deficiency': 2,
            'unfair_trade_practice': 2,
            'medical_negligence': 2,
            'real_estate': 3,  # Special cases may have longer periods
            'insurance_claim': 3
        }
    
    def _setup_remedies(self) -> Dict[str, List[str]]:
        """Setup available remedies for different complaint types"""
        return {
            'defective_product': [
                'replacement', 'refund', 'repair', 'compensation for damages'
            ],
            'service_deficiency': [
                'service completion', 'compensation', 'refund', 'penalty'
            ],
            'unfair_trade_practice': [
                'refund', 'compensation', 'punitive damages', 'corrective advertisement'
            ],
            'delay': [
                'compensation for delay', 'interest on delayed delivery', 'alternative arrangement'
            ]
        }
    
    def identify_product_category(self, query: str) -> Optional[str]:
        """
        Identify product category from query.
        
        Args:
            query: Consumer query text
            
        Returns:
            Identified product category or None
        """
        query_lower = query.lower()
        
        for category, keywords in self.product_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return None
    
    def identify_service_category(self, query: str) -> Optional[str]:
        """
        Identify service category from query.
        
        Args:
            query: Consumer query text
            
        Returns:
            Identified service category or None
        """
        query_lower = query.lower()
        
        for category, keywords in self.service_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return None
    
    def extract_transaction_details(self, query: str) -> Dict[str, Any]:
        """
        Extract transaction details from query.
        
        Args:
            query: Consumer query text
            
        Returns:
            Dictionary with transaction information
        """
        transaction_info = {
            'amount': None,
            'transaction_date': None,
            'purchase_location': None,
            'payment_method': None,
            'invoice_available': False
        }
        
        query_lower = query.lower()
        
        # Extract transaction amount
        amount_patterns = [
            r'(?:paid|cost|price|amount|worth|value).*?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(?:rs\.?|rupees?|inr)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs\.?|rupees?|inr)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|lakhs)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|crores)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = float(match.group(1).replace(',', ''))
                # Handle lakh/crore multipliers
                if 'lakh' in pattern:
                    amount *= 100000
                elif 'crore' in pattern:
                    amount *= 10000000
                transaction_info['amount'] = int(amount)
                break
        
        # Extract time information
        time_patterns = [
            r'(\d+)\s*(?:months?|mon)\s*(?:ago|back)',
            r'(\d+)\s*(?:years?|yr)\s*(?:ago|back)',
            r'(\d+)\s*(?:days?)\s*(?:ago|back)',
            r'last\s+(\d+)\s*(?:months?|years?|days?)',
            r'(\d+)[-/](\d+)[-/](\d+)',  # Date format
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'month' in pattern:
                    months_ago = int(match.group(1))
                    transaction_info['transaction_date'] = f"{months_ago} months ago"
                elif 'year' in pattern:
                    years_ago = int(match.group(1))
                    transaction_info['transaction_date'] = f"{years_ago} years ago"
                elif 'day' in pattern:
                    days_ago = int(match.group(1))
                    transaction_info['transaction_date'] = f"{days_ago} days ago"
                break
        
        # Check for invoice/receipt
        if any(term in query_lower for term in ['bill', 'invoice', 'receipt', 'voucher']):
            transaction_info['invoice_available'] = True
        
        # Identify purchase location
        if any(term in query_lower for term in ['online', 'website', 'internet']):
            transaction_info['purchase_location'] = 'online'
        elif any(term in query_lower for term in ['shop', 'store', 'showroom', 'market']):
            transaction_info['purchase_location'] = 'physical_store'
        
        return transaction_info
    
    def identify_complaint_type(self, query: str) -> List[str]:
        """
        Identify types of consumer complaints from query.
        
        Args:
            query: Consumer query text
            
        Returns:
            List of applicable complaint types
        """
        complaint_types = []
        query_lower = query.lower()
        
        for complaint_type, indicators in self.complaint_types.items():
            if any(indicator in query_lower for indicator in indicators):
                complaint_types.append(complaint_type)
        
        return complaint_types
    
    def determine_forum_jurisdiction(self, amount: Optional[int]) -> Dict[str, Any]:
        """
        Determine appropriate consumer forum based on claim amount.
        
        Args:
            amount: Transaction/claim amount
            
        Returns:
            Dictionary with forum information
        """
        if not amount:
            return {
                'forum': 'district',
                'reason': 'amount_not_specified',
                'name': self.consumer_forums['district']['name']
            }
        
        for forum_type, forum_info in self.consumer_forums.items():
            if amount <= forum_info['max_value']:
                return {
                    'forum': forum_type,
                    'reason': f'amount_{amount}_within_{forum_type}_jurisdiction',
                    'name': forum_info['name'],
                    'max_value': forum_info['max_value']
                }
        
        # Default to national forum
        return {
            'forum': 'national',
            'reason': 'amount_exceeds_state_jurisdiction',
            'name': self.consumer_forums['national']['name']
        }
    
    def check_limitation_period(self, transaction_date: Optional[str], complaint_type: str) -> Dict[str, Any]:
        """
        Check if complaint is within limitation period.
        
        Args:
            transaction_date: When transaction occurred
            complaint_type: Type of complaint
            
        Returns:
            Dictionary with limitation analysis
        """
        limitation_info = {
            'within_limitation': None,
            'limitation_period': self.limitation_periods.get(complaint_type, 2),
            'time_left': None,
            'urgent': False
        }
        
        if not transaction_date:
            limitation_info['within_limitation'] = 'unknown'
            return limitation_info
        
        # Parse time information
        if 'months ago' in transaction_date:
            months_ago = int(re.search(r'(\d+)', transaction_date).group(1))
            years_passed = months_ago / 12
        elif 'years ago' in transaction_date:
            years_passed = int(re.search(r'(\d+)', transaction_date).group(1))
        elif 'days ago' in transaction_date:
            days_ago = int(re.search(r'(\d+)', transaction_date).group(1))
            years_passed = days_ago / 365
        else:
            limitation_info['within_limitation'] = 'unknown'
            return limitation_info
        
        limitation_years = limitation_info['limitation_period']
        
        if years_passed < limitation_years:
            limitation_info['within_limitation'] = True
            limitation_info['time_left'] = f"{limitation_years - years_passed:.1f} years"
            if years_passed > (limitation_years * 0.8):  # 80% of limitation period passed
                limitation_info['urgent'] = True
        else:
            limitation_info['within_limitation'] = False
        
        return limitation_info
    
    def extract_facts(self, query: str) -> List[str]:
        """
        Extract legal facts from consumer protection query.
        
        Args:
            query: Consumer query text
            
        Returns:
            List of Prolog facts
        """
        facts = ['consumer(user).']
        
        # Extract product/service information
        product_category = self.identify_product_category(query)
        if product_category:
            facts.append(f"product_category(user, '{product_category}').")
        
        service_category = self.identify_service_category(query)
        if service_category:
            facts.append(f"service_category(user, '{service_category}').")
        
        # Extract transaction details
        transaction_info = self.extract_transaction_details(query)
        if transaction_info['amount']:
            facts.append(f"transaction_amount(user, {transaction_info['amount']}).")
        if transaction_info['transaction_date']:
            facts.append(f"transaction_date(user, '{transaction_info['transaction_date']}').")
        if transaction_info['purchase_location']:
            facts.append(f"purchase_location(user, '{transaction_info['purchase_location']}').")
        if transaction_info['invoice_available']:
            facts.append('invoice_available(user, true).')
        
        # Extract complaint types
        complaint_types = self.identify_complaint_type(query)
        for complaint_type in complaint_types:
            facts.append(f"complaint_type(user, '{complaint_type}').")
        
        return facts
    
    def analyze_legal_position(self, facts: List[str]) -> Dict[str, Any]:
        """
        Analyze legal position for consumer protection matters.
        
        Args:
            facts: List of extracted legal facts
            
        Returns:
            Dictionary with legal analysis
        """
        analysis = {
            'can_file_consumer_complaint': False,
            'appropriate_forum': 'district',
            'within_limitation_period': None,
            'available_remedies': [],
            'estimated_compensation': None,
            'required_documents': [],
            'recommended_actions': [],
            'applicable_acts': []
        }
        
        facts_str = ' '.join(facts)
        
        # Check if consumer complaint can be filed
        if any(complaint in facts_str for complaint in ['complaint_type(', 'product_category(', 'service_category(']):
            analysis['can_file_consumer_complaint'] = True
        
        # Determine appropriate forum
        amount_match = re.search(r'transaction_amount\(user, (\d+)\)', facts_str)
        if amount_match:
            amount = int(amount_match.group(1))
            forum_info = self.determine_forum_jurisdiction(amount)
            analysis['appropriate_forum'] = forum_info['forum']
            analysis['forum_name'] = forum_info['name']
            
            # Estimate compensation (typically amount + interest + costs)
            analysis['estimated_compensation'] = amount + (amount * 0.12)  # 12% interest approximately
        
        # Check limitation period
        date_match = re.search(r"transaction_date\(user, '([^']+)'\)", facts_str)
        complaint_type_match = re.search(r"complaint_type\(user, '([^']+)'\)", facts_str)
        
        if date_match and complaint_type_match:
            transaction_date = date_match.group(1)
            complaint_type = complaint_type_match.group(1)
            limitation_info = self.check_limitation_period(transaction_date, complaint_type)
            analysis['within_limitation_period'] = limitation_info['within_limitation']
            if limitation_info['urgent']:
                analysis['recommended_actions'].append('File complaint urgently - nearing limitation period')
        
        # Determine available remedies
        complaint_types = re.findall(r"complaint_type\(user, '([^']+)'\)", facts_str)
        for complaint_type in complaint_types:
            if complaint_type in self.remedies:
                analysis['available_remedies'].extend(self.remedies[complaint_type])
        
        # Required documents
        analysis['required_documents'] = [
            'Purchase receipt/invoice',
            'Identity proof',
            'Address proof',
            'Evidence of defect/deficiency'
        ]
        
        if 'service_category(user, \'real_estate\')' in facts_str:
            analysis['required_documents'].extend([
                'Sale agreement',
                'RERA registration certificate',
                'Possession timeline documents'
            ])
        
        # Recommended actions
        if analysis['can_file_consumer_complaint']:
            analysis['recommended_actions'].extend([
                'File complaint in appropriate consumer forum',
                'Gather all supporting documents',
                'Calculate total loss/damages',
                'Consider pre-litigation notice to opposite party'
            ])
        
        # Applicable acts
        analysis['applicable_acts'] = ['Consumer Protection Act, 2019']
        
        if 'service_category(user, \'real_estate\')' in facts_str:
            analysis['applicable_acts'].append('Real Estate (Regulation and Development) Act, 2016')
        
        if 'product_category(user, \'food_products\')' in facts_str:
            analysis['applicable_acts'].append('Food Safety and Standards Act, 2006')
        
        if 'purchase_location(user, \'online\')' in facts_str:
            analysis['applicable_acts'].append('Information Technology Act, 2000')
        
        return analysis
