"""
Azure-powered multilingual translation for HybEx-Law
Uses environment variables for secure credential management
"""

import os
from typing import Dict, Optional
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

class MultilingualTranslator:
    """
    Azure Translator integration for 6 Indian languages + English
    """
    
    # Load credentials from environment variables
    AZURE_API_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
    AZURE_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "centralindia")
    AZURE_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
    
    # Supported languages with native scripts
    LANGUAGES = {
        'en': {'name': 'English', 'native': 'English', 'flag': '🇬🇧'},
        'hi': {'name': 'Hindi', 'native': 'हिंदी', 'flag': '🇮🇳'},
        'bn': {'name': 'Bengali', 'native': 'বাংলা', 'flag': '🇮🇳'},
        'te': {'name': 'Telugu', 'native': 'తెలుగు', 'flag': '🇮🇳'},
        'mr': {'name': 'Marathi', 'native': 'मराठी', 'flag': '🇮🇳'},
        'ta': {'name': 'Tamil', 'native': 'தமிழ்', 'flag': '🇮🇳'},
        'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી', 'flag': '🇮🇳'}
    }
    
    def __init__(self):
        """Initialize Azure Translator with environment variables"""
        
        # Check if API key is set
        if not self.AZURE_API_KEY:
            print("⚠️ Azure API key not configured - Using English only mode")
            print("👉 Set AZURE_TRANSLATOR_KEY in your .env file")
            self.client = None
        else:
            try:
                # Initialize Azure client
                self.client = TextTranslationClient(
                    credential=AzureKeyCredential(self.AZURE_API_KEY),
                    endpoint=self.AZURE_ENDPOINT,
                    region=self.AZURE_REGION
                )
                print("✅ Azure Translator initialized successfully")
            except Exception as e:
                print(f"❌ Azure Translator init failed: {e}")
                self.client = None
    
    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code (hi, bn, te, mr, ta, gu)
            source_language: Source language code (default: en)
            
        Returns:
            Translated text (or original if translation fails)
        """
        # No translation needed
        if target_language == source_language or target_language == 'en':
            return text
        
        # No API client - return original
        if not self.client:
            return text
        
        # Empty or very short text - skip translation
        if not text or len(text.strip()) < 2:
            return text
        
        try:
            # Call Azure Translator API
            response = self.client.translate(
                body=[{"text": text}],
                to_language=[target_language],
                from_language=source_language
            )
            
            # Extract translated text
            if response and len(response) > 0 and response[0].translations:
                translation = response[0].translations[0]
                return translation.text
            
            return text
        
        except HttpResponseError as e:
            print(f"⚠️ Translation API error: {e.status_code} - {e.message}")
            return text
        except Exception as e:
            print(f"⚠️ Unexpected translation error: {e}")
            return text
    
    def translate_batch(self, texts: list, target_language: str, source_language: str = 'en') -> list:
        """
        Translate multiple texts in one API call (more efficient)
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            List of translated texts
        """
        if target_language == source_language or target_language == 'en' or not self.client:
            return texts
        
        try:
            # Prepare batch request
            body = [{"text": text} for text in texts]
            
            # Call API
            response = self.client.translate(
                body=body,
                to_language=[target_language],
                from_language=source_language
            )
            
            # Extract translations
            translated = []
            for i, item in enumerate(response):
                if item.translations:
                    translated.append(item.translations[0].text)
                else:
                    translated.append(texts[i])  # Fallback to original
            
            return translated
        
        except Exception as e:
            print(f"⚠️ Batch translation error: {e}")
            return texts
    
    def translate_dict(self, texts: Dict[str, str], target_language: str) -> Dict[str, str]:
        """
        Translate dictionary of texts
        
        Args:
            texts: Dictionary of {key: text} to translate
            target_language: Target language code
            
        Returns:
            Dictionary of {key: translated_text}
        """
        if target_language == 'en' or not self.client:
            return texts
        
        # Extract keys and values
        keys = list(texts.keys())
        values = list(texts.values())
        
        # Translate values in batch
        translated_values = self.translate_batch(values, target_language)
        
        # Reconstruct dictionary
        return dict(zip(keys, translated_values))
    
    def is_available(self) -> bool:
        """Check if Azure translator is available"""
        return self.client is not None
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get native language name"""
        return MultilingualTranslator.LANGUAGES.get(lang_code, {}).get('native', 'English')
    
    @staticmethod
    def get_language_options() -> Dict[str, str]:
        """
        Get language dropdown options
        
        Returns:
            Dictionary of {display_name: language_code}
        """
        return {
            f"{lang['flag']} {lang['native']} ({lang['name']})": code 
            for code, lang in MultilingualTranslator.LANGUAGES.items()
        }
    
    @staticmethod
    def get_supported_languages() -> list:
        """Get list of supported language codes"""
        return list(MultilingualTranslator.LANGUAGES.keys())


# Singleton instance for easy import
_translator_instance = None

def get_translator() -> MultilingualTranslator:
    """Get or create translator instance"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = MultilingualTranslator()
    return _translator_instance
