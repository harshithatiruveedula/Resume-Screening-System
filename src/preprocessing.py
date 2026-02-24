"""
Text Preprocessing Module

This module handles NLP preprocessing of text:
- Lowercasing
- Remove punctuation & numbers
- Remove stopwords
- Lemmatization using spaCy
"""

import re
import spacy
from typing import List, Optional
import string


# Load spaCy model (will be initialized on first use)
_nlp = None


def get_spacy_model() -> spacy.Language:
    """
    Get or initialize spaCy model for lemmatization.
    
    Returns:
        Loaded spaCy model
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            raise OSError(
                "spaCy English model not found. Please install it using: "
                "python -m spacy download en_core_web_sm"
            )
    return _nlp


def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    
    Args:
        text: Input text string
        
    Returns:
        Lowercased text
    """
    return text.lower()


def remove_punctuation_and_numbers(text: str) -> str:
    """
    Remove punctuation and numbers from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with punctuation and numbers removed
    """
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text using spaCy.
    
    Args:
        text: Input text string
        
    Returns:
        Text with stopwords removed
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    # Filter out stopwords
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using spaCy.
    
    Args:
        text: Input text string
        
    Returns:
        Lemmatized text
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    # Extract lemmatized tokens
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)


def preprocess_text(text: str, remove_stopwords_flag: bool = True) -> str:
    """
    Complete preprocessing pipeline for text.
    
    Args:
        text: Raw input text
        remove_stopwords_flag: Whether to remove stopwords (default: True)
        
    Returns:
        Preprocessed text ready for vectorization
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Lowercase
    text = lowercase_text(text)
    
    # Step 2: Remove punctuation and numbers
    text = remove_punctuation_and_numbers(text)
    
    # Step 3: Remove stopwords (optional)
    if remove_stopwords_flag:
        text = remove_stopwords(text)
    
    # Step 4: Lemmatization
    text = lemmatize_text(text)
    
    # Final cleanup: remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_batch(texts: List[str], remove_stopwords_flag: bool = True) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of raw text strings
        remove_stopwords_flag: Whether to remove stopwords (default: True)
        
    Returns:
        List of preprocessed texts
    """
    return [preprocess_text(text, remove_stopwords_flag) for text in texts]





