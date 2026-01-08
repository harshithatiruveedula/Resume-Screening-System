"""
Vectorizer Module

This module handles TF-IDF vectorization of text documents.
Uses scikit-learn's TfidfVectorizer to convert text to numerical features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import numpy as np


class ResumeVectorizer:
    """
    TF-IDF Vectorizer for resume and job description text.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to keep (default: 5000)
            ngram_range: Range of n-grams to extract (default: (1, 2) for unigrams and bigrams)
        """
        # FIXED: Use correct TF-IDF configuration
        # lowercase=True: Handles case normalization
        # stop_words='english': Removes common English stopwords
        # min_df=1: Include terms that appear in at least 1 document (very permissive)
        # max_df=0.95: Exclude terms that appear in >95% of documents (very common words)
        # These parameters ensure we capture relevant terms while filtering noise
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,  # Minimum document frequency (1 = include all terms)
            max_df=0.95,  # Maximum document frequency (exclude very common words)
            stop_words='english',  # Remove English stopwords
            lowercase=True,  # Convert to lowercase (CRITICAL for matching)
            sublinear_tf=True  # Apply sublinear TF scaling (log scaling) for better results
        )
        self.is_fitted = False
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the vectorizer on documents and transform them to vectors.
        
        CRITICAL: This method must preserve document order and handle all documents,
        even if some are empty (empty documents will get zero vectors).
        
        Args:
            documents: List of raw text documents (TF-IDF handles preprocessing internally)
            
        Returns:
            TF-IDF matrix as numpy array (n_documents x n_features)
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # CRITICAL FIX: Do NOT filter documents - preserve order and indices
        # Empty documents will get zero vectors, which is correct behavior
        # Filtering would break the index mapping between documents and vectors
        
        # Ensure all documents are strings (handle None or empty)
        processed_docs = [str(doc) if doc else "" for doc in documents]
        
        # Fit and transform - this preserves document order
        vectors = self.vectorizer.fit_transform(processed_docs)
        self.is_fitted = True
        
        # Convert to dense array (n_documents x n_features)
        return vectors.toarray()
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to vectors using the fitted vectorizer.
        
        Args:
            documents: List of preprocessed text documents
            
        Returns:
            TF-IDF matrix as numpy array (n_documents x n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        vectors = self.vectorizer.transform(documents)
        return vectors.toarray()
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary) from the vectorizer.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_keywords(self, document_vector: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a document vector.
        
        Args:
            document_vector: Single document vector (1D array)
            top_n: Number of top keywords to return (default: 10)
            
        Returns:
            List of tuples (keyword, tfidf_score) sorted by score
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before extracting keywords")
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Get indices of top features
        top_indices = document_vector.argsort()[-top_n:][::-1]
        
        # Extract keywords and scores
        keywords = [(feature_names[idx], float(document_vector[idx])) 
                   for idx in top_indices if document_vector[idx] > 0]
        
        return keywords

