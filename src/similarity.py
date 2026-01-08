"""
Similarity Calculation Module

This module handles similarity calculation between resumes and job descriptions
using cosine similarity.
"""

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import numpy as np


def calculate_cosine_similarity(resume_vectors: np.ndarray, 
                                job_vector: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between resume vectors and job description vector.
    
    This function computes cosine similarity which measures the cosine of the angle
    between two vectors. For TF-IDF vectors (which are non-negative), this returns
    values in the range [0, 1], where 1 means identical and 0 means orthogonal.
    
    CRITICAL: Both input arrays must have the same number of features (columns).
    
    Args:
        resume_vectors: Array of resume vectors (n_resumes x n_features)
        job_vector: Job description vector (1 x n_features) or (n_features,)
        
    Returns:
        Array of similarity scores (n_resumes,) in range [0, 1]
    """
    # Ensure job_vector is 2D array (1 x n_features)
    if job_vector.ndim == 1:
        job_vector = job_vector.reshape(1, -1)
    elif job_vector.ndim == 2 and job_vector.shape[0] != 1:
        # If it's 2D but not (1 x n_features), take first row
        job_vector = job_vector[0:1]
    
    # Verify dimensions match
    if resume_vectors.shape[1] != job_vector.shape[1]:
        raise ValueError(
            f"Dimension mismatch: resume_vectors has {resume_vectors.shape[1]} features, "
            f"but job_vector has {job_vector.shape[1]} features. "
            "They must have the same number of features."
        )
    
    # Calculate cosine similarity: cos(θ) = (A · B) / (||A|| * ||B||)
    # sklearn's cosine_similarity handles normalization automatically
    similarities = cosine_similarity(resume_vectors, job_vector)
    
    # Flatten to 1D array (n_resumes,)
    # Result shape is (n_resumes, 1), so flatten to (n_resumes,)
    return similarities.flatten()


def normalize_score(score: float) -> float:
    """
    Normalize similarity score to percentage (0-100).
    
    Args:
        score: Cosine similarity score (typically 0-1)
        
    Returns:
        Normalized percentage score (0-100)
    """
    # Cosine similarity is already in range [0, 1] for non-negative vectors
    # Clamp to ensure it's in valid range
    score = max(0.0, min(1.0, score))
    return score * 100.0


def rank_resumes(resume_names: List[str], 
                 similarity_scores: np.ndarray) -> List[Dict[str, any]]:
    """
    Rank resumes based on similarity scores.
    
    Args:
        resume_names: List of resume file names
        similarity_scores: Array of similarity scores
        
    Returns:
        List of dictionaries containing resume name, score, and rank,
        sorted by score (descending)
    """
    if len(resume_names) != len(similarity_scores):
        raise ValueError("Number of resume names must match number of scores")
    
    # Create list of results
    results = []
    for name, score in zip(resume_names, similarity_scores):
        normalized_score = normalize_score(score)
        results.append({
            'name': name,
            'score': normalized_score,
            'raw_score': float(score)
        })
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Add rank
    for idx, result in enumerate(results, start=1):
        result['rank'] = idx
    
    return results


def calculate_and_rank(resume_vectors: np.ndarray,
                      job_vector: np.ndarray,
                      resume_names: List[str]) -> List[Dict[str, any]]:
    """
    Complete pipeline: calculate similarity and rank resumes.
    
    Args:
        resume_vectors: Array of resume vectors (n_resumes x n_features)
        job_vector: Single job description vector (1 x n_features or n_features,)
        resume_names: List of resume file names
        
    Returns:
        List of ranked resume results
    """
    # Calculate similarities
    similarities = calculate_cosine_similarity(resume_vectors, job_vector)
    
    # Rank resumes
    ranked_results = rank_resumes(resume_names, similarities)
    
    return ranked_results

