"""
Standalone verification script - proves the TF-IDF + cosine similarity logic works
Run this to verify the core logic is correct before debugging the app
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_similarity():
    """Test the exact same logic used in the app"""
    
    print("=" * 70)
    print("TESTING TF-IDF + COSINE SIMILARITY LOGIC")
    print("=" * 70)
    
    # Test case 1: Simple overlapping text
    print("\n📝 TEST 1: Simple overlapping keywords")
    print("-" * 70)
    job_desc = "Python developer machine learning"
    resume1 = "Python developer machine learning experience"
    resume2 = "Java developer web"
    
    all_docs = [job_desc, resume1, resume2]
    print(f"Job: '{job_desc}'")
    print(f"Resume 1: '{resume1}'")
    print(f"Resume 2: '{resume2}'")
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 1),  # USE ONLY UNIGRAMS for better matching
        min_df=1,
        max_df=0.95
    )
    
    vectors = vectorizer.fit_transform(all_docs).toarray()
    job_vec = vectors[0:1]
    resume_vecs = vectors[1:]
    
    similarities = cosine_similarity(resume_vecs, job_vec).flatten()
    
    print(f"\nResults:")
    print(f"  Resume 1 similarity: {similarities[0]:.4f} ({similarities[0]*100:.2f}%)")
    print(f"  Resume 2 similarity: {similarities[1]:.4f} ({similarities[1]*100:.2f}%)")
    
    if similarities[0] > 0.5:
        print("  ✅ PASS: Resume 1 has high similarity")
    else:
        print("  ❌ FAIL: Resume 1 should have high similarity")
    
    if similarities[1] < similarities[0]:
        print("  ✅ PASS: Resume 2 has lower similarity than Resume 1")
    else:
        print("  ❌ FAIL: Resume 2 should have lower similarity")
    
    # Test case 2: More realistic
    print("\n📝 TEST 2: Realistic job description and resume")
    print("-" * 70)
    job_desc2 = """
    We are looking for a Python developer with experience in machine learning 
    and data science. The candidate should have strong programming skills and 
    knowledge of scikit-learn, pandas, and numpy.
    """
    resume_good = """
    I am a Python developer with 5 years of experience in machine learning 
    and data science. I have worked extensively with scikit-learn, pandas, 
    numpy, and have strong programming skills.
    """
    resume_bad = """
    I am a Java developer with experience in web development using Spring 
    framework. I have worked on enterprise applications and REST APIs.
    """
    
    all_docs2 = [job_desc2.strip(), resume_good.strip(), resume_bad.strip()]
    
    vectors2 = vectorizer.fit_transform(all_docs2).toarray()
    job_vec2 = vectors2[0:1]
    resume_vecs2 = vectors2[1:]
    
    similarities2 = cosine_similarity(resume_vecs2, job_vec2).flatten()
    
    print(f"\nResults:")
    print(f"  Good resume similarity: {similarities2[0]:.4f} ({similarities2[0]*100:.2f}%)")
    print(f"  Bad resume similarity: {similarities2[1]:.4f} ({similarities2[1]*100:.2f}%)")
    
    if similarities2[0] > 0.3:
        print("  ✅ PASS: Good resume has reasonable similarity")
    else:
        print("  ❌ FAIL: Good resume should have higher similarity")
    
    if similarities2[1] < similarities2[0]:
        print("  ✅ PASS: Bad resume correctly has lower similarity")
    else:
        print("  ❌ FAIL: Bad resume should have lower similarity")
    
    # Test case 3: Check for zero vectors
    print("\n📝 TEST 3: Edge case - empty/very short text")
    print("-" * 70)
    job_desc3 = "Python"
    resume3 = "Python developer"
    
    all_docs3 = [job_desc3, resume3]
    vectorizer3 = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 1),  # USE ONLY UNIGRAMS
        min_df=1,
        max_df=0.95
    )
    vectors3 = vectorizer3.fit_transform(all_docs3).toarray()
    job_vec3 = vectors3[0:1]
    resume_vec3 = vectors3[1:]
    
    similarity3 = cosine_similarity(resume_vec3, job_vec3)[0][0]
    print(f"Job: '{job_desc3}'")
    print(f"Resume: '{resume3}'")
    print(f"Similarity: {similarity3:.4f} ({similarity3*100:.2f}%)")
    
    if similarity3 > 0:
        print("  ✅ PASS: Even short texts produce non-zero similarity")
    else:
        print("  ❌ FAIL: Should produce non-zero similarity")
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print("\nIf all tests pass, the logic is correct.")
    print("If tests fail, there's an issue with the TF-IDF/cosine similarity setup.")

if __name__ == "__main__":
    test_similarity()

