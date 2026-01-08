"""
Streamlit Resume Screening Application

Main application file for the NLP Resume Screening System.
Allows users to upload multiple resumes and match them against a job description.
"""

import streamlit as st
import sys
import os
from typing import List, Dict, Optional
import traceback
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_extraction import extract_text


# Page configuration
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: clamp(2.2rem, 3vw, 4.5rem) !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""


def get_match_category_hybrid(match_percentage: float, matched_skill_count: int) -> tuple[str, str]:
    """
    HYBRID DECISION LOGIC: Categorize match using both similarity score AND skill count.
    
    This hybrid approach prevents false rejections where resumes with many matched skills
    are incorrectly rejected due to lower similarity scores. Real ATS systems consider
    both overall relevance (similarity) and hard skill requirements (skill matches).
    
    Decision Rules:
    1. HIGH MATCH: match_percentage >= 60
       - Strong overall similarity indicates good fit regardless of explicit skill count
    
    2. MEDIUM MATCH: match_percentage >= 40 OR (match_percentage >= 30 AND matched_skill_count >= 5)
       - Uses OR logic: Either good similarity (≥40%) OR sufficient skills (≥30% + ≥5 skills)
       - This prevents rejecting skill-strong candidates with different writing styles
    
    3. LOW MATCH: match_percentage < 30 AND matched_skill_count < 5
       - Uses AND logic: Both low similarity AND insufficient skills must be true
       - Only rejects when candidate lacks both relevance and required skills
    
    Args:
        match_percentage: TF-IDF cosine similarity score (0-100)
        matched_skill_count: Number of skills found in both JD and resume
        
    Returns:
        Tuple of (category_label, emoji)
    """
    # Rule 1: HIGH MATCH - Strong similarity score (≥60%) indicates good overall fit
    # This candidate has strong textual similarity regardless of explicit skill count
    if match_percentage >= 60:
        return "High Match", "⭐"
    
    # Rule 2: MEDIUM MATCH - Uses OR logic to prevent false rejections
    # Condition A: Good similarity (≥40%) - proceed even with fewer skills
    # Condition B: Lower similarity (30-39%) BUT has sufficient skills (≥5) - skill match compensates
    # OR logic ensures: If EITHER condition is true, candidate is still viable
    # This prevents rejecting candidates who have required skills but different wording
    elif match_percentage >= 40 or (match_percentage >= 30 and matched_skill_count >= 5):
        return "Medium Match", "⚠️"
    
    # Rule 3: LOW MATCH - Uses AND logic for strict rejection
    # Both conditions must be true: low similarity (<30%) AND insufficient skills (<5)
    # AND logic ensures: Only reject when candidate lacks BOTH relevance AND skills
    # This prevents rejecting candidates who have either good similarity OR sufficient skills
    else:
        return "Low Match", "❌"


def get_recommendation_hybrid(match_percentage: float, matched_skill_count: int) -> str:
    """
    HYBRID DECISION LOGIC: Get recruiter recommendation using both similarity and skills.
    
    This function provides recommendations that reflect the hybrid decision logic,
    giving context about why a candidate is recommended or not.
    
    Recommendation rules match the category logic:
    - High Match (≥60%): Strong candidate
    - Medium Match (≥40% OR (≥30% AND ≥5 skills)): Can be considered
    - Low Match (<30% AND <5 skills): Not recommended
    
    Args:
        match_percentage: TF-IDF cosine similarity score (0-100)
        matched_skill_count: Number of skills found in both JD and resume
        
    Returns:
        Recommendation text string
    """
    # High Match: Strong candidate based on overall similarity (≥60%)
    if match_percentage >= 60:
        return "Strong candidate for interview"
    
    # Medium Match: Uses OR logic - either good similarity OR sufficient skills
    # This recommendation applies when candidate meets medium match criteria
    elif match_percentage >= 40 or (match_percentage >= 30 and matched_skill_count >= 5):
        return "Can be considered due to skill match"
    
    # Low Match: Both low similarity AND insufficient skills (AND logic)
    # Only recommend rejection when candidate lacks both dimensions
    else:
        return "Not recommended for shortlisting"


def extract_skills_from_text(text: str) -> set[str]:
    """
    Extract important technical skills from text.
    
    This function looks for common data science and software engineering skills
    in the text, handling variations and case-insensitive matching.
    
    Args:
        text: Input text (job description or resume)
        
    Returns:
        Set of found skills (normalized to lowercase)
    """
    # Define common skills with their variations
    skill_patterns = {
        'python': ['python', 'py'],
        'pandas': ['pandas', 'pd'],
        'numpy': ['numpy', 'np'],
        'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite'],
        'machine learning': ['machine learning', 'ml', 'machine-learning'],
        'scikit-learn': ['scikit-learn', 'scikit learn', 'sklearn', 'scikit'],
        'data analysis': ['data analysis', 'data analytics', 'analyzing data'],
        'nlp': ['nlp', 'natural language processing', 'natural-language-processing'],
        'streamlit': ['streamlit', 'stream-lit'],
        'tensorflow': ['tensorflow', 'tf'],
        'pytorch': ['pytorch', 'torch'],
        'keras': ['keras'],
        'matplotlib': ['matplotlib', 'plt'],
        'seaborn': ['seaborn'],
        'scipy': ['scipy'],
        'jupyter': ['jupyter', 'jupyter notebook', 'notebook'],
        'git': ['git', 'github', 'gitlab'],
        'docker': ['docker', 'containerization'],
        'aws': ['aws', 'amazon web services'],
        'azure': ['azure', 'microsoft azure'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform'],
        'spark': ['spark', 'apache spark', 'pyspark'],
        'hadoop': ['hadoop'],
        'tableau': ['tableau'],
        'power bi': ['power bi', 'powerbi', 'power-bi'],
        'excel': ['excel', 'microsoft excel'],
        'r': ['r programming', 'r language', ' r '],
        'java': ['java'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
        'react': ['react', 'reactjs'],
        'django': ['django'],
        'flask': ['flask'],
        'fastapi': ['fastapi', 'fast api'],
        'mongodb': ['mongodb', 'mongo'],
        'redis': ['redis'],
        'kubernetes': ['kubernetes', 'k8s'],
        'linux': ['linux', 'unix'],
        'api': ['api', 'rest api', 'restful'],
        'deep learning': ['deep learning', 'deep-learning', 'neural network', 'neural networks'],
        'computer vision': ['computer vision', 'cv', 'opencv'],
        'statistics': ['statistics', 'statistical', 'stats'],
        'data visualization': ['data visualization', 'data viz', 'visualization'],
    }
    
    text_lower = text.lower()
    found_skills = set()
    
    # Check for each skill pattern
    for skill_name, patterns in skill_patterns.items():
        for pattern in patterns:
            # Use word boundaries to avoid partial matches
            import re
            if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text_lower):
                found_skills.add(skill_name)
                break  # Found this skill, move to next
    
    return found_skills


def match_skills(job_description: str, resume_text: str) -> tuple[set[str], set[str]]:
    """
    Compare skills between job description and resume.
    
    Args:
        job_description: Job description text
        resume_text: Resume text
        
    Returns:
        Tuple of (matched_skills, missing_skills)
    """
    # Extract skills from both documents
    job_skills = extract_skills_from_text(job_description)
    resume_skills = extract_skills_from_text(resume_text)
    
    # Find matched and missing skills
    matched_skills = job_skills.intersection(resume_skills)
    missing_skills = job_skills - resume_skills
    
    return matched_skills, missing_skills


def process_resumes(uploaded_files: List, job_description: str) -> Optional[List[Dict]]:
    """
    Process uploaded resumes and calculate similarity scores.
    
    Args:
        uploaded_files: List of uploaded file objects
        job_description: Job description text
        
    Returns:
        List of ranked results or None if error occurs
    """
    try:
        if not uploaded_files:
            st.error("Please upload at least one resume file.")
            return None
        
        if not job_description or not job_description.strip():
            st.error("Please enter a job description.")
            return None
        
        # Step 1: Extract FULL text from all uploaded resume files
        # Note: We process the complete document, not just a portion
        # This ensures all skills and experience are captured for accurate matching
        resume_texts = []
        resume_names = []
        
        with st.spinner("Extracting text from resumes..."):
            for uploaded_file in uploaded_files:
                try:
                    file_content = uploaded_file.read()
                    # Extract FULL text from PDF/DOCX/TXT files
                    text = extract_text(file_content, uploaded_file.name)
                    
                    if text and text.strip():
                        # Store complete resume text for processing
                        resume_texts.append(text)
                        resume_names.append(uploaded_file.name)
                    else:
                        st.warning(f"⚠️ Could not extract text from {uploaded_file.name}. Skipping...")
                except Exception as e:
                    st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)}. Skipping...")
                    continue
        
        if not resume_texts:
            st.error("No valid text could be extracted from the uploaded files.")
            return None
        
        # Validation: Ensure we have non-empty text
        if not job_description.strip():
            st.error("Job description is empty. Please enter a job description.")
            return None
        
        # Validation: Check text lengths (very short text might cause issues)
        if len(job_description.strip()) < 10:
            st.warning("⚠️ Warning: Job description is very short. Results may be inaccurate.")
        
        for i, text in enumerate(resume_texts):
            if len(text.strip()) < 10:
                st.warning(f"⚠️ Warning: Resume '{resume_names[i]}' is very short. Results may be inaccurate.")
        
        # BULLETPROOF FIX: Use direct sklearn implementation (no wrapper classes)
        # This ensures we're using the exact same logic that works in tests
        
        with st.spinner("Vectorizing documents..."):
            # Step 1: Prepare documents - combine job description + resumes
            # CRITICAL: Job description MUST be first to maintain index 0
            # IMPORTANT: We process the FULL text of all documents, not just a portion
            all_documents = [job_description.strip()] + [text.strip() for text in resume_texts]
            
            # Show document info to confirm full text processing
            total_chars = sum(len(doc) for doc in all_documents)
            st.info(f"📄 Processing FULL documents: Job ({len(job_description)} chars) + {len(resume_texts)} resume(s) = {total_chars:,} total characters")
            
            # Step 2: Create TF-IDF vectorizer using scikit-learn's built-in tokenizer
            # This is deployment-safe and works on Streamlit Cloud without external dependencies
            # Scikit-learn's built-in tokenizer handles:
            # - Lowercasing (case-insensitive matching)
            # - Stopword removal (removes common English words)
            # - Tokenization (splits text into words)
            # - N-gram extraction (captures word combinations)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Use scikit-learn's built-in tokenizer - no external dependencies required
            # This ensures the app works on Streamlit Cloud without NLTK or spaCy downloads
            # Parameters:
            # - stop_words='english': Removes common English stopwords
            # - ngram_range=(1, 2): Uses unigrams and bigrams for better matching
            # - max_features=5000: Limits vocabulary size for efficiency
            # - lowercase=True: Case-insensitive matching
            # - min_df=1, max_df=0.95: Filters very rare and very common terms
            vectorizer = TfidfVectorizer(
                stop_words='english',      # Built-in English stopword removal
                lowercase=True,            # Case-insensitive matching
                ngram_range=(1, 2),        # Unigrams and bigrams for better phrase matching
                max_features=5000,         # Limit vocabulary size for efficiency
                min_df=1,                  # Term must appear in at least 1 document
                max_df=0.95                # Ignore terms appearing in >95% of documents
            )
            
            # Step 3: Fit and transform ALL documents together
            # This creates a consistent feature space where all documents share the same vocabulary
            # CRITICAL: Must fit on ALL documents (job + resumes) together
            # This ensures TF-IDF vectors are comparable and cosine similarity works correctly
            all_vectors = vectorizer.fit_transform(all_documents).toarray()
            
            # DEBUG: Show vocabulary information (optional, can be removed in production)
            feature_names = vectorizer.get_feature_names_out()
            with st.expander("🔍 Vocabulary Debug (Click to view)"):
                st.write(f"**Total vocabulary size:** {len(feature_names)}")
                st.write(f"**Sample features (first 50):** {', '.join(feature_names[:50])}...")
                
                # Check if common words are in vocabulary
                common_words = ['python', 'machine', 'learn', 'data', 'analysis', 'model', 'build', 'streamlit', 'panda', 'numpy', 'scikit']
                found_words = [w for w in common_words if w in feature_names]
                if found_words:
                    st.write(f"\n**Common words found:** {', '.join(found_words)}")
                missing_words = [w for w in common_words if w not in feature_names]
                if missing_words:
                    st.write(f"**Common words missing:** {', '.join(missing_words)}")
            
            # DEBUG: Check if vectors were created correctly
            if all_vectors.shape[0] != len(all_documents):
                st.error(f"❌ CRITICAL: Vector count mismatch! Expected {len(all_documents)}, got {all_vectors.shape[0]}")
                return None
            
            # Step 4: Extract job vector (index 0) and resume vectors (indices 1+)
            # Keep job vector as 2D array (1 x n_features) for sklearn compatibility
            job_vector = all_vectors[0:1]  # Shape: (1, n_features)
            resume_vectors = all_vectors[1:]  # Shape: (n_resumes, n_features)
            
            # Validation checks
            if len(resume_names) != len(resume_vectors):
                raise ValueError(
                    f"Critical error: {len(resume_names)} resumes but {len(resume_vectors)} vectors"
                )
            
            # Debug information (can be removed in production)
            job_vector_sum = job_vector.sum()
            resume_vector_sums = [vec.sum() for vec in resume_vectors]
            
            # Check for zero vectors (indicates text extraction problem)
            if job_vector_sum == 0:
                st.error("❌ ERROR: Job description vector is all zeros. Text extraction may have failed.")
                st.error(f"Job description text length: {len(job_description.strip())} characters")
                return None
            
            # Check resume vectors
            zero_vector_count = sum(1 for s in resume_vector_sums if s == 0)
            if zero_vector_count > 0:
                st.warning(f"⚠️ Warning: {zero_vector_count} resume(s) have zero vectors. They will show 0% match.")
            
            # Verify vector dimensions match
            if job_vector.shape[1] != resume_vectors.shape[1]:
                st.error(f"❌ CRITICAL ERROR: Dimension mismatch! Job: {job_vector.shape[1]}, Resumes: {resume_vectors.shape[1]}")
                return None
        
        # Step 5: Calculate cosine similarity using sklearn directly
        with st.spinner("Calculating similarity scores..."):
            # DEBUG: Show vector information
            with st.expander("🔍 Debug Information (Click to view)"):
                st.write(f"**Vector Shapes:**")
                st.write(f"- Job vector: {job_vector.shape}")
                st.write(f"- Resume vectors: {resume_vectors.shape}")
                st.write(f"- Number of features: {job_vector.shape[1]}")
                
                st.write(f"\n**Vector Statistics:**")
                st.write(f"- Job vector sum: {job_vector_sum:.4f}")
                st.write(f"- Job vector non-zero elements: {np.count_nonzero(job_vector)}")
                for i, (name, vec_sum) in enumerate(zip(resume_names, resume_vector_sums)):
                    st.write(f"- {name}: sum={vec_sum:.4f}, non-zero={np.count_nonzero(resume_vectors[i])}")
                
                # Show sample of job description and first resume
                # NOTE: We process the FULL document, this is just a preview for debugging
                st.write(f"\n**Text Preview (FULL documents are processed):**")
                st.write(f"- Job description: {len(job_description)} chars total")
                st.write(f"  Preview (first 200 chars): {job_description[:200]}...")
                if resume_texts:
                    for i, resume_text in enumerate(resume_texts):
                        st.write(f"- Resume {i+1} ({resume_names[i]}): {len(resume_text)} chars total")
                        st.write(f"  Preview (first 200 chars): {resume_text[:200]}...")
            
            # CRITICAL CHECK: Verify there's vocabulary overlap
            # Check if job vector and resume vectors share any non-zero features
            job_nonzero_indices = set(np.nonzero(job_vector[0])[0])
            feature_names = vectorizer.get_feature_names_out()
            
            for i, (name, resume_vec) in enumerate(zip(resume_names, resume_vectors)):
                resume_nonzero_indices = set(np.nonzero(resume_vec)[0])
                overlap = job_nonzero_indices.intersection(resume_nonzero_indices)
                
                # Show ALL features, not just a sample
                job_feature_indices = list(job_nonzero_indices)
                resume_feature_indices = list(resume_nonzero_indices)
                
                job_features = [feature_names[idx] for idx in job_feature_indices]
                resume_features = [feature_names[idx] for idx in resume_feature_indices]
                
                job_set = set(job_features)
                resume_set = set(resume_features)
                actual_overlap = job_set.intersection(resume_set)
                
                if len(actual_overlap) == 0:
                    st.error(f"❌ CRITICAL: No vocabulary overlap detected for '{name}'. This will result in 0% similarity.")
                    
                    st.write(f"  **ALL Job features ({len(job_features)}):** {', '.join(sorted(job_features))}")
                    st.write(f"  **ALL Resume features ({len(resume_features)}):** {', '.join(sorted(resume_features))}")
                    
                    # Check for expected matches
                    expected_words = ['python', 'machine', 'learn', 'data', 'analysis', 'model', 'build', 'streamlit', 'panda', 'numpy', 'scikit', 'science']
                    found_in_job = [w for w in expected_words if w in job_set]
                    found_in_resume = [w for w in expected_words if w in resume_set]
                    should_match = [w for w in expected_words if w in job_set and w in resume_set]
                    
                    st.write(f"\n  **Analysis:**")
                    if should_match:
                        st.error(f"  ❌ BUG DETECTED! These words SHOULD match but vectors show 0 overlap: {', '.join(should_match)}")
                        st.error(f"  This indicates the vectors are not being extracted correctly!")
                    else:
                        st.write(f"  - Expected words in job: {', '.join(found_in_job) if found_in_job else 'NONE'}")
                        st.write(f"  - Expected words in resume: {', '.join(found_in_resume) if found_in_resume else 'NONE'}")
                        st.write(f"  - Words that should match: {', '.join(should_match) if should_match else 'NONE'}")
                    
                    # Note: Using scikit-learn's built-in tokenizer (deployment-safe)
                    st.info("ℹ️ Using scikit-learn's built-in tokenizer with stopword removal.")
                else:
                    st.success(f"✅ Found {len(actual_overlap)} overlapping features: {', '.join(sorted(actual_overlap))}")
                    if len(overlap) == 0:
                        st.error(f"  ❌ BUG: Feature sets overlap but vector indices don't! This is a critical error.")
                        st.write(f"  Feature overlap: {actual_overlap}")
                        st.write(f"  Vector index overlap: {overlap}")
            
            # Use sklearn's cosine_similarity directly (most reliable)
            # Input: resume_vectors (n_resumes x n_features), job_vector (1 x n_features)
            # Output: (n_resumes x 1) array
            similarities = cosine_similarity(resume_vectors, job_vector)
            
            # DEBUG: Show raw similarities before conversion
            with st.expander("🔍 Debug Information (Click to view)"):
                st.write(f"**Raw Similarity Scores:**")
                for name, sim in zip(resume_names, similarities.flatten()):
                    st.write(f"- {name}: {sim:.6f} (raw), {sim*100:.2f}%")
                
                # Show vocabulary overlap info
                st.write(f"\n**Vocabulary Overlap:**")
                for i, (name, resume_vec) in enumerate(zip(resume_names, resume_vectors)):
                    job_nonzero = set(np.nonzero(job_vector[0])[0])
                    resume_nonzero = set(np.nonzero(resume_vec)[0])
                    overlap_count = len(job_nonzero.intersection(resume_nonzero))
                    st.write(f"- {name}: {overlap_count} overlapping features out of {len(job_nonzero)} job features")
            
            # Flatten to 1D array (n_resumes,)
            similarity_scores = similarities.flatten()
            
            # HYBRID DECISION LOGIC: Convert to percentages and create results
            # This step implements the hybrid approach that considers BOTH similarity score AND skill count
            # This prevents false rejections where resumes with many matched skills are rejected
            # due to lower similarity scores (e.g., different writing style but same skills)
            results = []
            for i, (name, score) in enumerate(zip(resume_names, similarity_scores)):
                # Step 1: Calculate match percentage from TF-IDF cosine similarity
                # This represents overall textual relevance between job description and resume
                score_clamped = max(0.0, min(1.0, float(score)))
                match_percentage = score_clamped * 100.0
                
                # Step 2: Extract and match skills for this resume
                # This identifies hard skill requirements that are explicitly mentioned
                # Note: resume_texts[i] corresponds to resume_names[i] - process FULL text
                matched_skills, missing_skills = match_skills(job_description, resume_texts[i])
                matched_skill_count = len(matched_skills)
                
                # Step 3: Apply HYBRID DECISION LOGIC
                # Decision considers BOTH:
                #   - match_percentage: Overall textual similarity (TF-IDF cosine similarity)
                #   - matched_skill_count: Number of hard skills found in both documents
                # 
                # Why hybrid logic?
                # - Prevents false rejections: A resume with 8 matched skills but 35% similarity
                #   should not be rejected if skills are critical requirements
                # - Balances soft and hard requirements: Similarity captures overall fit,
                #   skill count captures specific technical requirements
                # - Mimics real ATS behavior: Professional ATS systems use multi-factor scoring
                category, emoji = get_match_category_hybrid(match_percentage, matched_skill_count)
                recommendation = get_recommendation_hybrid(match_percentage, matched_skill_count)
                
                # Step 4: Build result dictionary with all information
                # This includes both similarity metrics and skill analysis for transparency
                results.append({
                    'name': name,
                    'score': match_percentage,  # Match percentage for display
                    'raw_score': score_clamped,  # Raw similarity score
                    'matched_skill_count': matched_skill_count,  # Number of matched skills
                    'category': category,  # Final category from hybrid logic
                    'emoji': emoji,  # Category emoji
                    'recommendation': recommendation,  # Recruiter recommendation
                    'matched_skills': sorted(matched_skills),  # List of matched skills
                    'missing_skills': sorted(missing_skills)  # List of missing skills
                })
            
            # Sort by score (descending) and add rank
            # This ensures best matches appear first
            results.sort(key=lambda x: x['score'], reverse=True)
            for idx, result in enumerate(results, start=1):
                result['rank'] = idx
        
        return results
    
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        st.error("Please check the console for more details.")
        st.code(traceback.format_exc())
        return None


def display_results(results: List[Dict]):
    """
    Display ranked results in compact ATS-style format optimized for recruiter workflow.
    
    UX Design Principles:
    - Key information visible at a glance (match %, category, recommendation)
    - Detailed information in expandable sections to reduce vertical clutter
    - Use columns to maximize horizontal space utilization
    - Compact skill formatting for quick scanning
    
    Args:
        results: List of ranked resume results with enhanced information
    """
    st.markdown("---")
    st.markdown("## 📊 Screening Results")
    
    if not results:
        st.info("No results to display.")
        return
    
    # Summary metrics - compact horizontal layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resumes", len(results))
    with col2:
        avg_score = sum(r['score'] for r in results) / len(results)
        st.metric("Average Match Score", f"{avg_score:.2f}%")
    with col3:
        max_score = max(r['score'] for r in results)
        st.metric("Highest Match", f"{max_score:.2f}%")
    
    st.markdown("---")
    st.markdown("### 🎯 Ranked Resumes")
    
    # Display each result in compact format
    for result in results:
        with st.container():
            # Header: Rank and filename in single row
            col_rank, col_name = st.columns([1, 5])
            with col_rank:
                st.markdown(f"### #{result['rank']}")
            with col_name:
                st.markdown(f"### {result['name']}")
            
            # KEY INFORMATION: Match Percentage, Category, Recommendation in single row
            # Using columns reduces vertical space - recruiter sees all key info at once
            col_pct, col_cat, col_rec = st.columns([2, 2, 3])
            
            with col_pct:
                # Match Percentage with compact progress bar
                st.markdown("**Match:**")
                st.progress(result['score'] / 100.0)
                st.markdown(f"**{result['score']:.2f}%**")
            
            with col_cat:
                # Final Category - highlight with appropriate color
                st.markdown("**Category:**")
                category_display = f"{result['emoji']} {result['category']}"
                # Use color coding ONLY for final decision - creates clear visual hierarchy
                if result['category'] == "High Match":
                    st.success(category_display)
                elif result['category'] == "Medium Match":
                    st.warning(category_display)
                else:
                    st.error(category_display)
            
            with col_rec:
                # Recommendation - concise and actionable
                st.markdown("**Recommendation:**")
                st.info(result['recommendation'])
            
            # DETAILED INFORMATION: Move skills into expandable section
            # Expanders reduce visual clutter - recruiter expands only when needed
            # This follows the principle of progressive disclosure in UX design
            matched_count = result.get('matched_skill_count', len(result.get('matched_skills', [])))
            
            with st.expander(f"📋 View Skill Match Details ({matched_count} matched, {len(result.get('missing_skills', []))} missing)"):
                # Skill breakdown in compact format
                col_skills1, col_skills2 = st.columns(2)
                
                with col_skills1:
                    st.markdown("**✅ Matched Skills:**")
                    if result['matched_skills']:
                        # Compact comma-separated format - easier to scan than bullet lists
                        # Removes visual noise while preserving all information
                        skills_text = ", ".join(result['matched_skills'])
                        st.write(skills_text)
                        st.caption(f"{len(result['matched_skills'])} skills matched")
                    else:
                        st.warning("No skills matched")
                
                with col_skills2:
                    st.markdown("**❌ Missing Skills:**")
                    if result['missing_skills']:
                        # Compact comma-separated format for quick scanning
                        skills_text = ", ".join(result['missing_skills'])
                        st.write(skills_text)
                        st.caption(f"{len(result['missing_skills'])} skills missing")
                    else:
                        st.success("All required skills present! 🎉")
                
                # Decision explanation with correct hybrid logic description
                # This explains how the system combines similarity and skill matching
                with st.expander("ℹ️ How this decision was made"):
                    st.markdown("**The system uses a hybrid evaluation approach:**")
                    st.markdown("""
                    • **TF-IDF similarity** measures overall relevance between resume and job description
                    • **Skill matching** verifies required technical skills
                    """)
                    st.markdown("---")
                    st.markdown("**Decision rules:**")
                    st.markdown("""
                    • **High Match:** Similarity ≥ 60%
                    • **Medium Match:** Similarity ≥ 40% OR (Similarity ≥ 30% AND at least 5 skills matched)
                    • **Low Match:** Similarity < 30% AND fewer than 5 skills matched
                    """)
                    st.markdown("---")
                    st.caption("This hybrid approach reduces false rejections and reflects real-world ATS shortlisting behavior.")
                    
                    # Show specific values for this candidate
                    st.markdown("---")
                    st.markdown("**This candidate's metrics:**")
                    st.write(f"• Match Percentage: {result['score']:.2f}%")
                    st.write(f"• Matched Skills: {matched_count} skills")
                    st.write(f"• Final Category: {result['category']}")
                    st.write(f"• Recommendation: {result['recommendation']}")
            
            st.markdown("---")


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">📄 Resume Screening System</p>', unsafe_allow_html=True)
    st.markdown("### Match resumes with job descriptions using advanced NLP techniques")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        1. **Upload Resumes**: Select one or more resume files (PDF, DOCX, or TXT)
        2. **Enter Job Description**: Paste or type the job description text
        3. **Click Analyze**: The system will process and rank the resumes
        4. **View Results**: See ranked results with match percentages
        """)
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses:
        - **TF-IDF Vectorization** for feature extraction
        - **Cosine Similarity** for matching
        - **NLP Preprocessing** (stopword removal, n-gram extraction)
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt', 'doc'],
            accept_multiple_files=True,
            help="Upload one or more resume files in PDF, DOCX, or TXT format"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.markdown("### 💼 Job Description")
        job_description = st.text_area(
            "Enter or paste the job description",
            height=300,
            help="Enter the complete job description text here",
            value=st.session_state.job_description
        )
        st.session_state.job_description = job_description
        
        if job_description:
            word_count = len(job_description.split())
            st.info(f"📝 {word_count} words")
    
    st.markdown("---")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Resumes",
            type="primary",
            use_container_width=True
        )
    
    # Process when button is clicked
    if analyze_button:
        if not uploaded_files:
            st.error("❌ Please upload at least one resume file.")
        elif not job_description or not job_description.strip():
            st.error("❌ Please enter a job description.")
        else:
            # Process resumes
            results = process_resumes(uploaded_files, job_description)
            
            if results:
                st.session_state.results = results
                display_results(results)
    
    # Display previous results if available
    if st.session_state.results and not analyze_button:
        display_results(st.session_state.results)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "Built with ❤️ using Streamlit and scikit-learn"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

