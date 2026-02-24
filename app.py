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
import re
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_extraction import extract_text


@st.cache_resource(show_spinner=False)
def load_semantic_model():
    """
    Load SentenceTransformer model once and cache it.
    Using all-MiniLM-L6-v2 for efficient, high-quality sentence embeddings.
    
    If the model or library is not available (e.g. on Streamlit Cloud without
    the dependency), we gracefully fall back to a None model so the app can
    still return results using skills + experience only.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        # Show a one-time warning in the app; scoring will fall back to non-semantic.
        st.warning(
            "Semantic model (SentenceTransformers) could not be loaded. "
            "Falling back to skill- and experience-based scoring only."
        )
        return None


def clean_text(text: str) -> str:
    """
    Lightweight text cleaning without NLTK.

    - Lowercases text
    - Keeps letters, digits, basic punctuation
    - Normalizes extra whitespace
    """
    if not text:
        return ""
    text = text.lower()
    # Keep letters, digits, whitespace and basic punctuation; drop the rest
    text = re.sub(r"[^a-z0-9\s.,;:!?@#\$%\-\+_/]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


EXPERIENCE_KEYWORDS = [
    "project",
    "projects",
    "internship",
    "internships",
    "experience",
    "experienced",
    "developed",
    "built",
    "implemented",
    "deployed",
]


def compute_experience_score(text: str) -> float:
    """
    Compute experience / context score based on presence of experience-related keywords.

    Returns a percentage in [0, 100]:
    - 0% if no experience keywords are present
    - Up to 100% if many distinct experience keywords are present
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    hits = {kw for kw in EXPERIENCE_KEYWORDS if kw in text_lower}

    if not EXPERIENCE_KEYWORDS:
        return 0.0

    ratio = len(hits) / len(set(EXPERIENCE_KEYWORDS))
    return max(0.0, min(100.0, ratio * 100.0))


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
    # Preserve whether the last run was a single-resume vs multi-resume run
    # so the UI can keep the intended behavior across Streamlit reruns.
    if 'last_uploaded_count' not in st.session_state:
        st.session_state.last_uploaded_count = 0


def get_match_category_hybrid(final_score: float, matched_skill_count: int) -> tuple[str, str]:
    """
    HYBRID DECISION LOGIC (FINAL SCORE):
    Categorize candidate using the final hybrid score that combines:
    - Semantic similarity (Sentence Transformers)
    - Skill match score
    - Experience / context score
    
    Decision Rules (based on final_score in [0, 100]):
    - High Match:  final_score ≥ 65
    - Medium Match: 45 ≤ final_score < 65
    - Low Match:   final_score < 45
    
    Args:
        final_score: Final hybrid ATS-style score (0-100)
        matched_skill_count: Number of matched skills (kept for potential future use)
        
    Returns:
        Tuple of (category_label, emoji)
    """
    # HIGH MATCH: Strong overall fit across semantic similarity, skills, and experience.
    if final_score >= 65:
        return "High Match", "⭐"
    
    # MEDIUM MATCH: Reasonable fit; may be considered depending on context.
    elif final_score >= 45:
        return "Medium Match", "⚠️"
    
    # LOW MATCH: Hybrid score too low; candidate is unlikely to be a good fit.
    else:
        return "Low Match", "❌"


def get_recommendation_hybrid(final_score: float, matched_skill_count: int) -> str:
    """
    HYBRID DECISION LOGIC: Get recruiter recommendation from final hybrid score.
    
    Recommendation rules match the category logic:
    - High Match  (final_score ≥ 65): Strong candidate
    - Medium Match (45 ≤ final_score < 65): Can be considered
    - Low Match   (final_score < 45): Not recommended
    
    Args:
        final_score: Final hybrid ATS-style score (0-100)
        matched_skill_count: Number of matched skills (kept for potential use / logging)
        
    Returns:
        Recommendation text string
    """
    # High Match: Strong candidate based on overall hybrid score
    if final_score >= 65:
        return "Strong candidate for interview"
    
    # Medium Match: Can be considered depending on role and competition
    elif final_score >= 45:
        return "Can be considered due to skill match"
    
    # Low Match: Not recommended for shortlisting
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
        # Note: We process the complete document, not just a portion.
        # This ensures all skills and experience are captured for accurate matching.
        resume_texts = []
        resume_names = []
        extraction_debug: list[tuple[str, int]] = []  # (filename, extracted_length)
        
        with st.spinner("Extracting text from resumes..."):
            for uploaded_file in uploaded_files:
                try:
                    file_content = uploaded_file.read()
                    # Extract FULL text from PDF/DOCX/TXT files
                    text = extract_text(file_content, uploaded_file.name)
                    text_len = len(text.strip()) if text else 0
                    extraction_debug.append((uploaded_file.name, text_len))
                    
                    if text and text.strip():
                        # Store complete resume text for processing
                        resume_texts.append(text)
                        resume_names.append(uploaded_file.name)
                    else:
                        # If text extraction fails or returns empty, warn and skip this file
                        st.warning(f"⚠️ Could not extract text from {uploaded_file.name} (0 characters). Skipping...")
                except Exception as e:
                    st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)}. Skipping...")
                    extraction_debug.append((uploaded_file.name, 0))
                    continue
        
        # Debug: Show extracted text lengths for each resume so we can confirm PDF/DOCX/TXT extraction.
        # This helps quickly identify cases where extraction silently fails.
        with st.expander("🔍 Extraction Debug (Click to view)"):
            for fname, length in extraction_debug:
                st.write(f"- {fname}: {length} characters extracted")
        
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
        
        # --- HYBRID SCORING PIPELINE ---------------------------------------
        # 1) Semantic similarity (Sentence Transformers)
        # 2) Skill-based match score
        # 3) Experience / context score
        # final_score = 0.5 * semantic + 0.3 * skills + 0.2 * experience
        # -------------------------------------------------------------------
        
        # Prepare cleaned texts for semantic model.
        # PERFORMANCE: limit resume text length to first 4000 characters before embedding
        # to keep encoding fast and memory efficient on Streamlit Cloud.
        jd_clean = clean_text(job_description)
        resumes_clean = [clean_text(text[:4000]) for text in resume_texts]
        
        # Load sentence-transformer model once (cached)
        model = load_semantic_model()
        
        # Default: no semantic signal (will be overridden if model works)
        semantic_scores = np.zeros(len(resume_texts), dtype=float)
        
        if model is not None:
            try:
                with st.spinner("Computing semantic similarity with Sentence Transformers..."):
                    # Encode job description and resumes into embeddings
                    jd_emb = model.encode(
                        jd_clean,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    resume_embs = model.encode(
                        resumes_clean,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )  # shape: (n_resumes, dim)
                    
                    # Cosine similarity via dot product (embeddings are normalized).
                    # Raw range is [-1, 1]. We normalize to [0, 100] using:
                    #   norm_sim = (cos + 1) / 2  -> [0, 1]
                    #   semantic_scores = norm_sim * 100
                    semantic_sims = np.dot(resume_embs, jd_emb)  # (n_resumes,)
                    norm_sims = (semantic_sims + 1.0) / 2.0
                    semantic_scores = np.clip(norm_sims * 100.0, 0.0, 100.0)
            except Exception as e:
                # If anything goes wrong with embeddings, log the issue and fall back
                st.warning(
                    "Semantic similarity computation failed; falling back to "
                    "skill- and experience-based scoring only."
                )

        # Skill-based scoring: compute required skills once from job description
        job_skills = extract_skills_from_text(job_description)
        total_required_skills = len(job_skills)

        results: List[Dict] = []

        for i, name in enumerate(resume_names):
            resume_text = resume_texts[i]
            
            # A) Semantic similarity component (0–100)
            semantic_score = float(semantic_scores[i])
            
            # B) Skill-based component (0–100)
            #    Use existing keyword matcher to get matched / missing skills.
            matched_skills, missing_skills = match_skills(job_description, resume_text)
            matched_skill_count = len(matched_skills)
            
            # Each unique skill is counted once; repeating the same skill word
            # many times in the resume does NOT increase matched_skill_count.
            if total_required_skills > 0:
                skill_match_score = (matched_skill_count / total_required_skills) * 100.0
            else:
                skill_match_score = 0.0
            
            # C) Experience / context component (0–100)
            keyword_context_score = compute_experience_score(resume_text)
            
            # BASE HYBRID SCORE (0–100)
            # Updated weights to strongly prioritize semantic understanding:
            #   0.55 * semantic + 0.30 * skills + 0.15 * context
            final_score = (
                0.55 * semantic_score
                + 0.30 * skill_match_score
                + 0.15 * keyword_context_score
            )
            
            # HARD GATING RULE:
            # If both semantic similarity and skill match are low, cap score at 40 (Low Match).
            if skill_match_score < 30.0 and semantic_score < 45.0:
                final_score = min(final_score, 40.0)
            
            # STRONGER ANTI SKILL-STUFFING RULE:
            # If skills are very high (>70%) but semantic similarity is still low (<50%),
            # treat this as keyword stuffing and subtract 15 points.
            if skill_match_score > 70.0 and semantic_score < 50.0:
                final_score -= 15.0
            
            # Clamp final score to [0, 100]
            final_score = max(0.0, min(100.0, final_score))
            
            # Map final_score to match category and recommendation
            category, emoji = get_match_category_hybrid(final_score, matched_skill_count)
            recommendation = get_recommendation_hybrid(final_score, matched_skill_count)
            
            # Human-readable explanation for this resume's score breakdown.
            score_explanation = (
                f"Semantic Score: {semantic_score:.1f}%, "
                f"Skill Score: {skill_match_score:.1f}%, "
                f"Context Score: {keyword_context_score:.1f}%, "
                f"Final Score: {final_score:.1f}%"
            )
            if skill_match_score < 30.0 and semantic_score < 45.0:
                score_explanation += " (hard gating applied: low skills and low semantic match)"
            elif skill_match_score > 70.0 and semantic_score < 50.0:
                score_explanation += " (skill-stuffing penalty applied)"
            
            results.append(
                {
                    "name": name,
                    "score": final_score,  # Final hybrid score for display
                    "semantic_score": semantic_score,
                    "skill_match_score": skill_match_score,
                    "keyword_context_score": keyword_context_score,
                    "matched_skill_count": matched_skill_count,
                    "category": category,
                    "emoji": emoji,
                    "recommendation": recommendation,
                    "matched_skills": sorted(matched_skills),
                    "missing_skills": sorted(missing_skills),
                    "score_explanation": score_explanation,
                }
            )

        # Sort by final hybrid score (descending) and add rank
        results.sort(key=lambda x: x["score"], reverse=True)
        for idx, result in enumerate(results, start=1):
            result["rank"] = idx

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
                # This explains how the system combines semantic similarity, skills, and experience
                with st.expander("ℹ️ How this decision was made"):
                    st.markdown("**The system uses a hybrid ATS-style evaluation approach:**")
                    st.markdown("""
                    • **Semantic similarity** (Sentence Transformers) measures how closely the resume meaning matches the job description  
                    • **Skill matching** verifies required technical skills from the job description  
                    • **Experience keywords** (projects, internships, built, developed, etc.) provide additional context weighting
                    """)
                    st.markdown("---")
                    st.markdown("**Decision rules:**")
                    st.markdown("""
                    • **High Match:** Final hybrid score ≥ 65  
                    • **Medium Match:** 45 ≤ Final hybrid score < 65  
                    • **Low Match:** Final hybrid score < 45
                    """)
                    st.markdown("---")
                    st.caption("This hybrid approach reduces false rejections and reflects real-world ATS shortlisting behavior.")
                    
                    # Show specific values for this candidate
                    st.markdown("---")
                    st.markdown("**This candidate's metrics:**")
                    st.write(f"• Final Hybrid Score: {result['score']:.2f}%")
                    st.write(f"• Matched Skills: {matched_count} skills")
                    st.write(f"• Final Category: {result['category']}")
                    st.write(f"• Recommendation: {result['recommendation']}")
            
            st.markdown("---")


def display_ranked_table(results: List[Dict]) -> None:
    """
    ATS-style multi-candidate ranking view (for multiple uploaded resumes):
    - Highlight Top 3 candidates
    - Show a ranked table of all candidates
    
    IMPORTANT: This function does not recompute scores; it only formats the
    already-computed `results` list returned by `process_resumes`.
    """
    import pandas as pd

    if not results:
        st.info("No results to display.")
        return

    # ---- Top 3 highlight section ----
    st.markdown("---")
    st.markdown("## 🔥 Top 3 Candidates")

    top_n = min(3, len(results))
    for r in results[:top_n]:
        category_display = f"{r.get('emoji', '')} {r.get('category', '')}".strip()
        text = (
            f"**#{r.get('rank', '-')}. {r.get('name', 'Unknown')}**  \n"
            f"Final Score: **{float(r.get('score', 0.0)):.2f}%**  \n"
            f"Category: **{category_display}**"
        )

        # Highlight based on final category (recruiter-friendly)
        if r.get("category") == "High Match":
            st.success(text)
        elif r.get("category") == "Medium Match":
            st.warning(text)
        else:
            st.error(text)

    # ---- Full ranked table ----
    st.markdown("---")
    st.markdown("## 📋 All Candidates (Ranked)")

    # Build the structured list requested
    table_rows: list[dict] = []
    for r in results:
        table_rows.append(
            {
                "Rank": r.get("rank"),
                "name": r.get("name"),
                "semantic": round(float(r.get("semantic_score", 0.0)), 2),
                "skill": round(float(r.get("skill_match_score", 0.0)), 2),
                "context": round(float(r.get("keyword_context_score", 0.0)), 2),
                "final": round(float(r.get("score", 0.0)), 2),
                "category": f"{r.get('emoji', '')} {r.get('category', '')}".strip(),
            }
        )

    df = pd.DataFrame(table_rows).sort_values("Rank", ascending=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


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
        - **Semantic Similarity** (Sentence Transformers: `all-MiniLM-L6-v2`) to capture meaning beyond exact keywords
        - **Skill Keyword Matching** to verify required skills from the job description
        - **Experience / Context Signals** (projects, internships, built, developed, etc.) for practical relevance
        - **Hybrid ATS Score** (weighted combination + gating + anti-skill-stuffing rules)
        
        Notes:
        - Supports **multiple PDF resumes** and ranks candidates like an ATS
        - The semantic model is loaded once using caching for **Streamlit Cloud efficiency**
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more resume PDFs"
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
                st.session_state.last_uploaded_count = len(uploaded_files)
                # If only 1 resume uploaded, keep the current detailed behavior
                if len(uploaded_files) == 1:
                    display_results(results)
                else:
                    display_ranked_table(results)
    
    # Display previous results if available
    if st.session_state.results and not analyze_button:
        # Preserve behavior across reruns
        if st.session_state.last_uploaded_count == 1:
            display_results(st.session_state.results)
        else:
            display_ranked_table(st.session_state.results)
    
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

