"""
Quick test to verify lemmatization is working
"""
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    def lemmatize_tokenizer(text):
        doc = nlp(text.lower())
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
        return lemmas
    
    # Test cases
    test_cases = [
        ("building", "built"),
        ("modeling", "model"),
        ("skills", "skilled"),
        ("Python developer", "Python developers"),
        ("Machine Learning", "machine learning"),
        ("data analysis", "analyzing data"),
    ]
    
    print("Testing lemmatization:")
    print("=" * 60)
    for text1, text2 in test_cases:
        lemmas1 = lemmatize_tokenizer(text1)
        lemmas2 = lemmatize_tokenizer(text2)
        overlap = set(lemmas1) & set(lemmas2)
        print(f"\nText 1: '{text1}' -> {lemmas1}")
        print(f"Text 2: '{text2}' -> {lemmas2}")
        print(f"Overlap: {overlap}")
        if overlap:
            print("✅ MATCH")
        else:
            print("❌ NO MATCH")
    
    print("\n" + "=" * 60)
    print("If you see matches above, lemmatization is working!")
    
except Exception as e:
    print(f"Error: {e}")
    print("spaCy lemmatization not available")



