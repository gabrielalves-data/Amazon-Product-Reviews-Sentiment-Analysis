from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import streamlit as st

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_keybert(_model):
    return KeyBERT(_model)

embed_model = load_embed_model()
keybert = load_keybert(embed_model)

def get_embeddings(texts):
    return embed_model.encode(texts, normalize_embeddings=False, show_progress_bar=True)

def extract_keywords(text, top_n=3):
    keywords = keybert.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        use_maxsum=True,
        top_n=top_n
    )
    
    return [kb[0] for kb in keywords]

def extract_keywords_by_row(row):
    sentiment = row.get('sentiment')
    pros = row.get('pros', '')
    cons = row.get('cons', '')

    if sentiment == 2:
        return extract_keywords(pros)
    elif sentiment == 0:
        return extract_keywords(cons)
    else:
        return extract_keywords(pros) + extract_keywords(cons)