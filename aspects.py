import spacy
from sklearn.cluster import KMeans
from itertools import chain
import streamlit as st

from embeddings import load_embed_model

embed_model = load_embed_model()

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

@st.cache_resource
def get_kmeans_model(num_clusters, embeddings):
    model = KMeans(n_clusters=num_clusters, random_state=123)
    model.fit(embeddings)

    return model


cluster_to_aspect = {
    0:'Brand',
    1:'Price/Value',
    2:'Product Satisfaction / Overall Quality',
    3:'Customer Emotion / Sentiment',
    4:'Battery Life / Longevoty',
    5:'Usability / Experience',
    6:'General Sentiment / Product Opinion',
    7:'Battery / Power',
    8:'Performance / Functionality',
    9:'Shipping / Purchase Experience',
    10:'Retailer / Brand Context',
    11:'Quality / Brand Preception'
}

def normalize_phrases(phrases):
    clean_phrases = []
    for phrase in phrases:
        doc = nlp(phrase.lower())
        lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        clean = ' '.join(lemmas)
        if clean:
            clean_phrases.append(clean)

    return clean_phrases


def cluster_aspects(all_phrases, num_clusters=12):
    phrase_embeddings = embed_model.encode(all_phrases, normalize_embeddings=True)
    cluster_model = get_kmeans_model(num_clusters, phrase_embeddings)
    cluster_assignment = cluster_model.labels_

    clusters = {}

    for phrase, cluster_id in zip(all_phrases, cluster_assignment):
        clusters.setdefault(cluster_id, []).append(phrase)

    phrase_to_aspect = {phrase: cluster_to_aspect[cluster_id] for phrase, cluster_id in zip(all_phrases, cluster_assignment)}

    return phrase_to_aspect


def map_to_aspects(phrase_to_aspect, phrases):
    aspects = [phrase_to_aspect.get(p, 'miscellaneous') for p in phrases]
    
    return aspects