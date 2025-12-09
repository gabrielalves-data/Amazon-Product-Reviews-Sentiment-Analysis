import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import streamlit as st

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download("vader_lexicon")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
sia = SentimentIntensityAnalyzer()

def extract_pros_cons(text):
    doc = nlp(text)
    pros, cons = [], []

    for sent in doc.sents:
        score = sia.polarity_scores(sent.text)['compound']
        if score >= 0.3:
            pros.append(sent.text.strip())
        elif score <= -0.3:
            cons.append(sent.text.strip())
            
    return ' '.join(pros), ' '.join(cons)


def select_pros_cons(row):
    if row['sentiment'] == 2:
        return row['pros'], ''
    elif row['sentiment'] == 0:
        return '', row['cons']
    else:
        return row['pros'], row['cons']
        

@st.cache_data
def process_pros_cons(df):
    df[['pros', 'cons']] = df['reviews_text_clean'].apply(lambda x: pd.Series(extract_pros_cons(x)))
    df[['final_pros', 'final_cons']] = df.apply(select_pros_cons, axis=1, result_type='expand')

    return df