import pandas as pd
import re
import streamlit as st

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 .,!?]', '', text)
    
    return text


def rating_to_sentiment(rating):
    if rating <= 2:
        return 0 ## Negative
    elif rating == 3:
        return 1 ## Neutral
    else:
        return 2 ## Positive
    
    
@st.cache_data
def map_sentiment_labels(df):
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df['sentiment_label'] = df['sentiment'].map(sentiment_labels)

    return df


@st.cache_data
def calc_pos_neg_percentage(row, numerator_col, denominator_col=None):
    numerator = row[numerator_col]

    if denominator_col == None:
        all_others = row['Positive'] + row['Neutral'] + row['Negative'] - numerator
        
        if all_others == 0 and numerator > 0:
            return '100.0% (all positive reviews)'
        
        elif all_others == 0 and numerator == 0:
            return '0.0% (no reviews)'
        else:
            pct = ((numerator - all_others) / all_others) * 100
            sign = '+' if pct >= 0 else ''

            return f'{sign}{pct:.1f}%'

    else:
        denominator = row[denominator_col]

        if denominator == 0 and numerator > 0:
            return f'100.0% (no {denominator_col.lower()} reviews)'
        elif denominator == 0 and numerator == 0:
            return '0.0% (no reviews)'
        else:
            pct = ((numerator - denominator) / denominator) * 100
            sign = '+' if pct >= 0 else ''

            return f'{sign}{pct:.1f}%'