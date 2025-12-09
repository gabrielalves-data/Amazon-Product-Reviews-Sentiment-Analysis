import streamlit as st
import tensorflow as tf
import pandas as pd
import os, sys, traceback

from visualizations import (reviews_by_aspect, reviews_over_time, reviews_by_day_month, reviews_by_dof_and_sentiment,
                            reviews_table, reviews_percentage_diff)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.h5')
DATA_PATH = os.path.join(BASE_DIR, 'processed_reviews.parquet')

st.set_page_config(layout='wide')
st.title('Amazon Reviews Dashboard')

st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    font-size: 1.2rem !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab"] [aria-selected="true"] {
    font-size: 1.3rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_preprocessed_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_sentiment_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    df = load_preprocessed_data()
except Exception as e:
    st.error('Failed to load data')
    st.text(traceback.format_exc)
    sys.exit(1)
    
try: 
    model_nn = load_sentiment_model()
except Exception as e:
    st.error('Failed to load model')
    st.text(traceback.format_exc())
    sys.exit(1)

tab1, tab2 = st.tabs(['📊 Reviews', '📈 Time Analysis'])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.header('Number of Reviews by Aspect')
        st.plotly_chart(reviews_by_aspect(df))

    with col2:
        st.header('Number of Reviews By Sentiment and Product')
        st.plotly_chart(reviews_table(df), use_container_width=True)

    st.header('Reviews Percentage Difference')
    products_categories = st.radio('Choose Products or Categories', ['Products', 'Categories'])
    fig = reviews_percentage_diff(products_categories)
    if fig:
        st.plotly_chart(fig)

with tab2:
    st.header('Number of Reviews Over Time')
    st.plotly_chart(reviews_over_time(df))

    st.header('Number of Reviews By Day and Month')
    st.plotly_chart(reviews_by_day_month(df))

    st.header('Number of Reviews By Sentiment and Day of Week')
    st.plotly_chart(reviews_by_dof_and_sentiment(df))


st.markdown("""
<style>
.vertical-footer {
    position: fixed;
    top: 50%;
    left: 10px; /* adjust if needed */
    transform: rotate(-90deg);
    transform-origin: left top;
    font-size: 0.9rem;
    font-weight: 600;
    color: #777;
    z-index: 10000;
    opacity: 0.7;
    cursor: default;
}
.vertical-footer a {
    color: inherit;
    text-decoration: none;
    font-weight: 700;
}
.vertical-footer a:hover {
    color: #555;
}
</style>

<div class="vertical-footer">
    Project by <a href="https://profile.gabrieldatascience.com/" target="_blank">Gabriel Alves</a>
</div>
""", unsafe_allow_html=True)


