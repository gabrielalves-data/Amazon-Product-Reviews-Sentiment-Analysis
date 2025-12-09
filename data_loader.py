import pandas as pd
import kagglehub
import os
import streamlit as st

DATASET_NAME = "datafiniti/consumer-reviews-of-amazon-products"
CSV_FILENAME = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"

@st.cache_data
def get_dataset():
    if os.path.exists(CSV_FILENAME):
        return os.path.abspath(CSV_FILENAME)

    path = kagglehub.dataset_download(DATASET_NAME)
    full_path = os.path.join(path, CSV_FILENAME)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset not found after download: {full_path}")
    
    return full_path


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(['name', 'asins', 'imageURLs', 'keys', 'manufacturer', 'reviews.didPurchase',
         'reviews.doRecommend', 'reviews.id', 'reviews.numHelpful', 'reviews.sourceURLs',
         'reviews.username', 'sourceURLs'], axis=1, errors='ignore')
    df.columns = df.columns.str.replace('.', '_', regex=False)

    return df