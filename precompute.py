from itertools import chain
from sklearn.model_selection import train_test_split
import pandas as pd

from data_loader import load_data, get_dataset
from preprocessing import clean_text, rating_to_sentiment, map_sentiment_labels, calc_pos_neg_percentage
from pros_cons import process_pros_cons
from embeddings import get_embeddings, extract_keywords, extract_keywords_by_row
from aspects import normalize_phrases, cluster_aspects, map_to_aspects
from sentiment_model import build_nn, train_nn_model

path = get_dataset()
df = load_data(path)

df['sentiment'] = df['reviews_rating'].apply(rating_to_sentiment)
df['reviews_text_clean'] = df['reviews_text'].apply(clean_text)
df['text_for_model'] = df['reviews_title'].fillna('') + ' ' + df['reviews_text_clean']

embeddings = get_embeddings(df['text_for_model'].to_list())

X_train, X_test, y_train, y_test = train_test_split(embeddings, df['sentiment'], test_size=0.2, random_state=123)

model_nn = build_nn(embeddings.shape[1])
model_nn = train_nn_model(model_nn, X_train, y_train)

model_nn.save("sentiment_model.h5")

loss, acc = model_nn.evaluate(X_test, y_test)

print(f"Test Accuracy: {acc:.4f}")

df = process_pros_cons(df)

df = map_sentiment_labels(df)

df['key_phrases'] = df.apply(extract_keywords_by_row, axis=1)

df['normalized_phrases'] = df['key_phrases'].apply(normalize_phrases)

all_phrases = list(chain.from_iterable(df['normalized_phrases']))

phrase_to_aspect = cluster_aspects(all_phrases)

df['aspects'] = df['normalized_phrases'].apply(lambda x: map_to_aspects(phrase_to_aspect, x))

sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['sentiment_label'] = df['sentiment'].map(sentiment_labels)

df.drop(['categories', 'brand', 'dateAdded', 'dateUpdated', 'manufacturerNumber', 'reviews_dateSeen', 'reviews_text', 'reviews_title', 'sentiment', 'text_for_model', 'pros', 'cons', 'reviews_text_clean', 'final_pros', 'final_cons', 'key_phrases', 'normalized_phrases'], axis=1, inplace=True)

df = df.explode('aspects')

df['sentiment_label'] = df['sentiment_label'].astype(str).str.strip()
df['sentiment_label'] = pd.Categorical(df['sentiment_label'], categories=['Negative','Neutral','Positive'])
df['reviews_date'] = pd.to_datetime(df['reviews_date'], format='ISO8601', utc=True, errors='coerce')

df.to_parquet("processed_reviews.parquet")

print('DONE! Saved processed_reviews.parquet and sentiment_model.h5')

df = df.drop('aspects', axis=1)
df.drop_duplicates(inplace=True)
original_df = pd.read_csv(path)
original_df = original_df[['id', 'name']]
df = df.merge(original_df, on='id', how='left')

df_categories = df.groupby(['primaryCategories', 'sentiment_label']).size().reset_index(name='count')
df_products = df.groupby(['name', 'sentiment_label']).size().reset_index(name='count')

df_categories = df_categories.pivot_table(index='primaryCategories', columns='sentiment_label', values='count', fill_value=0).reset_index()
df_products = df_products.pivot_table(index='name', columns='sentiment_label', values='count', fill_value=0).reset_index()

df_categories['Pos/Neg Percentage'] = df_categories.apply(lambda row: calc_pos_neg_percentage(row, 'Positive', 'Negative'), axis=1)
df_categories['Pos/Neu Percentage'] = df_categories.apply(lambda row: calc_pos_neg_percentage(row, 'Positive', 'Neutral'), axis=1)
df_categories['Neg/Neu Percentage'] = df_categories.apply(lambda row: calc_pos_neg_percentage(row, 'Negative', 'Neutral'), axis=1)
df_categories['Pos/All Percentage'] = df_categories.apply(lambda row: calc_pos_neg_percentage(row, 'Positive'), axis=1)
df_categories['Neg/All Percentage'] = df_categories.apply(lambda row: calc_pos_neg_percentage(row, 'Negative'), axis=1)


df_products['Pos/Neg Percentage'] = df_products.apply(lambda row: calc_pos_neg_percentage(row, 'Positive', 'Negative'), axis=1)
df_products['Pos/Neu Percentage'] = df_products.apply(lambda row: calc_pos_neg_percentage(row, 'Positive', 'Neutral'), axis=1)
df_products['Neg/Neu Percentage'] = df_products.apply(lambda row: calc_pos_neg_percentage(row, 'Negative', 'Neutral'), axis=1)
df_products['Pos/All Percentage'] = df_products.apply(lambda row: calc_pos_neg_percentage(row, 'Positive'), axis=1)
df_products['Neg/All Percentage'] = df_categories.apply(lambda row: calc_pos_neg_percentage(row, 'Negative'), axis=1)


df_categories.to_parquet('processed_reviews_categories_count.parquet')
df_products.to_parquet('processed_reviews_products_count.parquet')

print('DONE! Saved processed_reviews_categories_count.parquet and processed_reviews_products_count.parquet')