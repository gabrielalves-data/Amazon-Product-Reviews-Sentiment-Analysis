# Amazon Reviews Sentiment and Aspect Analysis

This project analyzes Amazon product reviews to extract customer sentiment and identify key aspects mentioned in the reviews. We leverage a neural network in combination with a BERT-based model to classify sentiment and extract the principal features highlighted by customers.

The insights from this analysis allow us to:
* Identify peak periods when users write reviews (by date and day of the week).  
* Determine which product aspects are most frequently mentioned.  
* Highlight strengths and weaknesses in products based on customer feedback.  

These insights can help the company:
* Improve products by addressing aspects with negative sentiment.  
* Understand which product qualities are most appreciated by customers.  
* Make data-driven decisions to optimize product offerings and customer satisfaction.

# Methodology

This document outlines the methodology used for analyzing customer reviews, extracting sentiments, key phrases, and clustering them into relevant aspects.

## 1. Text Embedding
- We use the `SentenceTransformer` model `"all-MiniLM-L6-v2"` to convert review texts into vector embeddings.
- Embeddings are generated for all reviews in the dataset using:
```python
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embed_model.encode(df['text_for_model'].tolist(), show_progress_bar=True)
```

## 2. Train/Test Split
The dataset is split into training and test sets (80/20) for sentiment classification:
```python
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df['sentiment'], test_size=0.2, random_state=123
)
```

## 3. Neural Network Sentiment Classification
A feedforward neural network is used to predict sentiment:
* Input layer matches embedding size.
* Two hidden layers with ReLU activations and a dropout layer (0.3).
* Output layer with 3 neurons (softmax for sentiment classes: negative, neutral, positive).

Model compilation and training:
```python
model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_nn.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
```

Model evaluation is performed on the test set:
```python
loss, acc = model_nn.evaluate(X_test, y_test)
```

## 4. Pros and Cons Extraction
VADER Sentiment Analyzer and spaCy are used to split reviews into sentences and classify them:
* Sentences with compound score ≥ 0.3 → pros
* Sentences with compound score ≤ -0.3 → cons
* Pros and cons are then filtered based on the predicted sentiment of the review.
```python
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")
```

## 5. Key Phrase Extraction
KeyBERT is used to extract the top 3 key phrases from pros and/or cons for each review:
* Phrases are normalized by lowercasing, lemmatizing, and removing stopwords.
```python
keybert = KeyBERT(embed_model)
```

## 6. Phrase Embedding and Clustering
* All normalized phrases are embedded using the same SentenceTransformer model.
* KMeans clustering groups phrases into 12 clusters representing different product aspects:
```python
cluster_model = KMeans(n_clusters=12, random_state=123)
cluster_model.fit(phrase_embeddings)
```

## 7. Mapping Clusters to Aspects
* Each cluster is manually assigned to an aspect, e.g., "Brand", "Price/Value", "Customer Emotion / Sentiment", etc.
* Every key phrase is then mapped to its corresponding aspect:
```python
phrase_to_aspect = {phrase: cluster_to_aspect[cluster_id] for phrase, cluster_id in zip(all_phrases, cluster_assignment)}
```

## Summary
This methodology combines:
* Sentence embeddings for semantic understanding.
* Neural networks for sentiment classification.
* Rule-based sentiment extraction (VADER) for pros/cons.
* Key phrase extraction (KeyBERT) and normalization.
* Clustering to group phrases into actionable product aspects.