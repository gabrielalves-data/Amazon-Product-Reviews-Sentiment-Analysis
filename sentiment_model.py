from tensorflow.keras import layers, models
import streamlit as st

@st.cache_resource
def build_nn(input_dim):
    model_nn = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model_nn

@st.cache_resource
def train_nn_model(model, X_train, y_train):
    model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=128, verbose=0)

    return model