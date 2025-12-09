![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-orange?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

# 📊 Amazon Reviews Dashboard

A **Streamlit dashboard** for visualizing Amazon product reviews with sentiment analysis powered by TensorFlow.  
This dashboard allows users to explore review trends over time, review distributions by aspect, sentiment, and product categories, and see differences in review percentages.

---

## 🧠 Features

- **Reviews by Aspect** – Visualize review counts grouped by product aspects.
- **Reviews by Sentiment & Product** – Compare positive, neutral, and negative reviews per product category.
- **Reviews Over Time** – Analyze how reviews evolve daily, monthly, or yearly.
- **Reviews by Day of Month / Day of Week** – Explore review patterns across calendar days and weekdays.
- **Percentage Differences** – Compare review counts between products or categories.
- **Interactive Filtering** – Filter by sentiment or product/category in real time.

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/gabriel-data/Amazon-Product-Reviews-Sentiment-Analysis.git
cd Amazon-Product-Reviews-Sentiment-Analysis
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Usage**
```bash
streamlit run app.py
```

##  🧱 Project Structure
```bash
amazon-reviews-dashboard/
│
├── app.py                                           # Main Streamlit app
├── aspects.py                                       # Aspects Processing functions
├── data_loader.py                                   # Data Loading functions
├── embeddings.py                                    # Processing Embeddings functions
├── precompute.py                                    # Precompute functions
├── preprocessing.py                                 # Pre-Process Data functions
├── pros_cons.py                                     # Pros and Cons Processing functions
├── sentiment_model.py                               # Sentiment Model functions
├── visualizations.py                                # Plotting functions
├── original_data.py                                 # Original Pre Loaded Data
├── processed_reviews_categories_output.py           # Pre-Loaded Data Pivoted on Categories By Sentiment functions
├── processed_reviews_products_output.py             # Pre-Loaded Data Pivoted on Products By Sentiment functions
├── processed_reviews.parquet                        # Preprocessed review data
├── sentiment_model.h5                               # Trained TensorFlow sentiment model
├── requirements.txt                                 # Python dependencies
├── .gitignore                                       # Ignored files
└── README.md                                        # This file
```

## 📚 Dependencies
Key Python libraries used:
* Streamlit - Web app framework
* Pandas - Data manipulation
* Plotly - Interactive visualizations
* TensorFlow - Sentiment analysis model
* NumPy - Numerical operations

## 🤝 Contributing
Contributions are welcome!
* Fork the repository
* Create a new branch (git checkout -b feature/my-feature)
* Make your changes
* Submit a pull request

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

## 👨‍💻 Author
Project created by **Gabriel Alves**
