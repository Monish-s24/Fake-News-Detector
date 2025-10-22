# Fake News Detection (TF-IDF + Naive Bayes)

## Overview
TF-IDF vectorization + Multinomial Naive Bayes classifier trained on the "Fake and Real News Dataset" (Kaggle).

## How to run (Colab)
1. Open the provided `notebook.ipynb` in Google Colab.
2. (Optional) Upload `kaggle.json` to download dataset automatically using Kaggle API OR upload `True.csv` and `Fake.csv` manually to Colab.
3. Run cells sequentially. Model artifacts will be saved as `.pkl` files.

## Files
- `notebook.ipynb` — full notebook
- `naive_bayes_fake_news_model.pkl` — trained model
- `tfidf_vectorizer.pkl` — saved TF-IDF vectorizer

## Requirements
See `requirements.txt`
