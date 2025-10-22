# 📰 Fake News Detection using Machine Learning (TF-IDF + Naive Bayes)

This project aims to detect **fake or misleading news articles** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
It uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text data into numerical features and a **Multinomial Naive Bayes** classifier for binary classification between *real* and *fake* news articles.

---

## 🚀 Features

- Preprocesses text data (cleans punctuation, URLs, digits, and extra spaces)
- Uses **TF-IDF** for text vectorization  
- Trains a **Multinomial Naive Bayes** model for classification  
- Achieves **~92% accuracy** on the test dataset  
- Saves trained model and vectorizer as `.pkl` files  
- Includes confusion matrix visualization and evaluation metrics  

---

## 🗂️ Dataset

**Source:** [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

| File | Description |
|------|--------------|
| `True.csv` | Contains true (real) news articles |
| `Fake.csv` | Contains fake news articles |

Each CSV includes news **title**, **text**, and other metadata.  
We label real news as `1` and fake news as `0`.

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Environment:** Google Colab / Jupyter Notebook  
- **Libraries:**  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

---

## 📋 Project Workflow

### 1️⃣ Data Preprocessing
- Merge `True.csv` and `Fake.csv`
- Label data → `1` = Real, `0` = Fake  
- Clean text:
  - Lowercase all words  
  - Remove punctuation, digits, and URLs  
  - Strip whitespace

### 2️⃣ Feature Extraction (TF-IDF)
- Convert text into numerical features  
- Use `TfidfVectorizer(stop_words='english', max_df=0.7)`

### 3️⃣ Model Training (Naive Bayes)
- Model: `MultinomialNB()`  
- Split data: 75% training / 25% testing  
- Train on TF-IDF vectors

### 4️⃣ Evaluation
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix visualization using Seaborn

### 5️⃣ Save Model
- `naive_bayes_fake_news_model.pkl`
- `tfidf_vectorizer.pkl`

---

## 🧩 How to Run (Google Colab)

1. **Open the Notebook**  
   Upload or open the provided `notebook.ipynb` in Google Colab.

2. **Dataset Setup**  
   Option A: Upload `True.csv` and `Fake.csv` manually  
   Option B: Use Kaggle API  
   ```bash
   !pip install kaggle
   !kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
   !unzip fake-and-real-news-dataset.zip
