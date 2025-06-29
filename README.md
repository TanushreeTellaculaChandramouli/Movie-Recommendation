# 🎬 Movie Recommendation System

A simple Streamlit web app that recommends **5 movies similar** to the one you enter — based on **genre similarity using TF-IDF** and **cosine similarity**.

---

## 🚀 Features

- Input a movie title (case-insensitive)
- Recommends 5 similar movies from the dataset
- Uses TF-IDF to vectorize genres
- Built with Python, Streamlit, Pandas, Scikit-learn

---

## 🗃️ Dataset

The app uses a `movies.csv` file which contains movie titles and their corresponding genres. Make sure this file is in the same folder as `app.py`.

---

## 📦 Installation

1. Clone or download this repository.

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
