import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gzip

# ✅ Load dataset (corrected path)
print("📁 Loading dataset...")
df = pd.read_csv("/Users/tanushree/Desktop/Movie Recommendation/data/movies.csv")
df = df.head(10000).copy()  
print(f"✅ Using full dataset with {len(df)} movies.")

# ✅ Preprocessing
df['genres'] = df['genres'].fillna('')
df['title_lower'] = df['title'].str.lower()

# ✅ TF-IDF vectorization
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = vectorizer.fit_transform(df['genres'])

# ✅ Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ Package everything into a model dictionary
model_data = {
    "df": df,
    "cosine_sim": cosine_sim
}

# ✅ Save model to 'models/model.pkl'
os.makedirs("models", exist_ok=True)
with gzip.open("models/model.pkl.gz", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model trained and saved to models/model.pkl")
