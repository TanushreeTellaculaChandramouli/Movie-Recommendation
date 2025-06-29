import streamlit as st
import pandas as pd
import pickle
import gzip

# Load model
MODEL_ID = "1l56Zhuf8ZzBW0HBI2VtpkGlqnJfS30uW"
MODEL_URL = f"https://drive.google.com/file/d/1l56Zhuf8ZzBW0HBI2VtpkGlqnJfS30uW/view?usp=sharing"
MODEL_PATH = "models/model.pkl.gz"


try:
    with gzip.open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    df = model_data["df"]
    cosine_sim = model_data["cosine_sim"]
except Exception as e:
    st.error(f"❌ Model could not be loaded: {e}")
    st.stop()

# Recommendation function
def recommend_movies(title):
    title_lower = title.lower()
    if title_lower not in df['title_lower'].values:
        return None
    idx = df[df['title_lower'] == title_lower].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genres']]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.caption("Get top 5 similar movies based on genre similarity")

# Autocomplete box to select movie
movie_input = st.selectbox(
    "🎥 Select a movie you like:",
    options=sorted(df['title'].tolist()),
    placeholder="Start typing a movie title..."
)

if movie_input:
    # Show the genre of the selected movie
    selected_genre = df[df['title'] == movie_input]['genres'].values[0]
    st.info(f"📚 Genre of **{movie_input}**: `{selected_genre}`")

    # Get recommendations
    recommendations = recommend_movies(movie_input)
    if recommendations is not None:
        st.success(f"🎯 Top 5 movies similar to **{movie_input}**:")
        for i, row in recommendations.iterrows():
            st.write(f"**{i+1}. {row['title']}** — *{row['genres']}*")
    else:
        st.error(f"❌ Movie '{movie_input}' not found in the dataset.")
