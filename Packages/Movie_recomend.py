import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time # For simulating a loading state

# --- 1. CONFIGURATION & WIDGET SETUP ---
# Configure Streamlit page settings
st.set_page_config(
    page_title="Pro Movie Recommender 🎬",
    page_icon="🎥",
    layout="centered", # 'wide' or 'centered' for responsiveness
    initial_sidebar_state="collapsed"
)

# Apply custom styling (optional but highly recommended for a professional look)
st.markdown("""
<style>
    /* General styling for a professional look */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #1f2937; /* Dark text */
    }
    /* Header/Title styling */
    .st-emotion-cache-p5msec {
        color: #2e86c1; /* Professional blue for the title */
        font-weight: 700;
        text-align: center;
    }
    /* Button styling */
    .stButton>button {
        background-color: #3498db; /* Blue button */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2e86c1; /* Darker blue on hover */
    }
    /* Recommendation List styling */
    .recommendation-list {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. DATA LOADING AND MODEL ARTIFACTS ---
@st.cache_data # Cache the data loading and expensive calculations
def load_data_and_model():
    """Loads data, prepares features, and computes similarity matrix."""
    try:
        # Load Dataset
       file_path = os.path.join(
    os.path.dirname(__file__), 
    'IMDB-Movie-Data (1).csv'
)

# Load the Dataset using the reliable file_path
movie_df = pd.read_csv(file_path)
        # Clean Data: Fill missing values in text columns
        movie_df['Genre'] = movie_df['Genre'].fillna('')
        movie_df['Description'] = movie_df['Description'].fillna('')
        movie_df['Actors'] = movie_df['Actors'].fillna('')
        movie_df['Director'] = movie_df['Director'].fillna('')

        # Combine features for content-based recommendation
        movie_df['combined_features'] = (
            movie_df['Genre'] + " " +
            movie_df['Description'] + " " +
            movie_df['Actors'] + " " +
            movie_df['Director']
        )

        # Vectorize the text using TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movie_df['combined_features'])

        # Compute cosine similarity
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Build title index mapping (Series for quick lookup)
        indices = pd.Series(movie_df.index, index=movie_df['Title']).drop_duplicates()
        
        return movie_df, cosine_sim_matrix, indices
    except FileNotFoundError:
        st.error("🚨 Error: 'IMDB-Movie-Data (1).csv' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.stop()


movie, cosine_sim, indices = load_data_and_model()


# --- 3. RECOMMENDATION FUNCTION ---
def get_recommendations(title, cosine_sim_matrix=cosine_sim):
    """Generates top 5 movie recommendations based on cosine similarity."""
    if title not in indices:
        return None # Return None for not found

    idx = indices[title]
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the top 5 most similar movies (excluding itself)
    sim_scores = sim_scores[1:6]  
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the titles
    return movie['Title'].iloc[movie_indices].tolist()


# --- 4. STREAMLIT APPLICATION LAYOUT ---

st.title("🎬 Professional Movie Recommender System")

st.markdown("""
Welcome to the Content-Based Recommender. Select a movie, and the system will suggest 
**5 similar movies** based on their **Genre, Description, Actors, and Director** using 
**TF-IDF** vectorization and **Cosine Similarity**.
""")
st.markdown("---")


# Select Box for Movie Input (Mobile-Friendly Dropdown)
# The selectbox is highly responsive and good for mobile use
movie_list = sorted(movie['Title'].unique().tolist())
selected_movie = st.selectbox(
    "👉 **Select a Movie:**",
    movie_list,
    index=movie_list.index("Prometheus") if "Prometheus" in movie_list else 0, # Set a default
    placeholder="Start typing or select a movie...",
    key="movie_select"
)

# Button to trigger recommendation
if st.button("✨ Get Recommendations", use_container_width=True):
    # Use a spinner to show a loading state, enhancing user experience
    with st.spinner(f"Searching for recommendations for **{selected_movie}**..."):
        # Simulate a network delay (optional, but makes the spinner noticeable)
        time.sleep(1) 
        
        # Get recommendations
        recommendations = get_recommendations(selected_movie)

        if recommendations is None:
            st.error(f"❌ Movie **'{selected_movie}'** not found in database. Please try another.")
        elif recommendations:
            st.success(f"✅ Found **5** highly similar movies!")
            
            # Display results in a clean, professional format
            st.markdown(f'<div class="recommendation-list"><h3>Top Recommendations for "{selected_movie}":</h3>', unsafe_allow_html=True)
            
            # Use columns and markdown list for a clean look
            col1, col2 = st.columns(2)
            
            # Split recommendations into two columns for better window/desktop view
            # While maintaining a good mobile flow (Streamlit stacks columns on mobile)
            rec1 = recommendations[:3]
            rec2 = recommendations[3:]
            
            with col1:
                st.markdown("##### 🍿 Movies 1-3:")
                st.markdown('\n'.join(f'* **{title}**' for title in rec1))
            
            with col2:
                st.markdown("##### 🍿 Movies 4-5:")
                st.markdown('\n'.join(f'* **{title}**' for title in rec2))
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No similar movies found. Check data or algorithm parameters.")

st.markdown("---")
st.caption("Developed by **Data Scientist Ngama Jude Chinedu** ⚡.")

