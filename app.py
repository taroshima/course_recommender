import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load data and embeddings
# Replace these with your actual data and embeddings
df = pd.read_csv("cleaned_courses.csv")  # Replace with your dataset path
embeddings = np.load("embeddings.npy")  # Replace with your precomputed embeddings

# Normalize embeddings
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Function to recommend courses
def recommend_courses(course_index, embeddings, data, method="cosine", top_n=5):
    """
    Recommends courses similar to the given course.
    :param course_index: The index of the input course.
    :param embeddings: The feature set for all courses.
    :param data: The original dataset containing course information.
    :param method: Similarity method ('cosine' or 'knn').
    :param top_n: Number of recommendations to return.
    :return: DataFrame containing the top N recommended courses.
    """
    if method == "Cosine Similarity":
        input_embedding = embeddings[course_index]
        similarities = cosine_similarity(input_embedding.reshape(1, -1), embeddings).flatten()
        similar_indices = similarities.argsort()[-(top_n + 1):-1][::-1]
    elif method == "K Nearest Neigbours":
        knn = NearestNeighbors(n_neighbors=top_n + 1, metric="euclidean")  # +1 to include the input course
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings[course_index].reshape(1, -1))
        similar_indices = indices.flatten()[1:]  # Exclude the input course itself

    recommended_courses = data.iloc[similar_indices]
    return recommended_courses

# Streamlit UI
st.title("Course Recommendation System")
st.sidebar.header("Select a Course")

# Dropdown to select a course
course_titles = ["Select a course"] + df["title"].tolist()
selected_course = st.sidebar.selectbox("Choose a course you have completed:", course_titles)

# Choose recommendation method
method = st.sidebar.radio("Recommendation Method", ("Cosine Similarity", "K Nearest Neigbours"))

# Show recommendations only when a valid course is selected
if selected_course and selected_course != "Select a course":
    course_index = course_titles.index(selected_course) - 1  # Adjust index due to placeholder
    st.write(f"### Recommended courses similar to: **{selected_course}**")
    
    
    
    # Get recommendations
    recommended_courses = recommend_courses(course_index, normalized_embeddings, df, method=method)
    
    # Display recommendations
    for _, course in recommended_courses.iterrows():
        st.write(f"**{course['title']}**")
        st.write(f"Category: {course['category']} | Subcategory: {course['subcategory']}")
        st.write(f"Price: ${round(course['price'], 2)} | Rating: {round(course['avg_rating'], 2)}")
        st.write("---")

    st.write("Please select a course to get recommendations.")
