import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from preproc import load_and_preprocess_data
from model import make_embeddings

# Cache the preprocessing step to run only once
@st.cache_data
def get_cleaned_data(file_path):
    return load_and_preprocess_data(file_path)

# Cache embedding generation to run only once
@st.cache_resource
def get_embeddings(data):
    return make_embeddings(data)

# Runs concurrently during app setup
file = get_cleaned_data("Course_info.csv")
df = pd.read_csv(file)  
normalized_embeddings = get_embeddings(df)

# Recommendation function
def recommend_courses(course_index, embeddings, data, method="cosine", top_n=5):
    if method == "Cosine Similarity":
        input_embedding = embeddings[course_index]
        similarities = cosine_similarity(input_embedding.reshape(1, -1), embeddings).flatten()
        similar_indices = similarities.argsort()[-(top_n + 1):-1][::-1]
    elif method == "K Nearest Neigbours":
        knn = NearestNeighbors(n_neighbors=top_n + 1, metric="euclidean")  # +1 to include the input course
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings[course_index].reshape(1, -1))
        similar_indices = indices.flatten()[1:]  

    recommended_courses = data.iloc[similar_indices]
    return recommended_courses



# UI Starts here
st.title("Course Recommendation System")
st.sidebar.header("Select a Course")

# Dropdown to select a course
course_titles = ["Select a course"] + df["title"].tolist()
selected_course = st.sidebar.selectbox("Choose a course you have completed:", course_titles)

# Choose recommendation method
method = st.sidebar.radio("Recommendation Method", ("Cosine Similarity", "K Nearest Neigbours"))

# Show recommendations only when a valid course is selected
if selected_course and selected_course != "Select a course":
    course_index = course_titles.index(selected_course) - 1  # Adjust index coz of placeholder
    st.write(f"### Recommended courses similar to: **{selected_course}**")
    
    recommended_courses = recommend_courses(course_index, normalized_embeddings, df, method=method)
    
    # Display recommendations
    for _, course in recommended_courses.iterrows():
        st.write(f"**{course['title']}**")
        st.write(f"Category: {course['category']} | Subcategory: {course['subcategory']}")
        st.write(f"Price: ${round(course['price'], 2)} | Rating: {round(course['avg_rating'], 2)}")
        st.write("---")
else:
    st.write("Please select a course to get recommendations.")
