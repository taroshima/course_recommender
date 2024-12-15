# Course Recommendation System

This is a **Course Recommendation System** built using **Streamlit** for the user interface and a combination of machine learning techniques to recommend courses based on similarity. The system uses **neural network embeddings**, **cosine similarity**, and **K-Nearest Neighbors** to generate course recommendations.

---

## Features

1. **Preprocessing Pipeline**:
   - Cleans and preprocesses course data (e.g., handling missing values, encoding categorical data, normalizing numerical features, and tokenizing text).
   - Adds weights to specific features like categories and subcategories for better recommendation accuracy.

2. **Embedding Generation**:
   - Creates a low-dimensional representation of the course data using a neural network.
   - Normalizes embeddings to facilitate similarity computations.

3. **Recommendation Methods**:
   - **Cosine Similarity**: Measures the similarity between courses based on their embeddings.
   - **K-Nearest Neighbors (KNN)**: Finds the most similar courses using the Euclidean distance.

4. **Streamlit User Interface**:
   - Allows users to select a course and recommendation method.
   - Displays similar courses with relevant details like category, subcategory, price, and rating.

5. **Caching for Performance**:
   - Preprocessing and embedding generation steps are cached to speed up the app setup.

---

## File Structure

```bash
course_recommendation_system/
├── app.py                        # Streamlit app for the UI
├── preproc.py                    # Preprocessing functions
├── model.py                      # Functions for embedding generation
├── Course_info.csv               # Input course dataset
├── cleaned_courses.csv           # Preprocessed dataset
├── embeddings.npy                # Generated course embeddings
├── recommendation_system.ipynb   # Jupyter Notebook with original implementation
```
## How to Run the Application

1. Clone this repository:
   ```bash
   git clone https://github.com/taroshima/course_recommender
   cd course_recommendation_system
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser (usually at ```http://localhost:8501```).


## Usage Instructions

1. **Upload Dataset** (Optional):
   - By default, the app uses `Course_info.csv`. Replace it with a new dataset if needed.

2. **Select a Course**:
   - Use the dropdown menu in the sidebar to select a course you've completed.

3. **Choose a Recommendation Method**:
   - **Cosine Similarity** or **K-Nearest Neighbors**.

4. **View Recommendations**:
   - The app will display a list of recommended courses similar to your selection, along with key details like category, subcategory, price, and rating.

## Technologies Used

- **Python** (for preprocessing, modeling, and logic)
- **Streamlit** (for the user interface)
- **Pandas** and **NumPy** (for data manipulation)
- **Scikit-learn** (for similarity and distance calculations)
- **TensorFlow/Keras** (for neural network-based embedding generation)
