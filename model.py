import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

def make_embeddings(df) :

    category_weight = 5
    subcategory_weight = 4

    # Numerical data normalization
    numerical_columns = ['price', 'num_subscribers', 'avg_rating', 'num_reviews', 'num_lectures', 'content_length_min']
    scaler = StandardScaler()
    numerical_data = df[numerical_columns].copy()  
    normalized_numerical_features = scaler.fit_transform(numerical_data)

    # Categorical data encoding
    categorical_columns = ['category', 'subcategory', 'is_paid']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical_data = encoder.fit_transform(df[categorical_columns])

    # Extract specific parts of the encoded categorical data for weighted scaling
    encoded_category = encoded_categorical_data[:, :len(df['category'].unique())]  # 'category' columns
    encoded_subcategory = encoded_categorical_data[:, len(df['category'].unique()):(len(df['category'].unique()) + len(df['subcategory'].unique()))]  # 'subcategory' columns
    encoded_is_paid = encoded_categorical_data[:, -1]  # 'is_paid' column

    weighted_category = encoded_category * category_weight
    weighted_subcategory = encoded_subcategory * subcategory_weight

    encoded_categorical_df = np.hstack([weighted_category, weighted_subcategory, encoded_is_paid.reshape(-1, 1)])

    # Text tokenization for 'Topic'
    tokenizer = Tokenizer(num_words=5000)  # Limit vocabulary to top 5000 words
    tokenizer.fit_on_texts(df['topic'])
    topic_sequences = tokenizer.texts_to_sequences(df['topic'])

    # Padding topic
    topic_padded = pad_sequences(topic_sequences, padding='post')

    combined_features = np.hstack([
        topic_padded,  
        normalized_numerical_features,  
        encoded_categorical_df  
    ])

    # Create the embedding model
    embedding_dim = 64  

    embedding_model = models.Sequential([
        layers.Input(shape=(combined_features.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(embedding_dim, activation='linear')  # Embedding layer
    ])

    # Generate embeddings
    embeddings = embedding_model.predict(combined_features)

    # Normalizing embeddings for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return normalized_embeddings


