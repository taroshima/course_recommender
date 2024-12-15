import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    df.columns = df.columns.str.strip()
    
    # Filter for English courses
    df = df[df['language'] == 'English']
    
    # Drop columns that wont be used
    columns_to_drop = [
        'id', 'course_url', 'instructor_url', 
        'published_time', 'last_update_date', 
        'instructor_name', 'language'
    ]
    df = df.drop(columns=columns_to_drop)
    
    # Drop rows with missing or empty values
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = df.dropna()
    
    output_file="cleaned_courses.csv"
    df.to_csv(output_file, index=False)

    return output_file




