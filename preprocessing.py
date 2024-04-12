# Libraries

import nltk
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

jobs_data = pd.read_csv('Resume-matcher-with-Job-description\dataset\gsearch_jobs.csv')
resume_data = pd.read_csv('Resume-matcher-with-Job-description\dataset\Resume1.csv')

def preprocess_text(text):

    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_text)

def drop_duplicates(df, column_name):

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    original_size = df.shape[0]
    df_cleaned = df.drop_duplicates(subset=[column_name])
    new_size = df_cleaned.shape[0]

    print(f"Dropped {original_size - new_size} duplicates from '{column_name}'. New dataset size: {new_size}")

    return df_cleaned

def add_token_count_column(df, column_name):

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Creating a copy of the DataFrame to avoid modifying a slice
    df_copy = df.copy()

    # Tokenize each entry in the specified column and count the number of tokens
    df_copy['token_count'] = df_copy[column_name].apply(lambda x: len(word_tokenize(x)) if pd.notnull(x) else 0)

    return df_copy

# Apply preprocessing
print("Preprocessing jobs data...")
jobs_data['processed_description'] = jobs_data['description'].apply(preprocess_text)
jobs_data['processed_title'] = jobs_data['title'].apply(preprocess_text)

print("Preprocessing resume data...")
resume_data['processed_resume'] = resume_data['Resume'].apply(preprocess_text)
print("Done!")

jobs_data_cleaned = drop_duplicates(jobs_data, column_name='description')
resume_data_cleaned = drop_duplicates(resume_data, column_name='Resume')

jobs_data_cleaned_with_tokens = add_token_count_column(jobs_data_cleaned, column_name='processed_description')
resume_data_cleaned_with_tokens = add_token_count_column(resume_data_cleaned, column_name='processed_resume')

# Dropping unnecessary columns from jobs data
jobs_data_final = jobs_data_cleaned_with_tokens[['processed_title', 'processed_description', 'token_count']]

# Dropping unnecessary columns from resume data
resume_data_final = resume_data_cleaned_with_tokens[['processed_resume', 'token_count']]