from sentence_transformers import SentenceTransformer
import pandas as pd
from custom_summarization_model import jobs_data_summarized, resume_data_summarized
from preprocessing import jobs_data_final, resume_data_final

class SentenceTransformerEncoder:
    """
    A class to handle sentence encoding using Sentence Transformers, directly working with pandas DataFrames.
    This class encodes text data in a specified DataFrame column into vector representations.

    Attributes:
        model (SentenceTransformer): The Sentence Transformer model used for encoding.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SentenceTransformerEncoder with a specified Sentence Transformer model.

        Args:
            model_name (str): The name of the Sentence Transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def encode_column(self, df, column, batch_size=32, encoded_column_suffix='_encoded'):
        """
        Encodes a specific column in a DataFrame and adds a new column with encoded vectors.

        Args:
            df (pd.DataFrame): The DataFrame containing the texts to encode.
            column (str): The name of the column to encode.
            batch_size (int): The size of each batch for processing.
            encoded_column_suffix (str): Suffix for the new column containing encoded vectors.

        Returns:
            pd.DataFrame: The original DataFrame with an additional column containing encoded vectors.
        
        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Encoding the text data in batches
        encoded_vectors = []
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_texts = df[column][start_idx:end_idx].tolist()
            batch_encoded = self.model.encode(batch_texts, show_progress_bar=True)
            encoded_vectors.extend(batch_encoded)

        # Adding the encoded vectors as a new column in the DataFrame
        df[column + encoded_column_suffix] = encoded_vectors
        return df
    
    
# Example Usage
encoder = SentenceTransformerEncoder(model_name='all-MiniLM-L6-v2')

# Encoding the summarized job descriptions
jobs_data_summarized_and_encoded = encoder.encode_column(jobs_data_summarized, 'summarized_description')

# Encoding the summarized resumes
resume_data_summarized_and_encoded = encoder.encode_column(resume_data_summarized, 'summarized_resume')


# Combine the jobs data
jobs_combined = pd.merge(
    jobs_data_final,
    jobs_data_summarized_and_encoded[['summarized_description', 'summarized_description_encoded']],
    left_index=True, right_index=True)

# Combine the resume data
resume_combined = pd.merge(
    resume_data_final,
    resume_data_summarized_and_encoded[['summarized_resume', 'summarized_resume_encoded']],
    left_index=True, right_index=True)

# Reset index of DataFrame
jobs_combined.reset_index(drop=True, inplace=True)
resume_combined.reset_index(drop=True, inplace=True)