from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Record, Filter
from random import uniform
from encoding_with_sentence_transformers import encoder
from preprocessing import jobs_combined, resume_combined

class QdrantInterface:
    """
    A class for interfacing with the Qdrant vector database.

    Attributes:
        client (QdrantClient): Client instance for interacting with Qdrant.
        vector_dimension (int): Dimension of the vectors used in the collection.
    """

    """
    A class for interfacing with the Qdrant vector database.
    ...
    """
    def __init__(self, url, api_key, vector_dimension):
        """
        Initializes the QdrantInterface with the specified Qdrant URL, API key, and vector dimension.

        Args:
            url (str): Full URL of the Qdrant server.
            api_key (str): API key for Qdrant.
            vector_dimension (int): Dimension of vectors to be stored in Qdrant.
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_dimension = vector_dimension

    def create_collection(self, collection_name, distance_metric=Distance.COSINE):
        """
        Creates or recreates a collection in Qdrant.

        Args:
            collection_name (str): Name of the collection.
            distance_metric (Distance): Distance metric for vector comparisons.
        """
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_dimension, distance=distance_metric)
        )

    def save_to_qdrant(self, df, collection_name, vector_col, payload_cols, batch_size=100):
        """
        Saves a DataFrame to Qdrant in batches.

        Args:
            df (pd.DataFrame): DataFrame containing data to save.
            collection_name (str): Name of the collection in Qdrant.
            vector_col (str): Name of the column containing vectors.
            payload_cols (list[str]): List of column names to include as payload.
            batch_size (int): Number of records to process in each batch.
        """

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            records = []
            for idx, row in batch.iterrows():
                # Debug print
                print(f"Index: {idx}, Vector Type: {type(row[vector_col])}, First 10 Elements: {row[vector_col][:10]}")
                record = Record(
                    id=idx,
                    vector=row[vector_col],
                    payload={col: row[col] for col in payload_cols}
                )
                records.append(record)
            self.client.upload_records(collection_name=collection_name, records=records)

            
            
    def retrieve_specific_records(self, collection_name, ids):
        """
        Retrieves specific records by their IDs from a Qdrant collection.

        Args:
            collection_name (str): The name of the collection.
            ids (list): List of record IDs to retrieve.

        Returns:
            List of specific records from the collection.
        """
        return self.client.retrieve(collection_name=collection_name, ids=ids)
    
    def view_sample_records(self, collection_name, vector_dimension, limit=10):
        """
        Retrieves a sample of records from a Qdrant collection using a dummy search.

        Args:
            collection_name (str): The name of the collection.
            vector_dimension (int): Dimension of vectors in the collection.
            limit (int): The number of records to retrieve.

        Returns:
            List of sample records from the collection.
        """
        # Generate a random vector
        random_vector = [uniform(-1, 1) for _ in range(vector_dimension)]

        # Perform a dummy search
        return self.client.search(
            collection_name=collection_name,
            query_vector=random_vector,
            limit=limit
        )
    
    def match_resumes_to_jobs(self, resume_vector, top_k=10):
        """
        Matches a given resume vector to job postings.

        Args:
            resume_vector (list): The vector representation of a resume.
            top_k (int): Number of top similar matches to return.

        Returns:
            List of matched job postings with similarity scores.
        """
        hits = self.client.search(
            collection_name="jobs",
            query_vector=resume_vector,
            limit=top_k,
            with_payload=True
        )
        return [(hit.payload, hit.score) for hit in hits]

    def match_jobs_to_resumes(self, job_vector, top_k=10):
        """
        Matches a given job vector to resumes.

        Args:
            job_vector (list): The vector representation of a job posting.
            top_k (int): Number of top similar matches to return.

        Returns:
            List of tuples containing matched resumes and their similarity scores.
        """
        hits = self.client.search(
            collection_name="resumes",
            query_vector=job_vector,
            limit=top_k,
            with_payload=True
        )
        return [(hit.payload, hit.score) for hit in hits]
    
vector_dimension = encoder.model.get_sentence_embedding_dimension()

print("vector dimesion"+vector_dimension) 

QUADRANT_ENDPOINT = 'https://1f930567-54d3-4ce3-aa0e-d85466e76165.us-east4-0.gcp.cloud.qdrant.io'
QUADRANT_API_KEY = '82zjOnCucpn61l3sF4u56dfBPZa26kLgveRXJvzzHip4i2iafN10Tg'

vector_dimension = encoder.model.get_sentence_embedding_dimension()
qdrant_interface = QdrantInterface(QUADRANT_ENDPOINT, QUADRANT_API_KEY, vector_dimension)
qdrant_interface.create_collection('jobs', Distance.COSINE)
qdrant_interface.create_collection('resumes', Distance.COSINE)

# Function to ensure vectors are in list format
def ensure_list_format(df, vector_col):
    df[vector_col] = df[vector_col].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    return df

# Ensure vectors are in the correct format before uploading
jobs_combined = ensure_list_format(jobs_combined, 'summarized_description_encoded')
resume_combined = ensure_list_format(resume_combined, 'summarized_resume_encoded')


# Now upload to Qdrant
qdrant_interface.save_to_qdrant(jobs_combined, 'jobs', 'summarized_description_encoded', ['processed_title', 'processed_description', 'token_count', 'summarized_description'])
qdrant_interface.save_to_qdrant(resume_combined, 'resumes', 'summarized_resume_encoded', ['processed_resume', 'token_count', 'summarized_resume'])


# View sample records from the 'jobs' collection
sample_jobs_records = qdrant_interface.view_sample_records('jobs', vector_dimension, limit=2)
sample_jobs_records

# Retrieve specific records by IDs from the 'jobs' collection
specific_jobs_records = qdrant_interface.retrieve_specific_records('jobs', ids=[1])

specific_jobs_records

