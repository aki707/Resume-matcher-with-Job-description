from qdrant_database import jobs_combined, qdrant_interface


# Example job vector from the jobs_combined DataFrame
example_job_vector = jobs_combined['summarized_description_encoded'].iloc[44]

jobs_combined['summarized_description'][44]

# Find top 5 matching resumes for the example job
matched_resumes = qdrant_interface.match_jobs_to_resumes(example_job_vector, top_k=5)
for resume, score in matched_resumes:
    print(f"Matched Resume: {resume['summarized_resume']}, Score: {score}")
    