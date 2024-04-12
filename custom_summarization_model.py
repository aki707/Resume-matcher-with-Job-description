from transformers import BartForConditionalGeneration, BartTokenizer
from preprocessing import jobs_data_final, resume_data_final
import torch

class TextSummarizer:


    def __init__(self, model_name):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize(self, text, max_input_length=1024, max_output_length=150, min_output_length=40):

        inputs = self.tokenizer([text], max_length=max_input_length, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(
            inputs['input_ids'].to(self.device),
            max_length=max_output_length,
            min_length=min_output_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
summarizer = TextSummarizer("geekradius/bart-large-cnn-fintetuned-samsum-repo")


from tqdm import tqdm
import pandas as pd

def batch_summarize(df, text_col, summarizer, batch_size=10, output_col=None):

    summarized_texts = []

    # Use the text_col as output_col if not specified
    if output_col is None:
        output_col = text_col

    # Iterate through the DataFrame in batches
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Summarizing"):
        end_idx = start_idx + batch_size
        batch = df[text_col][start_idx:end_idx]
        
        # Summarize each batch
        summarized_batch = [summarizer.summarize(text) for text in batch]
        summarized_texts.extend(summarized_batch)

    # Create a new DataFrame with the summarized text
    return pd.DataFrame({output_col: summarized_texts})


# Summarize the top 100 'processed_description' of jobs_data_final
top_jobs_data = jobs_data_final.head(100)

# Summariz jobs description
jobs_data_summarized = batch_summarize(top_jobs_data, 'processed_description', summarizer, batch_size=10, output_col='summarized_description')

# Summarize all 'processed_resume' in resume_data_final
resume_data_summarized = batch_summarize(resume_data_final, 'processed_resume', summarizer, batch_size=10, output_col='summarized_resume')