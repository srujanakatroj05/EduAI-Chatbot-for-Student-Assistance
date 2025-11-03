import time
import os
import pickle
import pandas as pd
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Set directories
data_dir = "./excel_files/"
index_dir = "./index_generated/Student_details/"

# Function to save objects
def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Initialize embeddings
def initialize_embeddings() -> HuggingFaceEmbeddings:
    model_name = "./all-MiniLM-L6-v2/"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

embeddings = initialize_embeddings()

# Load and concatenate all Excel files
df_list = []
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    df = pd.read_excel(file_path).dropna(how='all')  # Drop rows where all values are NaN
    if not df.empty:
        df_list.append(df)

final_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# Convert DataFrame rows to text, ignoring NaN values
df_texts = final_df.apply(lambda row: " ".join(row.dropna().astype(str)), axis=1).tolist()
df_texts = [text for text in df_texts if text.strip()]  # Remove empty strings

# Split texts into smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=512,
    separator="...",
    length_function=len,
    is_separator_regex=False,
)

split_texts = []
split_metadata = []

for idx, text in enumerate(df_texts):
    chunks = text_splitter.split_text(text)
    split_texts.extend(chunks)
    split_metadata.extend([final_df.iloc[idx].dropna().to_dict()] * len(chunks))  # Exclude NaN metadata

# Ensure metadata and texts are of equal length
assert len(split_texts) == len(split_metadata), "Mismatch between split_texts and metadata!"

# Generate FAISS index
def generate_index(texts: List, embeddings: HuggingFaceEmbeddings, metadata: List) -> FAISS:
    return FAISS.from_texts(texts, embeddings, metadata)

print("INFO: Generating Index...")
start = time.time()

vectorstore = generate_index(split_texts, embeddings, split_metadata)
vectorstore.save_local(index_dir + "index_files/")
save_object(vectorstore, os.path.join(index_dir, 'retriever.pkl'))

end = time.time()
print(f"INFO: Index generated in {round(end - start, 2)} seconds.")
