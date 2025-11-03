import time
from pathlib import Path
from typing import List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

import pickle
import os

# set variables
doc_path = "./md_processed_2" # "./md_processed_1"
index_dir = "./index_generated/Subjects/" # "./index_generated/Faculty_details/"

def load_from_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###", "Header 3")] #,("####", "Header 4")
headers_values_to_split_on ={ "Header 1" : "#","Header 2":"##","Header 3":"###"} #,"Header 4":"####"

def initialize_embeddings() -> HuggingFaceEmbeddings:
    model_name = "./all-MiniLM-L6-v2/"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def load_documents() -> List:
    dirs = os.listdir(doc_path)
    docs=[]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    for file in dirs:
        if file.endswith('.md'):
            with open(doc_path+'/'+file, 'r',encoding="utf-8") as file:
                data = file.read()
                # html_header_splits = md_splitter.create_documents([data])
                html_header_splits = md_splitter.split_text(data)
                # docs.append(html_header_splits)
                for doc in html_header_splits:
                    doc.metadata['source']=Path(file.name).stem +'.md'
                    doc.metadata['Heading']=Path(file.name).stem
                    heading1 = Path(file.name).stem
                    doc.page_content = f"\n\n# {heading1}\n{doc.page_content}".strip()
                    docs.append(doc)
    return docs

def split_chunks(sources: List,child_splitter) -> List:
    chunks = []
    for doc in child_splitter.split_documents(sources):
        # heading1 = doc.metadata.get('Header 1', '')
        # heading2 = doc.metadata.get('Header 2', '')
        # if doc.page_content[:2] != '# ':
        #     if doc.page_content[:2] != "##" :
        #         doc.page_content = f"\n\n# {heading1}\n## {heading2}\n{doc.page_content}".strip()
        #     else:
        #         doc.page_content = f"\n\n# {heading1}\n{doc.page_content}".strip()
        chunks.append(doc)
    return chunks

def generate_index(chunks: List, embeddings: HuggingFaceEmbeddings) -> FAISS:
    texts=[]
    metadatas=[]
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

embeddings = initialize_embeddings()

start = time.time()
sources = load_documents()

child_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=1, separators=[" ", ",", "\n"]) #chunk_overlap=16
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)

chunks = split_chunks(sources,child_splitter)
print("Creating Index")
vectorstore = generate_index(chunks, embeddings)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
# bm25_retriever = BM25Retriever.from_documents(sources)
# bm25_retriever.k = 2

retriever.add_documents(sources)
vectorstore.save_local(index_dir+"/index_files/")
save_object(store, './'+index_dir+'/retriever.pkl')
# save_object(bm25_retriever, './'+index_dir+'/bm25.pkl')

print("Index files saved at "+index_dir)
end = time.time()
print("\nEmbedding and Indexing time : " +str(end - start) +" sec")

# index = FAISS.load_local(index_dir, embeddings,allow_dangerous_deserialization=True)