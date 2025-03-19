from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
import streamlit as st

class Retriever:
    def __init__(self, doc_paths):
        self.doc_paths = doc_paths
        self.retriever = self.load_retriever_model()
        self.documents = self.load_documents()

    def retrieve_context(self, question):

        # Create index
        index, chunks = self.create_index(self.documents, self.retriever)

        # Vectorize the question
        query_embedding = self.retriever.encode([question])

        # Search for relevant documents
        k = 3  # Get top k relevant documents
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)

        # Get the most relevant chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = " ".join(relevant_chunks)

        return context

    # Load retriever model
    @st.cache_resource
    def load_retriever_model(_self):
        retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
        return retriever_model

    # Load documents
    @st.cache_data
    def load_documents(_self):
        docs = []
        # Load a document file
        file_paths = _self.doc_paths
        for file_path in file_paths:
            if os.path.exists(file_path):
                 if file_path.endswith(".txt"):
                     with open(file_path, "r", encoding="utf-8") as f:
                         docs.append(f.read())
        return docs


    # Function to split documents into chunks
    def split_documents(self, documents, max_length=100):
        chunks = []
        for doc in documents:
            # Simple splitting method: split by periods or Japanese periods
            sentences = re.split(r'[.]', doc)
            sentences = [s.strip() for s in sentences if s.strip()]

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_length:
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence + "."

            if current_chunk:
                chunks.append(current_chunk)

        return chunks


    # Create search index
    def create_index(self, documents, retriever):
        # Split documents into chunks
        chunks = self.split_documents(documents)

        # Encode each chunk
        chunk_embeddings = retriever.encode(chunks)

        # Create FAISS index
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings).astype('float32'))

        return index, chunks

