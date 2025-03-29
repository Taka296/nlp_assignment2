import streamlit as st
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from pathlib import Path
import re


###
# Install the necessary libraries
# pip install haystack-ai accelerate "sentence-transformers>=3.0.0" "datasets>=2.6.1" streamlit
# pip install nltk>=3.9.1

# Architecture:
# Streamlit GUI
# Haystack Framework for NLP tasks
# TextFileToDocument from Haystack to load documents
# InmemoryDocumentStore for storing documents in memory
# Transformers for the reader model
# SentenceTransformers for the retriever model
# Haystack for the pipeline and document store

# Document Input:
# All the text files under the "texts" directory
# The directory structure is as follows:
# texts
# ├── text1.txt
# ├── text2.txt
# ├── text3.txt

class HayStackRetriever:
    def __init__(_self):
        """
        Initialize the HayStackRetriever class.
        """
        pass

    @st.cache_resource
    def load_documents(_self, _directory_path):
        """
        Load documents from the specified directory.

        Args:
            directory_path (str): Path to the directory containing text files.

        Returns:
            DocumentStore: The document store containing the loaded documents.
        """

        # Check if the directory exists
        if not Path(_directory_path).exists():
            st.error(f"Directory {_directory_path} does not exist.")

        # Get all text files in the directory
        file_names = [str(file) for file in Path(_directory_path).glob("*.txt")]

        # Initialize the document store
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        #document_store = InMemoryDocumentStore()

        # Build the pipeline
        # This pipeline will convert text files to documents, clean them, split them into smaller chunks,
        # and write them to the document store
        # The pipeline consists of the following components:
        # 1. TextFileToDocument: Converts text files to documents
        # 2. DocumentCleaner: Cleans the documents
        # 3. DocumentSplitter: Splits the documents into smaller chunks
        # 4. DocumentWriter: Writes the documents to the document store

        pipeline = Pipeline()
        pipeline.add_component("converter", TextFileToDocument())

        # Removes \n, \t, punctuation, special characters
        pipeline.add_component("cleaner", DocumentCleaner(remove_regex=r"-\s*|~"))
        # pipeline.add_component("splitter", DocumentSplitter(split_by="passage", split_length=10, split_overlap = 3))
        pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=100, split_overlap = 10))
        pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))

        pipeline.connect("converter", "cleaner")
        pipeline.connect("cleaner", "splitter")
        pipeline.connect("splitter", "embedder")
        pipeline.connect("embedder", "writer")
        pipeline.run({"converter": {"sources": file_names}})

        return document_store

    @st.cache_resource
    def retrieve_context(_self, question):
        """
        Initialize the retriever model.

        Parameters:
            question (str): The question to be answered.

        Returns:
            dict: The result of the retrieval process.
        """

        _document_store = _self.load_documents(Path(__file__).parent / "texts")

        # # Initialize the reader
        # answer_reader = ExtractiveReader()
        # # Initialize the reader and retriever models
        # answer_reader.warm_up()

        # Initialize the pipeline
        pipeline = Pipeline()

        # Add the components to the pipeline
        pipeline.add_component(instance=SentenceTransformersTextEmbedder(), name="text_embedder")  # Question encoder model
        pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=_document_store),
                            name="text_retriever")  # Retrieve documents based on the query embedding
        # pipeline.add_component(instance=answer_reader, name="reader")

        # Connect the components in the pipeline
        pipeline.connect("text_embedder.embedding", "text_retriever.query_embedding")
        # pipeline.connect("text_retriever.documents", "reader.documents")
        # result = pipeline.run({"text_embedder": {"text": question}, "text_retriever": {"top_k": 1}, "reader": {"query": question, "top_k": 1}})

        result = pipeline.run({"text_embedder": {"text": question}, "text_retriever": {"top_k": 3}})

        retrieved_documents = result["text_retriever"]["documents"]

        # Extract the file path from meta information in the documents
        retrieved_file_paths = [doc.meta["file_path"] for doc in retrieved_documents if "file_path" in doc.meta]

        # Extract the content from the retrieved documents
        retrieved_content = [str(doc.content) for doc in retrieved_documents if doc.content is not None]

        return retrieved_documents, retrieved_file_paths, retrieved_content


    def answer_question(_self, question, extraction_pipeline):
        """
        Answer the question using the document store.

        Args:
            question (str): The question to be answered.
            extraction_pipeline (Pipeline): The pipeline to use for extraction.

        Returns:
            dict: The answer to the question.
        """

        # Run the pipeline to get the answer
        result = extraction_pipeline.run({"text_embedder": {"text": question}, "text_retriever": {"top_k": 3},
                                        "reader": {"query": question, "top_k": 2}})

        return result


    # Launch the Streamlit app
    def main(self):
        st.title("Canadian Employment Law Q&A System")
        st.subheader("Ask questions about Canadian employment law.")

        # Question input form
        question = st.text_input("Enter your question:")

        if question:
            # Show a loading spinner while processing the question
            with st.spinner("Processing..."):
                # Answer the question using the document store
                source = self.retrieve_context(question)
                context =[doc.content for doc in source["text_retriever"]["documents"]]

            # Display the answer
            st.subheader("Context:")
            st.write(context)


if __name__ == "__main__":
    HayStackRetriever().main()