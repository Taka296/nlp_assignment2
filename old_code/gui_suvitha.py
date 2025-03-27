import streamlit as st

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from pathlib import Path

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


class Gui:
    def __init__(self):
        """
        Initialize the GUI class.
        """
        # Initialize the document store
        # This will store the documents in memory for later retrieval
        self.document_store = InMemoryDocumentStore()
    
        # Initialize the pipeline
        self.extraction_pipeline = None


        self.document_initialized = False
        
    def load_documents(self, directory_path):
        """
        Load documents from the specified directory.

        Args:
            directory_path (str): Path to the directory containing text files.
        """

        # Check if the directory exists
        if not Path(directory_path).exists():
            st.error(f"Directory {directory_path} does not exist.")

        # Get all text files in the directory
        file_names = [str(file) for file in Path(directory_path).glob("*.txt")]

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
        pipeline.add_component("cleaner", DocumentCleaner())
        pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10, split_overlap = 3))
        pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
        pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))

        pipeline.connect("converter", "cleaner")
        pipeline.connect("cleaner", "splitter")
        pipeline.connect("splitter", "embedder")
        pipeline.connect("embedder", "writer")
        pipeline.run({"converter": {"sources": file_names}})

        self.document_initialized = True


    def initialize_retreiever_pipeline(self):
        """
        Initialize the retriever model.
        """
        
        answer_reader = ExtractiveReader()

        # Initialize the reader and retriever models
        answer_reader.warm_up()

        # Initialize the pipeline
        pipeline = Pipeline()

        # Add the components to the pipeline
        pipeline.add_component(instance=SentenceTransformersTextEmbedder(), name="text_embedder") # Convert the query to an embedding using the SentenceTransformers model
        pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=self.document_store), name="text_retriever") # Retrieve documents based on the query embedding
        #pipeline.add_component(instance=answer_reader, name="reader")

        # Connect the components in the pipeline
        pipeline.connect("text_embedder.embedding", "text_retriever.query_embedding")
        #pipeline.connect("text_retriever.documents", "reader.documents")

        self.extraction_pipeline = pipeline

    
    def answer_question(self, question):
        """
        Answer the question using the document store.

        Args:
            question (str): The question to be answered.

        Returns:
            dict: The answer to the question.
        """

        # Run the pipeline to get the answer
        #result = self.extraction_pipeline.run({"text_embedder": {"text": question}, "text_retriever": {"top_k": 3}, "reader": {"query": question, "top_k": 2}})
        result = self.extraction_pipeline.run({"text_embedder": {"text": question}, "text_retriever": {"top_k": 3}})
        return result
        
    # Launch the Streamlit app
    def launch(self):
        st.title("Canadian Employment Law Q&A System")
        st.subheader("Ask questions about Canadian employment law.")

        if self.document_initialized == False:

            # Show a loading spinner while loading documents
            with st.spinner("Loading documents..."):

                # Load the documents from the specified directory
                self.load_documents(Path(__file__).parent / "texts")

        if self.extraction_pipeline is None:

            # Initialize the retriever pipeline
            self.initialize_retreiever_pipeline()

        # Question input form
        question = st.text_input("Enter your question:", placeholder="Example: What is the purpose of this act?")

        if question:
            # Show a loading spinner while retrieving the answer
            with st.spinner("Retrieving answer..."):
                answer = self.answer_question(question)

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    gui = Gui()
    gui.launch()