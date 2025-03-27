import re
import os
import streamlit as st
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special chars
    return text.strip()

# Cache preprocessed data and pipeline setup
@st.cache_resource
#def setup_haystack_pipeline(txt_path):
def setup_haystack_pipeline(folder_path):
    
    # Initialize list to hold documents
    docs = []   
    # Load all .txt files from the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    raw_content = file.read()
                # Clean content and add to docs list
                cleaned_content = clean_text(raw_content)
                docs.append({"content": cleaned_content, "meta": {"name": filename}})
            except Exception as e:
                st.warning(f"Failed to process {filename}: {str(e)}")

    if not docs:
        raise ValueError("No valid .txt files found in the specified folder.")
    
    # Initialize InMemoryDocumentStore
    document_store = InMemoryDocumentStore(
        similarity="dot_product"  # DPR uses dot product
    )

    # Preprocess: Split into passages
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True
    )


    processed_docs = preprocessor.process(docs)

    # Write documents to store
    document_store.write_documents(processed_docs)

    # Initialize DPR Retriever
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Index embeddings
    document_store.update_embeddings(retriever)

    # Initialize RoBERTa Reader
    reader = FARMReader(
        model_name_or_path="deepset/roberta-base-squad2"
    )

    # Build pipeline
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    return pipeline, document_store

# Streamlit UI
def main():
    st.title("Canadian Employment Law Q&A System")
    st.subheader("Ask questions about Canadian employment law.")
    st.write("Ask a question")

    # Specify the folder path containing .txt files
    folder_path = "../texts"  # Replace with your folder path
    if not os.path.isdir(folder_path):
        st.error(f"Folder '{folder_path}' not found. Please provide a valid folder path.")
        return
    
    # Load and set up pipeline
    try:
        pipeline, document_store = setup_haystack_pipeline(folder_path)
    except Exception as e:
        st.error(f"Error setting up pipeline: {str(e)}")
        return

    # Question input
    question = st.text_input("Enter your question:", "")

    # Process question and display answers
    if question:
        with st.spinner("Finding answers..."):
            prediction = pipeline.run(
                query=question,
                params={
                    "Retriever": {"top_k": 5},  # Retrieve top 5 passages
                    "Reader": {"top_k": 3}      # Return top 3 answers
                }
            )

        st.subheader("Answers:")
        answers = prediction["answers"]
        if answers:
            for i, answer in enumerate(answers, 1):
                st.write(f"{i}. **{answer.answer}** (Score: {answer.score:.3f})")
                st.write(f"Context: {answer.context}")
                st.write("---")
        else:
            st.write("No answers found.")

if __name__ == "__main__":
    main()