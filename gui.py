import streamlit as st
import pandas as pd
from text_extractor import TextExtractor
from retriever import Retriever
from reader import Reader
from fine_tuning import FineTuning
import os

# Main GUI
def main():

    st.title("Canadian Legal Q&A System")
    st.subheader("Ask questions about Canadian employment law")

    # Extract texts from pdf files
    text_extractor = TextExtractor()
    doc_paths = text_extractor.extract_text()

    # If a QA model has not been fine-tuned yet, fine-tune the model
    model_path = './fine_tuned_model'
    if not os.path.exists(model_path):
        with st.spinner("Fine tuning the model..."):
            os.mkdir(model_path)
            fine_tuning_csv = pd.read_csv("./fine_tuning_csv/QA_system.csv")
            contexts = fine_tuning_csv['context']
            questions = fine_tuning_csv['question']
            model_name = "deepset/roberta-base-squad2"
            fine_tuner = FineTuning(model_name)
            fine_tuner.fine_tuning(contexts, questions)

    # Question input form
    question = st.text_input("Enter your question:", placeholder="Example: What is the purpose of this act?")

    if question:

        # Retriever part: Retrieve context based on the question
        with st.spinner("Retrieving context..."):
            retriever = Retriever(doc_paths)
            context = retriever.retrieve_context(question)

        # Reader part: Generate an answer
        with st.spinner('Generating answer...'):
            reader = Reader(model_path)
            answer, confidence = reader.answer_question(context, question)

        # Display answer
        st.subheader("Answer:")
        st.write(answer)

        # Display confidence score
        st.subheader("Confidence Score:")
        st.write(f"{confidence:.2f}")

        # Display context(source)
        st.subheader("Source:")
        st.info(context)

if __name__ == "__main__":
    main()