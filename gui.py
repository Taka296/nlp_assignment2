import streamlit as st
from text_extractor import TextExtractor
from retriever import Retriever
from reader import Reader

# Main GUI
def main():
    st.title("Canadian Legal Q&A System")

    # Extract texts from pdf files
    text_extractor = TextExtractor()
    doc_paths = text_extractor.extract_text()

    # Question input form
    question = st.text_input("Enter your question:", placeholder="Example: What is the purpose of this act?")

    if question:

        # Retriever part: Retrieve context based on the question
        with st.spinner("Retrieving context..."):
            retriever = Retriever(doc_paths)
            context = retriever.retrieve_context(question)

        # Reader part: Generate an answer
        with st.spinner('Generating answer...'):
            reader = Reader()
            answer, context = reader.answer_question(context, question)

        # Display results
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Source:")
        st.info(context)

if __name__ == "__main__":
    main()