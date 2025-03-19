from transformers import pipeline
import streamlit as st

class Reader:
    def __init__(self):
        self.qa_model = self.load_reader_model()

    # Load reader model
    @st.cache_resource
    def load_reader_model(_self):
        reader_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
        return reader_model

    # Answer to a question
    def answer_question(self, context, question):

        # Generate answer using a reader model (QA model)
        if context:
            result = self.qa_model(question=question, context=context)
            answer = result["answer"]
        else:
            answer = "No relevant information found."
            context = ""

        return answer, context