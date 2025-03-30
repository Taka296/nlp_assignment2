from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st

class Reader:
    def __init__(self, model_path):
        self.model_path = model_path

    # Load reader model
    @st.cache_resource
    def load_fine_tuned_models(_self):
        reader_model = AutoModelForQuestionAnswering.from_pretrained(_self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(_self.model_path)
        return reader_model, tokenizer

    def answer_question(self, context, question):

        # Load the fine-tuned model and tokenizer
        model, tokenizer = self.load_fine_tuned_models()

        # Tokenize the input
        inputs = tokenizer(question, context, return_tensors="pt",
                           max_length=512, truncation=True, padding="max_length")

        # Get the model prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the most probable start and end positions
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Compute no-answer score from [CLS] token (position 0)
        no_answer_score = start_logits[0].item() + end_logits[0].item()

        # Try top-k start and end positions
        top_k = 10
        start_indexes = torch.topk(start_logits, top_k).indices.tolist()
        end_indexes = torch.topk(end_logits, top_k).indices.tolist()

        # Find the best valid answer span (excluding position 0)
        best_score = float('-inf')
        best_answer = ""
        confidence = 0

        for start_i in start_indexes:
            for end_i in end_indexes:
                if start_i == 0 or end_i == 0:
                    continue  # Skip [CLS] position
                if start_i <= end_i and end_i - start_i < 50:
                    score = start_logits[start_i] + end_logits[end_i]
                    if score > best_score:
                        best_score = score
                        tokens = inputs.input_ids[0][start_i:end_i + 1]
                        answer = tokenizer.decode(tokens, skip_special_tokens=True)
                        confidence = score.item()
                        best_answer = answer

        # Compare with no-answer score
        if no_answer_score > best_score or confidence < 7.0:
            return "No valid answer found", no_answer_score
        else:
            return best_answer, confidence
