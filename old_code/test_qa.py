from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st


def test_qa_model(model_path, context, question):

    # Load the fine-tuned model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors="pt",
                       max_length=384, truncation=True, padding="max_length")

    # Get the model prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most probable start and end positions
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Get the most probable start and end positions
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    # Calculate confidence score
    confidence = (start_logits[start_idx].item() + end_logits[end_idx].item())

    # If we don't have a valid span, return empty string
    if start_idx > end_idx:
        return "", -1, -1, confidence

    # Convert tokens to text
    input_ids = inputs.input_ids[0]
    tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)

    # Find the character position in the original context
    answer_char_start = context.find(answer)

    return answer, answer_char_start, answer_char_start + len(answer), confidence


# Add this to your Streamlit app
st.title("QA Model Test")

# Use saved model path from previous function
model_path = "./pseudo_labeled_model"

# Test inputs
test_context = st.text_area("Context",
                            "The Environmental Enforcement Act was enacted in 2010 to ensure compliance with Canada's environmental protection laws.")
test_question = st.text_input("Question", "When was the Environmental Enforcement Act enacted?")

if st.button("Get Answer"):
    answer, start, end, confidence = test_qa_model(model_path, test_context, test_question)

    st.write("### Answer:")
    st.write(answer)

    st.write("### Confidence Score:")
    st.write(f"{confidence:.2f}")

    # Highlight the answer in the context if found
    if start >= 0:
        highlighted_context = test_context[:start] + "**" + test_context[start:end] + "**" + test_context[end:]
        st.write("### Context with highlighted answer:")
        st.markdown(highlighted_context)