# Execution example
contexts = ["The Environmental Enforcement Act was enacted in 2010 to ensure compliance with Canada's environmental protection laws.",
            "The Environmental Enforcement Act strengthens the authority of enforcement officers to ensure compliance with environmental protection regulations."]
questions = ["When was the Environmental Enforcement Act enacted?",
             "What does the Environmental Enforcement Act strengthen?"]

st.title("Fine Tuning Test")
trainer = FineTuning()
model, tokenizer = trainer.fine_tuning(contexts, questions)
st.write(model)
st.write(tokenizer)

# Get the most probable start and end positions
"""
start_idx = torch.argmax(start_logits).item()
end_idx = torch.argmax(end_logits).item()

st.write(f"start_idx: {start_idx}")
st.write(f"end_idx: {end_idx}")

# If we don't have a valid span, return empty string
if start_idx > end_idx:
    return ""

# Convert tokens to text
input_ids = inputs.input_ids[0]
tokens = input_ids[start_idx:end_idx + 1]
answer = tokenizer.decode(tokens, skip_special_tokens=True)

return answer        
"""