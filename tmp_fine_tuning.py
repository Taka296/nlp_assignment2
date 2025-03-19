from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np


def pseudo_label_fine_tuning(contexts, questions, base_model_name="deepset/bert-base-cased-squad2",
                             confidence_threshold=10.0, output_dir="./pseudo_labeled_model"):
    """
    Fine-tuning using pseudo-label generation

    Args:
        contexts: List of context texts
        questions: List of question texts
        base_model_name: Name of the base SQuAD model
        confidence_threshold: Confidence threshold
        output_dir: Output directory

    Returns:
        fine_tuned_model: Fine-tuned model
        tokenizer: Tokenizer
    """
    # 1. Load the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)
    base_model.eval()  # Set to evaluation mode

    # 2. Generate pseudo labels
    pseudo_answers = []
    high_confidence_indices = []

    for i, (context, question) in enumerate(zip(contexts, questions)):
        inputs = tokenizer(question, context, return_tensors="pt",
                           max_length=384, truncation=True, padding="max_length")

        with torch.no_grad():
            outputs = base_model(**inputs)

        # Identify the most probable start and end positions
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Get the most probable start and end positions
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        # Check if it is a valid answer range (start position <= end position)
        if start_idx <= end_idx:
            # Compute confidence score
            confidence = start_logits[start_idx].item() + end_logits[end_idx].item()

            # Convert tokens to text
            input_ids = inputs.input_ids[0]
            tokens = input_ids[start_idx:end_idx + 1]
            answer_text = tokenizer.decode(tokens, skip_special_tokens=True)

            # Find the actual start position of the answer in the context
            answer_start_char = context.find(answer_text)

            # If the answer exists in the context and confidence is above the threshold
            if answer_start_char >= 0 and confidence >= confidence_threshold:
                pseudo_answers.append({
                    "text": [answer_text],
                    "answer_start": [answer_start_char]
                })
                high_confidence_indices.append(i)
            else:
                pseudo_answers.append({"text": [""], "answer_start": [0]})
        else:
            pseudo_answers.append({"text": [""], "answer_start": [0]})

    print(
        f"High-confidence answers found: {len(high_confidence_indices)}/{len(contexts)} ({len(high_confidence_indices) / len(contexts) * 100:.2f}%)")

    # 3. Create dataset
    # Use only high-confidence examples
    filtered_contexts = [contexts[i] for i in high_confidence_indices]
    filtered_questions = [questions[i] for i in high_confidence_indices]
    filtered_answers = [pseudo_answers[i] for i in high_confidence_indices]

    if len(filtered_contexts) < 10:
        print("Warning: Too few high-confidence answers. The model may not function properly.")
        # Consider relaxing the confidence condition to ensure a minimum dataset size

    # Create dataset
    train_dataset = Dataset.from_dict({
        'id': [f"sample-{i}" for i in range(len(filtered_contexts))],
        'context': filtered_contexts,
        'question': filtered_questions,
        'answers': filtered_answers
    })

    # 4. Data preprocessing
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            stride=128,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully contained within the context, set labels to (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find the token corresponding to the start and end of the answer
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # 5. Process dataset
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    # 6. Training settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,  # Slightly lower learning rate than usual
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    # 7. Create a new model instance (optional)
    # You can use the same model, but creating a new instance allows better comparison
    fine_tune_model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)

    # 8. Initialize Trainer and train the model
    trainer = Trainer(
        model=fine_tune_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 9. Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return fine_tune_model, tokenizer


# Execution example
contexts = ["The Environmental Enforcement Act was enacted in 2010 to ensure compliance with Canada's environmental protection laws.",
            "The Environmental Enforcement Act strengthens the authority of enforcement officers to ensure compliance with environmental protection regulations."]
questions = ["When was the Environmental Enforcement Act enacted?",
             "What does the Environmental Enforcement Act strengthen?"]

model, tokenizer = pseudo_label_fine_tuning(contexts, questions)
