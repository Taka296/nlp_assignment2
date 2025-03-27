from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import streamlit as st
import pandas as pd
import os

class FineTuning:
    def __init__(self, base_model_name="deepset/roberta-base-squad2",
                 confidence_threshold=7.0, output_dir="./fine_tuned_model"):
        self.base_model_name = base_model_name
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir

    def fine_tuning(self, contexts, questions):

        # Load the base model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        base_model = AutoModelForQuestionAnswering.from_pretrained(self.base_model_name)
        base_model.eval()

        # Generate pseudo labels
        # To fine tune the base model, we need "answer" in addition to context and question
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
                if answer_start_char >= 0 and confidence >= self.confidence_threshold:
                    pseudo_answers.append({
                        "text": [answer_text],
                        "answer_start": [answer_start_char]
                    })
                    high_confidence_indices.append(i)
                else:
                    pseudo_answers.append({"text": [""], "answer_start": [0]})
            else:
                pseudo_answers.append({"text": [""], "answer_start": [0]})

        st.write(
            f"High-confidence answers found: {len(high_confidence_indices)}/{len(contexts)} ({len(high_confidence_indices) / len(contexts) * 100:.2f}%)")

        # Create dataset
        # Use only high-confidence examples
        filtered_contexts = [contexts[i] for i in high_confidence_indices]
        filtered_questions = [questions[i] for i in high_confidence_indices]
        filtered_answers = [pseudo_answers[i] for i in high_confidence_indices]

        # Convert answer dictionary to plain text
        answers_text = [ans['text'][0] if ans['text'] else "" for ans in filtered_answers]

        # Create a DataFrame
        df = pd.DataFrame({
            'context': filtered_contexts,
            'question': filtered_questions,
            'answer': answers_text
        })

        high_confidence_csv = os.path.join(self.output_dir, "high_confidence_qa.csv")
        # Save to CSV as an intermediate state
        df.to_csv(high_confidence_csv, index=False, encoding="utf-8-sig")


        # Create dataset
        train_dataset = Dataset.from_dict({
            'id': [f"sample-{i}" for i in range(len(filtered_contexts))],
            'context': filtered_contexts,
            'question': filtered_questions,
            'answers': filtered_answers
        })

        # Data preprocessing
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

            # test_token = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            # st.write(test_token)

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

        # Process dataset
        tokenized_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Training settings
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=5e-5,  # Slightly lower learning rate than usual
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )

        # Create a new model instance
        fine_tune_model = AutoModelForQuestionAnswering.from_pretrained(self.base_model_name)

        # Initialize Trainer and train the model
        trainer = Trainer(
            model=fine_tune_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Save the model
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

