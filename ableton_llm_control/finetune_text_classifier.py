import json

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Define constants

SEED = 123456
DATA_FILE = "./data/command_examples.json"
HF_MODEL = "distilbert-base-uncased"
MODEL_NAME = "ableton_osc_matcher"
MODEL_SAVE_PATH = f"./artifacts/{MODEL_NAME}_trained"

# Prepare dataset

with open(DATA_FILE, "r", encoding="utf8") as f:
    raw_dataset = json.loads(f.read())
print(raw_dataset)

keys = raw_dataset.keys()
num_labels = len(keys)

dataset = (
    Dataset.from_list(
        [
            {"label": label, "command": command}
            for label, examples in enumerate(raw_dataset.values())
            for command in examples
        ]
    )
    .train_test_split(test_size=0.1, seed=SEED)
    .shuffle(seed=SEED)
)

id2label = {i: label for i, label in enumerate(keys)}
label2id = {label: i for i, label in enumerate(keys)}

print(dataset)
print(f"Example: {dataset['train'][0]}")
print(id2label, label2id)

# Preprocess data

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)


def preprocess_function(examples):
    return tokenizer(examples["command"], truncation=True)


tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set up evaluation

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Train

model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    learning_rate=2e-5,
    per_device_train_batch_size=42,
    per_device_eval_batch_size=42,
    num_train_epochs=60,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Test inference

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

command = "Now let's pick it up from the beginning"
classifier(command)

command = "xxx"
classifier(command)

# Save trained model

model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
