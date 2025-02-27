from datasets import Dataset
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = ''

# Model and tokenizer
model_name_or_path = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
metric = evaluate.load("accuracy")

# Labels for sentiment classification
labels = ['positive', 'neutral', 'negative',]
label2id = {label: i for i, label in enumerate(labels)} 
id2label = {i: label for label, i in label2id.items()}

# Process text data
def tokenize_function(example):
    text = example['text_corrected']

    # Ensure text is a valid string
    if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
        text = "empty"  # Replace with default text

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    #print(example['label'])
    example["labels"] = label2id[example['label']] # Convert labels to numeric values
    return inputs

# Compute metrics function
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

if __name__=="__main__":

    # Load dataset
    df = pd.read_csv('folds.csv')

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(tokenize_function, remove_columns=["image_name", "text_ocr", "text_corrected", "humour", "sarcasm", "offensive", "motivational", "label"], batched=False)
    #print(dataset[:3])

    # Split into train & test
    ds = dataset.train_test_split(test_size=0.2)

    # Load RoBERTa model
    print("Downloading model ...")
    model = RobertaForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir='models'
    )
    print("Finished downloading model ...")

    # Training arguments
    training_args = TrainingArguments(
    output_dir="./roberta-sentiment-analysis",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    print("Start Training the model ...")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluate model
    print("Start evaluating the model ...")
    metrics = trainer.evaluate(ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
