# load the dataset
from datasets import load_dataset, Dataset
import torch
from transformers import ViTImageProcessor, ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import pandas as pd
import os
from data import MemotionDataset
from PIL import Image

os.environ['CURL_CA_BUNDLE'] = ''


model_name_or_path = 'google/vit-base-patch16-224-in21K'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path) # image encoder ViT model
metric = evaluate.load("accuracy")
labels = ['positive', 'neutral', 'negative']
label2id = {label: i for i, label in enumerate(labels)} 
id2label = {i: label for label, i in label2id.items()}


# Function to load images from disk
def load_image(example):
    image_path = os.path.join('memotion_dataset_7k', 'images', example['image_name'])
    #print(image_path)
    example['images']=None
    example['labels']=None

    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}. Skipping...")
        return example  # Skip 
    
    try:
        example['images'] = Image.open(image_path).convert("RGB")  # Load and convert to RGB
    except Exception as e:
        print(f"Warning: Failed to load {image_path} due to {e}. Skipping...")
        return example  # Skip corrupt images
    example["labels"] = label2id[example['label']]
    
    return example

# process batch of examples and prepare them for the ViT model
def transform(example_batch):
    inputs = feature_extractor(images=example_batch['images'], return_tensors='pt')
    inputs['labels'] = torch.tensor(example_batch['labels'])  
    return inputs

#stack individual examples in a batch (for correct shape)
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch], dtype=torch.long)
    }


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

if __name__=="__main__":
    
    df = pd.read_csv('folds.csv')

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Apply image loading function
    dataset = dataset.map(load_image, remove_columns=["image_name", "text_ocr", "text_corrected", "humour", "sarcasm", "offensive", "motivational", "label"], batched=False)
    dataset = dataset.filter(lambda x: x['images'] is not None and x['labels'] is not None)
    print(dataset[:3])
    # Split into train & test
    ds = dataset.train_test_split(test_size=0.2)
    # Transform dataset
    prepared_ds = ds.with_transform(transform)

    print("Downloading model ...")
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id, 
        cache_dir='models'
    )
    print("Finished downloading model ...")

    training_args = TrainingArguments(
    output_dir="./vit-sentiment-analysis",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=feature_extractor,
    )
    print("Start Training the model ...")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    print("Start evaluating the model ...")
    metrics = trainer.evaluate(prepared_ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)