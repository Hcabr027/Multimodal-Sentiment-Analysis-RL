from datasets import load_dataset, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTImageProcessor, ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments, RobertaTokenizer, RobertaModel, ViTForImageClassification
import numpy as np
import evaluate
import pandas as pd
import os
from PIL import Image
from train_ROBERTa import *
from train_ViT import *
from tqdm import tqdm
from PIL import ImageFile
import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CURL_CA_BUNDLE'] = ''

# Load RoBERTa for text embeddings
text_model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
text_model = RobertaModel.from_pretrained(text_model_name)

# Load ViT for image embeddings
image_model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(image_model_name)
image_model = ViTModel.from_pretrained(image_model_name)


def extract_text_embedding(text):
    if not isinstance(text, str) or text.strip() == "":  # Ensure it's a valid string
        text = "unknown"  # Default placeholder for missing values
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        output = text_model(**inputs)
    return output.last_hidden_state[:, 0, :]


def extract_image_embedding(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Missing image: {image_path}")
        return torch.zeros(768)  # Default zero vector for missing images
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = image_model(**inputs)
    return output.pooler_output.squeeze(0)


# def multimodal_fusion(text, image_path):
#     text_emb = extract_text_embedding(text)
#     image_emb = extract_image_embedding(image_path)
#     fused_embedding = torch.cat((text_emb, image_emb), dim=1)  # concat text & image embeddings
#     return fused_embedding


class SAC(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAC, self).__init__()
        self.msd = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )  # MSD
        self.dpsr = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )  # DPSR

    def forward(self, x):
        latent = self.msd(x)  # Modality-Sentiment Disentanglement
        reconstructed = self.dpsr(latent)  # Deep Phase Space Reconstruction
        return latent, reconstructed


class SAC_Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SAC_Policy, self).__init__()
        self.policy_nn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        result = self.policy_nn(x)
        return result


def reward_function(text_output, image_output, labels):
    combined = text_output + image_output  # Combine both agent predictions
    predicted = torch.argmax(combined, dim=1)
    reward = (predicted == labels).float().mean()  # Reward based on accuracy
    return reward


def custom_collate(batch):
    text_embs, image_embs, labels = zip(*batch)
    return torch.stack(text_embs), torch.stack(image_embs), torch.tensor(labels)

# come back to this -> fix
def predict(text, image_path):

    text_emb = extract_text_embedding(text)
    image_emb = extract_image_embedding(image_path)
    # fused_emb = multimodal_fusion(text_emb, image_emb)
    text_latent, _ = text_agent(text_emb)
    image_latent, _ = image_agent(image_emb)
    text_output = text_policy(text_latent)
    image_output = image_policy(image_latent)
    combined_output = text_output + image_output
    predicted_class = torch.argmax(combined_output, dim=1).item()
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_label = id2label[predicted_class]
    
    return predicted_label
    

if __name__=="__main__":

    image_dir = '/images'

    df = pd.read_csv('folds.csv')
    df['image_name'] = df['image_name'].astype(str).str.strip()
    df['text_corrected'] = df['text_corrected'].astype(str).str.strip().fillna("unknown")

    text_embeddings = []
    image_embeddings = []
    labels = []

    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
        # print(f"Label Type: {type(row['label'])}, Value: {row['label']}")

        text_emb = extract_text_embedding(row["text_corrected"]).squeeze(0)
        image_emb = extract_image_embedding(os.path.join(image_dir, row["image_name"]))
        
        text_embeddings.append(text_emb)
        image_embeddings.append(image_emb)
        labels.append(torch.tensor(label2id[row["label"]], dtype=torch.long))

    print("Finished embedding!")

    text_embeddings = torch.stack(text_embeddings)
    image_embeddings = torch.stack(image_embeddings)
    labels = torch.tensor(labels)
    data = list(zip(text_embeddings, image_embeddings, labels))

    text_dim = 768 
    image_dim = 768 
    hidden_dim = 512
    num_labels = 3 # positive, negative, neutral

    text_agent = SAC(text_dim, hidden_dim)
    image_agent = SAC(image_dim, hidden_dim)
    text_policy = SAC_Policy(hidden_dim, num_labels)
    image_policy = SAC_Policy(hidden_dim, num_labels)

    # train_dataset = SACDataset(df, '/images')
    # train_loader = DataLoader(data, batch_size=16, shuffle=True, collate_fn=custom_collate)
    train_loader = tqdm(DataLoader(data, batch_size=16, shuffle=True, collate_fn=custom_collate), desc="Loading data")

    optimizer = optim.Adam(list(text_agent.parameters()) + list(image_agent.parameters()) + list(text_policy.parameters()) + list(image_policy.parameters()), lr=1e-4)
    cross_entrop = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(5000):
        total_loss = 0
        for text_emb, image_emb, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()

            text_latent, _ = text_agent(text_emb)
            image_latent, _ = image_agent(image_emb)
            text_output = text_policy(text_latent)
            image_output = image_policy(image_latent)

            combined = text_output + image_output
            predicted = torch.argmax(combined, dim=1)
            reward = (predicted == labels).float().mean()

            loss_text = cross_entrop(text_output, labels)
            loss_image = cross_entrop(image_output, labels)
            loss = loss_text + loss_image - reward

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("loss.png")

    text = "This is an amazing day!"
    image_path = "img.jpg"
    prediction = predict(text, image_path)
    print(f"Prediction: {prediction}")
