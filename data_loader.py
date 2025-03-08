import numpy as np
import pickle
import torch
import torch.utils.data as data
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel, ViTModel, ViTFeatureExtractor
from datasets import Dataset
from PIL import Image


# data preprocess and feature extraction 

def create_vit_roberta_feature_extractor(vit_model='google/vit-base-patch16-224-in21K', roberta_model='roberta-base'):
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model)
    vit_model = ViTModel.from_pretrained(vit_model)# add CPU or GPU to(device) 
    tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
    roberta_model = RobertaModel.from_pretrained(roberta_model) # add CPU or GPU to(device)

    #unimodal agents
    def image_features_extractor(image):
        inputs = vit_feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
                outputs = vit_model(**inputs)
        image_features = outputs.last_hidden_state[:, 0, :] # Take CLS token output
        return image_features.squeeze() # check if need to squeeze 
    
    def text_features_extractor(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True) # add to device 
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        text_features = outputs.last_hidden_state[:, 0, :]
        return text_features.squeeze() # check dimension 
    
    return image_features_extractor, text_features_extractor

#process memotion data 
class MemotionData(data.Dataset):
    def __init__(self, dataset, images_folder, image_features_extractor, text_features_extractor, mode='train'):

        self.dataset = dataset
        self.image_folder = images_folder
        self.mode = mode
        self.image_features_extractor = image_features_extractor
        self.text_features_extractor = text_features_extractor
        self.mode = mode

        self.image_name_col = "image_name"
        self.text_col = "text_corrected"
        self.label_col = "label"
    
        self.label2id = {'positive': 0, 'neutral': 1, 'negative': 2}

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self,index):
        item = self.dataset[index]

        image_name_path = item[self.image_name_col]
        image_path = f"{self.image_folder}/{image_name_path}"
        text_string = item[self.text_col]
        label_str = item[self.label_col]
        label_id = self.label2id[label_str]

        image = Image.open(image_path).convert('RGB') # check vit input

        # extract image and text features 
        text_features = self.text_features_extractor(text_string)
        image_features = self.image_features_extractor(image)

        label_tensor = torch.tensor(label_id, dtype=torch.long)
        
        return text_features, image_features, label_tensor

def create_HF_dataset(csv_path, image_folder, device, test_size=0.2, val_size=0.2):
    data = pd.read_csv(csv_path)
    dataset_hf = Dataset.from_pandas(data)

    train_valid_test_split = dataset_hf.train_test_split(test_size=test_size) # add seed ?
    train_valid_dataset = train_valid_test_split['train']
    test_dataset_split = train_valid_test_split['test']

    text_features, image_features = MemotionData()



class Data(data.Dataset):
    def __init__(self, path, mode = 'train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        
        if mode == 'train':
            dataset = data['train']
        elif mode == 'valid':
            dataset = data['valid']
        else:
            dataset = data['test']
        text = dataset['text'].astype(np.float32)
        text[text == -np.inf] = 0
        self.text = torch.tensor(text)
        # audio = dataset['audio'].astype(np.float32)
        # audio[audio == -np.inf] = 0
        #self.audio = torch.tensor(audio)
        vision = dataset['vision'].astype(np.float32)
        vision[vision == -np.inf] = 0
        self.vision = torch.tensor(vision)
        self.label = dataset['labels'].astype(np.float32)  ##happy, sad, angry, neutral
      

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        vision = self.vision[index]
        #audio = self.audio[index]
        label = torch.argmax(torch.tensor(self.label[index]), -1)
        return text, vision, label








