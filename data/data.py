import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from config import Config

# Data 
def create_folds(data_path):
    df = pd.read_csv(data_path)
    df = df.drop('Unnamed: 0', axis=1) # remove column 1 
    df = df.sample(frac=1).reset_index(drop=True) # randomize the order of the rows
    df['label'] = df['label']
    df['label'] = np.where(df['label'] == 'very_positive', 'positive', df['label'])
    df['label'] = np.where(df['label'] == 'very_negative', 'negative', df['label'])

    
    # cross validation technique that ensures that each fold has same proportion of classes as the whole data set
    mskf = StratifiedKFold(n_splits=5) # divide data into 5 folds for cross validation
    

    # df['kfold'] = -1
    # for fold, (train, valid) in enumerate(mskf.split(X=df, y=df['label'])):
    #     df.loc[valid, 'kfold'] = fold
    
    # df['label'] = df['label'].map({
    #     'positive': 0, 
    #     'neutral': 1,
    #     'negative':2
    # })

    df.to_csv('folds.csv', index=False)

import os
import numpy as np
from PIL import Image

class MemotionDataset():
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer)
        self.transforms = A.Compose([
            A.Resize(height=Config.img_size[0], width=Config.img_size[1]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[ix]

        # Construct the correct image path
        image_path = os.path.join('memotion_dataset_7k', 'images', row['image_name'])

        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')  # Ensure RGB format
        img = np.array(img)  # Convert to NumPy array
        img = self.transforms(image=img)['image']  # Apply transformations

        # Tokenize text
        text = str(row['text_corrected']).lower()
        out = self.tokenizer(
            text=text, 
            max_length=Config.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'Image': img, 
            'input_ids': out['input_ids'].squeeze(),
            'attention_mask': out['attention_mask'].squeeze(),
            'label': torch.LongTensor([row['label']]).squeeze()
        }
if __name__=="__main__":
    create_folds(Config.data_path)