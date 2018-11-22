import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from PIL import Image

class RSNA_Dataset(Dataset):

    def __init__(
            self,
            path_to_images,
            transform=None,
            mode='train'):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv("rsna_labels.csv")
        self.df['patientId'] = self.df['patientId']+".png"
        self.df = self.df.drop_duplicates('patientId')
        
        np.random.seed(42)
        msk = np.random.rand(len(self.df)) < 0.8
        
        if (mode == 'train'):
            self.df = self.df[msk]
        elif (mode == 'val'):
            self.df = self.df[~msk]

        self.df = self.df.set_index("patientId")
        
        self.PRED_LABEL = [
            'Pneumonia',
            'not Pneumonia'
            ]

        RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        if (self.df['Target'].iloc[idx].astype('int') > 0):
            label[0] = 1 #pneumonia
            
        if self.transform:
            image = self.transform(image)

        return (image, label,self.df.index[idx])

