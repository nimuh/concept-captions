
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import pandas as pd
import numpy as np
from PIL import Image
import requests
from collections import Counter


class ConceptCapDataset(Dataset):
    def __init__(self, tsv_file, samples, language):
        self.dataframe = pd.read_csv(tsv_file, nrows=samples)
        self.dataframe = self.dataframe[self.dataframe.columns[1:]]
        print(self.dataframe)
        self.vocab = self.build_vocab(language=language)
        self.data_process()
        self.remove = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        im = None
        try:
            im = Image.open(requests.get(self.data[idx][0], timeout=20, stream=True).raw)
        except:
            print('Could not load URL: ', self.dataframe.iloc[idx, 1])
            self.remove.append(idx)
        if im != None:
            im = np.array(im.resize((64, 64)))
            print(im)
            
        caption = self.data[idx][1]
        sample = {'img': im, 'text': caption}
        return sample

    def build_vocab(self, language):
        self.tokenizer = get_tokenizer('spacy', language=language)
        counter = Counter()
        for i in range(len(self.dataframe)):
            counter.update(self.tokenizer(self.dataframe.iloc[i, 0]))
        return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def data_process(self):
        self.data = []
        for i in range(len(self.dataframe)):
            caption = torch.tensor([self.vocab[token] for token in self.tokenizer(self.dataframe.iloc[i, 0].rstrip("\n"))], dtype=torch.long)
            self.data.append((self.dataframe.iloc[i, 1], caption))
            




SAMPLES = 1000
LANGUAGE = 'en_core_web_sm'
TR_DATA_FILE = 'gcc_training_filtered_1000'

dataset = ConceptCapDataset(TR_DATA_FILE, samples=SAMPLES, language=LANGUAGE)


for i in range(len(dataset)):
    sample = dataset[i]
    #print(i, sample['text'])

#print('REMOVING ', dataset.remove)
#dataset.dataframe.drop(dataset.dataframe.index[[dataset.remove]]).to_csv('gcc_training_filtered_' + str(SAMPLES))

        