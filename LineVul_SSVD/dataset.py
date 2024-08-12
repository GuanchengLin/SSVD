import os
import six
import copy
import random
import pickle
# import logging
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaTokenizerFast
import pandas as pd
import os.path as osp
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class STTextDataset(Dataset):
    def __init__(self, data_path, transformers_model_name, max_seq_len=256, parse_data=True, mode='test', **kwargs):
        self.mode = mode
        self.data_path = data_path
        self.transformers_model_name = transformers_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformers_model_name)
        self.max_seq_len = max_seq_len
        self.cache_num = 40000
        self.cache = {}
        
        if parse_data:
            self.data, self.num_classes = self._parse_data(data_path)

        else:
            assert 'data' in kwargs, "`data` should be provided when parse_data=False"
            assert 'num_classes' in kwargs, "`num_classes` should be provided when parse_data=False"
            self.data = kwargs['data']
            self.num_classes = kwargs['num_classes']

        for i, row in enumerate(self.data.itertuples()):
            text = row.text
            X = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt',
                               add_special_tokens=True, max_length=self.max_seq_len, return_token_type_ids=True,
                               return_attention_mask=True)
            self.data.at[i, 'text'] = X

    def __getitem__(self, index):

        data = self.data.iloc[index]
        X = data['text']

        return {
            "input_ids": X["input_ids"].squeeze(0),
            "token_type_ids": X["token_type_ids"].squeeze(0),
            "attention_mask": X["attention_mask"].squeeze(0),
            "label": data['label'],
            "pseudo_label": data['pseudo_label'],
            "variance": data['variance'],
            "index": data['index']
        }

    def __len__(self):

        return len(self.data)

    def subseting_dataset(self, indices):

        self.old_data = copy.deepcopy(self.data)
        new_data = []
        for idx in indices:
            new_data.append(self.data[idx])

        self.data = new_data

        return self

    def update_data(self, index, content):
        for key in content.keys():
            self.data[key][index] = content[key]

    def clone(self):
        return STTextDataset(self.data_path, self.transformers_model_name,
                             parse_data=False, data=self.data, num_classes=self.num_classes)

    def _parse_data(self, data_path):
        
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
            dataset = dataset[['text', 'label']]

            dataset['pseudo_label'] = 0
            dataset['variance'] = 0.0

            dataset.reset_index(inplace=True)
            dataset['index'] = dataset.index
        num_classes = 0
        if self.mode == 'train':
            label_set = set()
            word_len = [0] * len(dataset)
            for i, row in enumerate(dataset.itertuples()):
                text, label = row.text, row.label
                label_set.add(label)
                X = self.tokenizer(text, padding=False, truncation=False, add_special_tokens=True,
                                   return_token_type_ids=False, return_attention_mask=False)
                word_len[i] = len(X['input_ids'])
            num_classes = len(label_set)
            print(f"Text length (max/min/median): {max(word_len)}/{min(word_len)}/{np.median(word_len)}")

        return dataset, num_classes

class TXTDataset(Dataset):
    def __init__(self, data_path, transformers_model_name, max_seq_len=256, parse_data=True, mode='test', **kwargs):
        self.mode = mode
        self.data_path = data_path
        self.data_root = osp.join(osp.dirname(osp.dirname(osp.dirname(data_path))), 'data')
        self.transformers_model_name = transformers_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformers_model_name)
        self.max_seq_len = max_seq_len
        self.cache_num = 80000
        self.cache = {}
        
        if parse_data:
            self.data, self.num_classes = self._parse_data(data_path)

        else:
            assert 'data' in kwargs, "`data` should be provided when parse_data=False"
            assert 'num_classes' in kwargs, "`num_classes` should be provided when parse_data=False"
            self.data = kwargs['data']
            self.num_classes = kwargs['num_classes']

    def __getitem__(self, index):

        data = self.data.iloc[index]
        if data['path'] in self.cache.keys():
            text_data = self.cache[data['path']]
        else:
            with open(osp.join(self.data_root, data['path']), "rb") as f:
                tdata = pd.read_pickle(f)
                text, label = tdata['text'][0], tdata['label'][0]
                X = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt',
                                   add_special_tokens=True, max_length=self.max_seq_len, return_token_type_ids=True,
                                   return_attention_mask=True)
                if len(self.cache.keys()) < self.cache_num:
                    self.cache.update({f'{data["path"]}': {'text': X, 'label': label}})
                else:
                    random_key = random.choice(list(self.cache.keys()))
                    self.cache.pop(random_key)
                    self.cache.update({f'{data["path"]}': {'text': X, 'label': label}})
            text_data = self.cache[data['path']]

        return {
            "input_ids": text_data['text']["input_ids"].squeeze(0),
            "token_type_ids": text_data['text']["token_type_ids"].squeeze(0),
            "attention_mask": text_data['text']["attention_mask"].squeeze(0),
            "label": text_data['label'],
            "pseudo_label": torch.tensor(data['pseudo_label'], dtype=torch.long),
            "variance": torch.tensor(data['variance']),
            "index": index
        }

    def __len__(self):

        return len(self.data)

    def subseting_dataset(self, indices):

        self.old_data = copy.deepcopy(self.data)
        new_data = []
        for idx in indices:
            new_data.append(self.data[idx])

        self.data = new_data

        return self

    def update_data(self, index, content):
        for key in content.keys():
            self.data[key][index] = content[key]

    def clone(self):
        return TXTDataset(self.data_path, self.transformers_model_name,
                             parse_data=False, data=self.data, num_classes=self.num_classes)

    def _parse_data(self, data_path):
        
        with open(data_path, "r") as file:
            lines = file.readlines()
        data_path_list = [line.strip() for line in lines]
        
        dataset = {'path': data_path_list, 'index': range(0, len(data_path_list)),
                   'pseudo_label': [0] * len(data_path_list), 'variance': [0.0] * len(data_path_list)}
        dataset = pd.DataFrame(dataset)
        label_set = set()

        for i, row in enumerate(dataset.itertuples()):
            with open(osp.join(self.data_root, row.path), "rb") as f:
                text_data = pd.read_pickle(f)
                text, label = text_data['text'][0], text_data['label'][0]
            X = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt',
                               add_special_tokens=True, max_length=self.max_seq_len, return_token_type_ids=True,
                               return_attention_mask=True)
            if len(self.cache.keys()) < self.cache_num:
                self.cache.update({f'{row.path}': {'text': X, 'label': label}})
            label_set.add(label)
        num_classes = len(label_set)
        return dataset, num_classes


if __name__ == '__main__':
    ssl_data = f'ssl_data/reveal'
    slice = 1
    rate = 0.1
    train_dset = TXTDataset(f'{ssl_data}/slice_{slice}/{rate}/train.txt',
                               'codebert', 512, mode='train')
    print(len(train_dset))
    print(train_dset[1])
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, pin_memory=True,
                              num_workers=8)
    def infinite_iter(iterable):
        it = iter(iterable)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(iterable)
    it = infinite_iter(train_loader)
    i = 0
    while True:
        i += 1
        if i % 1000 == 0:
            print(i)
        next(it)