import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import Dataset,DataLoader
from config import Config
from prompts import news_cls
config = Config()


def read_label_map_dict():
    labelmap = {}
    with open("../data/label_index2en2zh.json",'r',encoding="utf-8") as fr:
        for line in fr:
            line = json.loads(line)
            label = str(line["label"])
            label_zh = line["label_zh"]
            labelmap[label]=label_zh
    return labelmap




def read_data(data_file):
    labelmap = read_label_map_dict()
    texts = []
    labels = []
    with open(data_file,'r',encoding="utf-8") as fr:
        for line in fr:
            line = json.loads(line)
            line_prompt = news_cls(line["sentence"])
            texts.append(line_prompt)
            labels.append(labelmap[str(line["label"])])
    return texts,labels




class T5Dataset(Dataset):
    def __init__(self,data_file):
        super(T5Dataset, self).__init__()
        self.texts,self.labels = read_data(data_file)
        # self.set_type = set_type
        # if self.set_type == 'test':
        #     pass
            # self.labels = get_labels(df)

        self.tokenizer = config.TOKENIZER
        self.src_max_length = config.SRC_MAX_LENGTH
        self.tgt_max_length = config.TGT_MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        src_tokenized = self.tokenizer.encode_plus(
            self.texts[index],
            max_length=self.src_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        # if self.set_type != 'test':
        tgt_tokenized = self.tokenizer.encode_plus(
            self.labels[index],
            max_length=self.tgt_max_length,
            # pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
        tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()

        return {
            'src_input_ids': src_input_ids.long(),
            'src_attention_mask': src_attention_mask.long(),
            'tgt_input_ids': tgt_input_ids.long(),
            'tgt_attention_mask': tgt_attention_mask.long()
        }

        # return {
        #     'src_input_ids': src_input_ids.long(),
        #     'src_attention_mask': src_attention_mask.long()}