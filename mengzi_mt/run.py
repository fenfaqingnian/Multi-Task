# coding:utf-8
import torch
import os
import numpy as np
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
import copy
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from config import Config
from train import train,val
from dataset import T5Dataset
from model import T5Model
from sklearn.metrics import accuracy_score, f1_score, classification_report
config = Config()

# ref:https://github.com/hmichaeli/t5_classification/blob/main/t5_notebook.ipynb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(train_dataloader,val_dataloader):
    print("run training")
    # setting a seed ensures reproducible results.
    # seed may affect the performance too.
    model = T5Model()
    model.to(config.DEVICE)
    torch.manual_seed(config.SEED)

    criterion = nn.BCEWithLogitsLoss()

    # define the parameters to be optmized-
    # - and add regularization
    if config.FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=config.LR)

    num_training_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)
    max_val_macro_f1_score = 0
    print("start training:")
    for epoch in range(config.EPOCHS):
        print("train epoch:")
        train(model, train_dataloader, criterion, optimizer, scheduler, epoch)
        print("evaluate:")
        val_micro_f1_score, val_macro_f1_score = val(model, val_dataloader)
        print("dummy test:")
        # dummy_test(model, config.TOKENIZER)

        if config.SAVE_BEST_ONLY:
            if val_macro_f1_score > max_val_macro_f1_score:
                best_model = copy.deepcopy(model)
                best_val_macro_f1_score = val_macro_f1_score

                model_name = 't5_best_model'
                torch.save(best_model.state_dict(), model_name + '.pt')

                print(f'--- Best Model. Val loss: {max_val_macro_f1_score} -> {val_macro_f1_score}')
                max_val_macro_f1_score = val_macro_f1_score

                return best_model, best_val_macro_f1_score


if "__main__"==__name__:
    trainDataset = T5Dataset(config.TRAIN_FILE)
    valDataset = T5Dataset(config.VAL_FILE)
    train_dataloader = DataLoader(trainDataset,batch_size=config.BATCH_SIZE,shuffle=True)
    val_dataloader = DataLoader(valDataset,batch_size=config.BATCH_SIZE)
    _,best_val_macro_f1_score = run(train_dataloader,val_dataloader)


