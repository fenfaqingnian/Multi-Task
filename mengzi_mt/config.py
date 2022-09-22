import torch
from transformers import T5Tokenizer

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.TRAIN_FILE = '../data/train.json'
        self.VAL_FILE = '../data/val.json'
        self.MODEL_PATH = '../pretrain'

        # data
        self.TOKENIZER = T5Tokenizer.from_pretrained(self.MODEL_PATH)
        self.SRC_MAX_LENGTH = 512
        self.TGT_MAX_LENGTH = 2
        self.BATCH_SIZE = 8
        self.GRAD_ACCUM = 3
        self.VALIDATION_SPLIT = 0.2

        # model
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = True
        self.N_VALIDATE_DUR_TRAIN = 1
        self.EPOCHS = 10