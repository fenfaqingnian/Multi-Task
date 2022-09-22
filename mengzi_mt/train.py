#coding:utf-8

import torch
import numpy as np
from tqdm import tqdm
from config import Config
from dataset import read_label_map_dict
from sklearn.metrics import accuracy_score, f1_score, classification_report
config = Config()
device = config.DEVICE


def train(model,train_dataloader,criterion,optimizer,scheduler,epoch):
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = config.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)

    train_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_dataloader,desc='Epoch ' + str(epoch))):
        # set model.eval() every time during training
        model.train()

        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)

        lm_labels = batch['tgt_input_ids'].to(device)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # clear accumulated gradients

        # forward pass
        outputs = model(input_ids=b_src_input_ids,
                        attention_mask=b_src_attention_mask,
                        lm_labels=lm_labels,
                        decoder_attention_mask=b_tgt_attention_mask)

        loss = outputs[0]
        train_loss += loss.item()

        #  average loss (grad accum.)
        loss = loss / config.GRAD_ACCUM

        # backward pass
        loss.backward()

        if step % config.GRAD_ACCUM == 0:
            # update weights
            optimizer.step()
            optimizer.zero_grad()

        # update scheduler
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)



# 评估时
def val(model, val_dataloader, label_stats=False):
    val_loss = 0
    true, pred = [], []

    # set model.eval() every time during evaluation
    model.eval()

    for step, batch in enumerate(val_dataloader):
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)

        b_tgt_input_ids = batch['tgt_input_ids']
        lm_labels = b_tgt_input_ids.to(device)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # using torch.no_grad() during validation/inference is faster -
        # - since it does not update gradients.
        with torch.no_grad():
            # forward pass
            outputs = model(
                input_ids=b_src_input_ids,
                attention_mask=b_src_attention_mask,
                lm_labels=lm_labels,
                decoder_attention_mask=b_tgt_attention_mask)
            loss = outputs[0]

            val_loss += loss.item()

            # get true
            for true_id in b_tgt_input_ids:
                true_decoded = config.TOKENIZER.decode(true_id)
                true.append(true_decoded)

            # get pred (decoder generated textual label ids)
            pred_ids = model.t5_model.generate(
                input_ids=b_src_input_ids,
                attention_mask=b_src_attention_mask
            )
            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = config.TOKENIZER.decode(pred_id)
                pred.append(pred_decoded)

    true_ohe = get_ohe(true)
    pred_ohe = get_ohe(pred)

    # if label_stats:
    #     true_count = np.sum(true_ohe, axis=0)
    #     pred_count = np.sum(pred_ohe, axis=0)
    #     labels = [i for i in range(len(true_count))]
    #     plt.bar(labels, true_count, label='true specialties count')
    #     plt.bar(labels, pred_count, label='pred specialties count')

    avg_val_loss = val_loss / len(val_dataloader)
    print('Val loss:', avg_val_loss)
    print('Val accuracy:', accuracy_score(true_ohe, pred_ohe))

    val_micro_f1_score = f1_score(true_ohe, pred_ohe, average='micro')
    val_macro_f1_score = f1_score(true_ohe, pred_ohe, average='macro')
    print('Val micro f1 score:', val_micro_f1_score)
    print('Val macro f1 score:', val_macro_f1_score)
    return val_micro_f1_score, val_macro_f1_score


def get_ohe(x):
    label2idx = {}
    num2label = read_label_map_dict()
    for k,v in num2label.items():
        if int(k)<=104:
            label2idx[v]=int(k)-100
        elif int(k)<=110:
            label2idx[v] = int(k) - 101
        else:
            label2idx[v] = int(k) - 102
    # print(label2idx)
    clean_x = [xl.replace('<pad>', '').replace('</s>', '').replace(' ', '') for xl in x]
    print(clean_x)
    y = [labels.split(',') for labels in clean_x]
    ohe = []
    for labels in y:
        # print("predicted labels: ", labels)
        temp = [0] * 15
        for label in labels:
            if label in label2idx.keys():
                idx = label2idx[label]
                # print("label: ", label, "idx: ", idx)
                temp[idx] = 1
        ohe.append(temp)
    ohe = np.array(ohe)
    return ohe
