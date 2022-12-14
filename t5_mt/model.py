# coding:utf-8

from numpy import datetime_data
from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from torch import nn
import numpy as np
import torch
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")

# 在多任务数据集上微调

class MNLIT5(nn.Module):
    def __init__(self, original_t5: bool, checkpoint: str, soft_prompt: bool, prefix: str, infix: str, suffix: str):
        super(MNLIT5, self).__init__()
        self.original_t5 = original_t5
        self.config = T5Config.from_pretrained(checkpoint)
        self.t5 = T5ForConditionalGeneration.from_pretrained(checkpoint, config=self.config)
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.soft_embedding_layer = None
        self.normal_embedding_layer = self.t5.get_input_embeddings()
        self.soft_prompt = soft_prompt
        if soft_prompt:

            self.prefix_soft_index, self.infix_soft_index, self.suffix_soft_index = eval(prefix), eval(infix), eval(
                suffix)
            # [3,27569,10],[11167,10],[31484,17,10,1]
            self.p_num, self.i_num, self.s_num = len(self.prefix_soft_index), len(self.infix_soft_index), len(
                self.suffix_soft_index)
            self.prefix_soft_embedding_layer = nn.Embedding(
                self.p_num, self.config.hidden_size
            )
            self.infix_soft_embedding_layer = nn.Embedding(
                self.i_num, self.config.hidden_size
            )
            self.suffix_soft_embedding_layer = nn.Embedding(
                self.s_num, self.config.hidden_size
            )

            self.prefix_soft_embedding_layer.weight.data = torch.stack(
                [self.normal_embedding_layer.weight.data[i, :].clone().detach().requires_grad_(True) for i in
                 self.prefix_soft_index]
            )
            self.infix_soft_embedding_layer.weight.data = torch.stack(
                [self.normal_embedding_layer.weight.data[i, :].clone().detach().requires_grad_(True) for i in
                 self.infix_soft_index]
            )
            self.suffix_soft_embedding_layer.weight.data = torch.stack(
                [self.normal_embedding_layer.weight.data[i, :].clone().detach().requires_grad_(True) for i in
                 self.suffix_soft_index]
            )
            self.prefix_soft_ids = torch.tensor(range(self.p_num))
            self.infix_soft_ids = torch.tensor(range(self.i_num))
            self.suffix_soft_ids = torch.tensor(range(self.s_num))
            for param in self.t5.parameters():
                param.requires_grad_(False)

    def forward(self, input_ids, attention_mask, hypothesis_ids, premise_ids, hypothesis_attention_mask,
                premise_attention_mask, labels):
        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.zeros(batch_size, 1, dtype=int).to(input_ids.device)
        if self.soft_prompt:

            prefix_soft_ids = torch.stack([self.prefix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            infix_soft_ids = torch.stack([self.infix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            suffix_soft_ids = torch.stack([self.suffix_soft_ids for i in range(batch_size)]).to(input_ids.device)

            prefix_soft_embeddings = self.prefix_soft_embedding_layer(prefix_soft_ids)
            infix_soft_embeddings = self.infix_soft_embedding_layer(infix_soft_ids)
            suffix_soft_embeddings = self.suffix_soft_embedding_layer(suffix_soft_ids)

            hypothesis_embeddings = self.normal_embedding_layer(hypothesis_ids)
            premise_embeddings = self.normal_embedding_layer(premise_ids)

            input_embeddings = torch.cat(
                [prefix_soft_embeddings, hypothesis_embeddings, infix_soft_embeddings, premise_embeddings,
                 suffix_soft_embeddings],
                dim=1
            )

            prefix_soft_attention_mask = torch.ones(batch_size, self.p_num).to(input_ids.device)
            infix_soft_attention_mask = torch.ones(batch_size, self.i_num).to(input_ids.device)
            suffix_soft_attention_mask = torch.ones(batch_size, self.s_num).to(input_ids.device)

            attention_mask = torch.cat(
                [prefix_soft_attention_mask, hypothesis_attention_mask, infix_soft_attention_mask,
                 premise_attention_mask, suffix_soft_attention_mask],
                dim=1
            )
            if self.original_t5:
                batch_loss = self.t5(
                    inputs_embeds=input_embeddings,
                    labels=labels,
                    attention_mask=attention_mask,
                    return_dict=True
                ).loss
                return None, batch_loss
            else:
                output = self.t5(
                    inputs_embeds=input_embeddings,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        else:
            if self.original_t5:
                batch_loss = self.t5(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    return_dict=True
                ).loss
                return None, batch_loss
            else:
                output = self.t5(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        logits = output.logits
        batch_score = logits[:, 0, [1176, 7163, 6136]]
        return batch_score, None
