import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from kobert_tokenizer import KoBERTTokenizer
from transformers import PreTrainedTokenizerFast, BertTokenizer

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx=0, label_idx=1, max_len=64, add_token=0):
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.vocab_size = self.tokenizer.vocab_size
        if add_token:
            self.added_token_num = self.tokenizer.add_tokens(add_token)


        self.sentences = [self.transform(i[sent_idx]) for i in self.dataset]
        self.labels = [np.int32(i[label_idx]) for i in self.dataset]

    def transform(self, data):
        data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
        data = self.tokenizer(data, max_length=self.max_len, padding="max_length", truncation=True,)
        return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx] + (self.labels[idx],)




class GPT2Dataset(Dataset):
    def __init__(self, dataset, max_len=40):
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                    bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',)
        self.q_token = "<usr>"
        self.a_token = "<sys>"
        self.mask_token = '<unused0>'
        self.sent_token = '<unused1>'

        ######
        self.categories = []
        ######

    def transform(self, data):
        question = data["Q"]
        question = re.sub(r"([?.!,])", r" ", question)

        sentiment = str(data['label'])

        q_toked = self.tokenizer.tokenize(self.q_token + question + self.sent_token + sentiment)
        q_len = len(q_toked)

        answer = data["A"]
        answer = re.sub(r"([?.!,])", r" ", answer)
        a_toked = self.tokenizer.tokenize(self.a_token + answer + self.tokenizer.eos_token)
        a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            q_len = self.max_len - 20
            a_len = self.max_len - q_len
            q_toked = q_toked[:q_len - 1] + q_toked[-1:]
            a_toked = a_toked[:a_len - 1] + a_toked[-1:]

        # token_ids
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        # mask
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        # labels
        labels = [self.mask_token, ] * q_len + a_toked[1:]
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        # return token_ids, np.array(mask), labels_ids

        return token_ids, np.array(mask), labels_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        return self.transform(data)




class kcBERTDataset(Dataset):
    def __init__(self, dataset, sent_idx=0, label_idx=1, max_len=64, add_token=0):
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
        self.vocab_size = self.tokenizer.vocab_size
        if add_token:
            self.added_token_num = self.tokenizer.add_tokens(add_token)


        self.sentences = [self.transform(i[sent_idx]) for i in self.dataset]
        self.labels = [np.int32(i[label_idx]) for i in self.dataset]

    def transform(self, data):
        data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
        data = self.tokenizer(data, max_length=self.max_len, padding="max_length", truncation=True,)
        return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx] + (self.labels[idx],)