import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import time
import datetime
from seqeval.metrics import f1_score, accuracy_score, classification_report

file_name = "./nerDataSet/train_data_naver_challenge.txt"
sent = []
sentences = []
label = []
labels = []
setTags= []
for ln in open(file_name):
    if ln == "\n":
        sentences.append(sent)
        labels.append(label)
        sent = []
        label = []
        continue
    num, word, tag = ln.strip().split("\t")
    sent.append(word)
    label.append(tag)
    setTags.append(tag)

sentence = []
tag = []

for s,t in zip(sentences, labels):

    list2StrSent = "[CLS] " + " ".join(s) + " [SEP]"
    list2StrLabel = "- " + " ".join(t) + " -"
    sentence.append(list2StrSent)
    tag.append(list2StrLabel)

se = []
ta = []
for j,k in zip(sentence, tag):
    if j == "\n":
        continue
    word = j.split(" ")
    t = k.split(" ")
    se.append(word)
    ta.append(t)


sentences = se
labels = ta

tag_values = list(set(setTags))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}
print(tag2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))

max_length = 100
batch_size = 32

# tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case = False)
