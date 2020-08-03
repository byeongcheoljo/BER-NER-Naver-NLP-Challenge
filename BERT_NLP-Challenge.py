import pandas as pd
import numpy as np
from tqdm import tqdm, trange
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
for j, k in zip(sentence, tag):
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

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

print(tokenized_texts[1])
print(labels[1])

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen = max_length, dtype = "long", value = 0.0, truncating = "post", padding = "post")
print(input_ids[0])
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen = max_length, value = tag2idx["PAD"], padding = "post", dtype = "long", truncating = "post")
print(tags[0])
attention_masks = [[float(i != 0.0) for i in x] for x in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state = 2018, test_size = 0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state = 2018, test_size = 0.1)
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler = valid_sampler, batch_size = batch_size)

# model = BertForTokenClassification.from_pretrained(
#     "bert-base-cased",
#     num_labels=len(tag2idx),
#     output_attentions = False,
#     output_hidden_states = False
# )

model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(tag2idx), output_attentions = False, output_hidden_states = False)
model.cuda()

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0} ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

# optimizer = AdamW(optimizer_grouped_parameters, lr = 3e-5, eps = 1e-8)
optimizer = AdamW(optimizer_grouped_parameters, lr = 3e-3, eps = 1e-8)

epochs = 10
max_grad_norm = 1.0
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

loss_values, validation_loss_values = [], []
for _ in trange(epochs, desc="Epoch"):
    print("="*10,"training","="*10)
    t0 = time.time()
    model.train()
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask = b_input_mask, labels = b_labels)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = max_grad_norm)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    loss_values.append(avg_train_loss)

    print("="*10,"validation","="*10)
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        ##gradient 계산 안함
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids = None,
                            attention_mask = b_input_mask, labels = b_labels)
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis = 2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()
    print(classification_report(pred_tags, valid_tags))
