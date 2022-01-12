import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from transformers import Trainer
from tqdm.auto import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import matplotlib.pyplot as plt
from datasets import load_metric

from Model import RNN, CNN, LSTM

def tokenize_function(example):
    return tokenizer(example["Phrase"], truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_vocab_size(phrase_dataset):
    vocab = []
    for phrase in phrase_dataset:
        for word in phrase.split():
            if word not in vocab:
                vocab.append(word)
    return len(vocab)

num_classes = 5
batch_size = 512
learning_rate = 0.001
dropout_rate = 0.1
embed_size = 200
hidden_size = 256
num_layers = 1
bidirectional = True

if __name__ == "__main__":
    data_files = {"train": "data/train.tsv", "validation": "data/validation.tsv", "test": "data/test.tsv"}
    data = load_dataset("csv", data_files=data_files, delimiter="\t")
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)
    vocab_size = get_vocab_size(data['train']['Phrase'])
    model = LSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate)
    tokenized_datasets = data.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["PhraseId", "SentenceId"])
    tokenized_datasets = tokenized_datasets.rename_column("Sentiment", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["Phrase"])
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments("test-trainer")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=16, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=16, collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=0.0001)

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    progress_bar = tqdm(range(num_training_steps))

    #model.train()
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if idx % 100 == 0:
                loss_list.append(loss.item())
                tqdm.write('step:{}, loss :{}'.format(idx/100, loss.item()))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    plt.plot(np.arange(len(loss_list)), np.array(loss_list))
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('distilled-bert-uncased')
    plt.show()

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()