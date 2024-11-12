import argparse
import csv
import json
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from datasets.dataset_dict import DatasetDict
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.optim
from trl import PPOTrainer
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaConfig, LlamaModel, LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments, T5ForSequenceClassification, T5Tokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Callable, Optional, Union
from datasets import load_dataset
from pathlib import Path
import sys
from utils import hf_login, seed

hf_login()

seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        required=True,
        type=str,
        help="Dataset to train classifier on: 'sst2' 'imdb' 'semeval' or 'gyafc'"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Location for model outputs and checkpoints"
    )
    parser.add_argument("--num_epochs", default="1", type=int, help="num training epochs")
    parser.add_argument("--model-name-or-path", required=True, type=Path, help="model weights location")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Training and evaluation batch size"
    )
    parser.add_argument("--save_file", default="classification_head", help="where to save final classification head")
    parser.add_argument("--eval", action="store_true", help="run only evaluation (no training)")
    args = parser.parse_args()
    return args


def get_semeval_dataset(tokenizer):
    restaurants = load_dataset('yqzheng/semeval2014_restaurants')
    laptops = load_dataset('yqzheng/semeval2014_laptops')
    train_dataset = concatenate_datasets([restaurants['train'], laptops['train']], axis=0)
    test_dataset = concatenate_datasets([restaurants['test'], laptops['test']], axis=0)
    def tokenize_function(examples):
        return tokenizer(examples['text'], max_length=2046, padding='max_length', truncation=True)
    tokenized_train = train_dataset.map(tokenize_function)
    tokenized_test = test_dataset.map(tokenize_function)
    tokenized_train = tokenized_train.class_encode_column('label')
    tokenized_test = tokenized_test.class_encode_column('label').shuffle(seed=2).select(range(500))
    
    return tokenized_train, tokenized_test, None


def get_imdb_dataset(tokenizer):
    dataset = load_dataset('imdb')
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=2046, truncation=True)
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_train = tokenized_dataset['train']
    test_len = len(tokenized_dataset['test'])
    test_midpoint = 2 * round(test_len / 10)
    tokenized_test = tokenized_dataset['test'].shuffle(seed=2).select(range(0, test_midpoint))
    tokenized_train = concatenate_datasets([tokenized_train, tokenized_dataset['test'].shuffle(seed=2).select(range(test_midpoint, test_len))])
    return tokenized_train, tokenized_test, None


def get_sst_dataset(tokenizer):
    dataset = load_dataset('SetFit/sst2')
    tokenizer.pad_token_id = 2
    def tokenize_function(examples):
        return tokenizer(examples['text'], max_length=2048, padding='max_length', truncation=True)
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_train = tokenized_dataset['train']
    tokenized_val = tokenized_dataset['validation']
    #tokenized_val = tokenized_dataset['validation'].shuffle(seed=2).select(range(200))
    tokenized_test = tokenized_dataset['test']
    return tokenized_train, tokenized_val, tokenized_test
   

def read_gyfac_file(f):
    labels = []
    text = []
    for line in f.readlines():
        line = line.split()
        label = line[0].strip('<>')
        labels.append(label)
        text.append(' '.join(line[1:]))
    return text, labels

def create_formality_dataset(tokenizer_path):
    train_file = 'GYFAC/train.bi.src'
    val_file = 'GYFAC/dev.bi.src'
    test_file = 'GYFAC/test.bi.src'
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = 2
    data_dict = {'train': {'label': [], 'sentence': []},
                 'val': {'label': [], 'sentence': []},
                 'test': {'label': [], 'sentence': []}}

    with open(train_file) as train, open(val_file) as val, open(test_file) as test:
        for file, name in zip([train, val, test], ['train', 'val', 'test']):
            text, labels = read_gyfac_file(file)
            data_dict[name]['sentence'] = text
            data_dict[name]['label'] = labels
    dataset = DatasetDict({'train': Dataset.from_dict(data_dict['train']), 'test': Dataset.from_dict(data_dict['test']), 'val': Dataset.from_dict(data_dict['val'])})
    dataset.save_to_disk('datasets/formality')

def get_formality_dataset(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], max_length=2048, padding='max_length', truncation=True)
    dataset = load_from_disk('datasets/formality')
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.class_encode_column('label')
    tokenized_train = tokenized_dataset['train']
    tokenized_val = tokenized_dataset['val']
    tokenized_test = tokenized_dataset['test']
    return tokenized_train, tokenized_val, tokenized_test


def get_multi_dataset():
    raise NotImplementedError

def main():
    args = parse_args()
    num_labels = 3 if args.style == 'semeval' else 2
   
    if 'llama' in str(args.model_name_or_path):
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        discrim = LlamaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                 device_map='auto',
                                                                 num_labels=num_labels,
                                                                 problem_type='single_label_classification')
        discrim.config.pad_token_id = tokenizer.pad_token_id
        for param in discrim.model.parameters():
            param.requires_grad = False
    else:
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
        discrim = T5ForSequenceClassification.from_pretrained(args.model_name_or_path, device_map='auto', num_labels=num_labels, problem_type='single_label_classification')
        for param in discrim.transformer.parameters():
            param.requires_grad = False
    if args.style == 'sst2':
        tokenized_train, tokenized_val, tokenized_test = get_sst_dataset(tokenizer)
    elif args.style == 'semeval':
        tokenized_train, tokenized_val, tokenized_test = get_semeval_dataset(tokenizer)
    elif args.style == 'imdb':
        tokenized_train, tokenized_val, tokenized_test = get_imdb_dataset(tokenizer)
    elif args.style == 'gyafc':
        tokenized_train, tokenized_val, tokenized_test = get_formality_dataset(tokenizer)
    elif args.style == 'multi':
        tokenized_Train, tokenized_val, tokenized_test = get_multi_dataset(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if 't5' in str(args.model_name_or_path):
             logits = logits[0]
        predictions = np.argmax(np.asarray(logits), axis=-1)
        return {'accuracy': accuracy_score(predictions, labels), 'precision': precision_score(predictions, labels, average='micro'), 'recall': recall_score(predictions, labels, average='micro'), 'f1': f1_score(predictions, labels, average='micro')}

    training_args = TrainingArguments(output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy='steps',
            save_total_limit=2,
            save_strategy='steps',
            logging_steps=10,
            gradient_accumulation_steps=8,
            eval_steps=200,
            save_steps=200,
            num_train_epochs=args.num_epochs,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy')

    trainer = Trainer(
        model=discrim,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )

    if args.eval:
        e = trainer.evaluate()
        print(e)
    else:
        from_checkpoint = 'checkpoint' in str(args.model_name_or_path)
        trainer.train(resume_from_checkpoint=from_checkpoint)
        # write classification head to file
        path = args.output_dir / args.save_file
        torch.save(discrim.score.state_dict(), path)

if __name__ == '__main__':
    main()
