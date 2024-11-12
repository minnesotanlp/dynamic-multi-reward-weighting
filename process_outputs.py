import pandas as pd
import sys
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import AutoTokenizer, LlamaForSequenceClassification, AutoModelForCausalLM, HfArgumentParser
#from trl import AutoModelForCausalLMWithValueHead
from typing import Optional
from dataclasses import dataclass, field
import torch
import numpy as np
import json
from utils import seed

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "the model name (used to instantiate tokenizer)"})
    generations_file: Optional[str] = field(default="output.csv", metadata={"help": "file containing outputs to be evaluated"})
    out_file: Optional[str] = field(default='classified_texts.csv', metadata={"help": "file to write results to"})
    formality_discrim: Optional[str] = field(default=None, metadata={"help": "path to formality discrim"})
    sentiment_discrim: Optional[str] = field(default=None, metadata={"help": "path to fine tuned sentiment discrim"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def score_text(text, model, tokenizer):
    text = text.strip('<s>')
    text = text.strip('</s>')
    text = text.strip()
    token_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(token_ids)
        return torch.nn.functional.softmax(out.logits).tolist()

def score_text_in_batches(series, model, tokenizer, batch_size=32):
    num_samples = len(series)
    batch_results = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_texts = series[start:end].tolist()
        batch_scores = score_text(batch_texts, model, tokenizer)
        batch_results.extend(batch_scores)

    return batch_results

if __name__ == '__main__':
    seed()
    prompt_df = pd.read_csv(script_args.prompt_file)
    cols = [c for c in prompt_df.columns if ('rl_' in c or 'PPLM_' in c) and ('sentiment' not in c) and ('formality' not in c)]
    model = LlamaForSequenceClassification.from_pretrained(script_args.sentiment_discrim, device_map='cuda')
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for col in cols:
        if f'{col} sentiment' in prompt_df.columns:
            print('skipping col (already exists)', col)
            continue
        prompt_df[f'{col} sentiment'] = prompt_df[col].apply(lambda output: score_text(output, model, tokenizer))
        prompt_df.to_csv(script_args.out_file)
    del model
    model = LlamaForSequenceClassification.from_pretrained(script_args.formality_discrim, device_map='cuda')
    model = model.eval()
    for col in cols:
        if f'{col} formality' in prompt_df.columns:
            print('skipping col (already exists)', col)
            continue
        prompt_df[f'{col} formality'] = prompt_df[col].apply(lambda output: score_text(output, model, tokenizer))
        prompt_df.to_csv(script_args.out_file)
    
