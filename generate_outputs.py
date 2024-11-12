import pandas as pd
import sys
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import AutoTokenizer, LlamaForSequenceClassification, AutoModelForCausalLM, HfArgumentParser
from tqdm.auto import tqdm
from typing import Optional
from dataclasses import dataclass, field
import torch
import numpy as np
import json
from utils import seed

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "Base model name (used to instatiate tokenizer)"})
    model_type: Optional[str] = field(default="positive", metadata={"help": "String describing control type (used in output CSV file)"})
    trained_model: Optional[str] = field(default=None, metadata={"help": "Path to fine tuned style control model"})
    prompt_file: Optional[str] = field(default='extended_prompts.csv', metadata={"help": "CSV file containing eval prompts"})
    out_file: Optional[str] = field(default='generated.csv', metadata={"help": "filepath to store generation and classification outputs"})
    formality_discrim: Optional[str] = field(default=None, metadata={"help": "path to formality discriminator"})
    sentiment_discrim: Optional[str] = field(default=None, metadata={"help": "path to sentiment discriminator"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
def seed(val=0):
    torch.manual_seed(val)
    np.random.seed(val)

def get_output(prompt, model, tokenizer):
    prompt = prompt.replace('<s>', '').replace('</s>', '')
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    generation_kwargs = {
        'min_length': 25,
        'top_p': 1.0,
        'top_k': 0,
        'pad_token_id': tokenizer.eos_token_id,
        'do_sample': True,
        'repetition_penalty': 1.5,
        'max_length': 60
    }
    with torch.no_grad():
        out = model.generate(input_ids, **generation_kwargs)
        text = tokenizer.decode(out[0])
        return text

def score_text(text, discrim, tokenizer):
    """Returns logits of given text"""
    token_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = discrim(token_ids)
        return out.logits.tolist()

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

    prompt_df = pd.read_csv(script_args.prompt_file)
    model_name = 'llama2' if 'llama2' in script_args.model_name else script_args.model_name  # shorten string for convenience
    tqdm.pandas(desc="bar")

    # Generate and classify outputs from baseline model
    baseline_model_column = f'baseline {model_name}'
    if baseline_model_column not in prompt_df.columns:
       model = AutoModelForCausalLM.from_pretrained(script_args.base_model_name)
       prompt_df[f'baseline {model_name}'] = prompt_df['prompt'].apply(lambda prompt: get_output(prompt, model, tokenizer))
       prompt_df.to_csv(script_args.out_file, index=False)
       del model

    if f'baseline {model_name} sentiment' not in prompt_df.columns:
       print('classifying sentiment of baseline generations')
       model = LlamaForSequenceClassification.from_pretrained(script_args.sentiment_discrim, device_map='cuda')
       model.eval()
       prompt_df[f'baseline {model_name} sentiment'] = prompt_df[f'baseline {model_name}'].apply(lambda prompt: score_text(prompt, model, tokenizer))
       prompt_df.to_csv(script_args.out_file, index=False)
       del model

    if f'baseline {model_name} formality' not in prompt_df.columns:
       print('classifying formality of baseline generations')
       model = LlamaForSequenceClassification.from_pretrained(script_args.formality_discrim, device_map='cuda')
       model.eval()
       prompt_df[f'baseline {model_name} formality'] = prompt_df[f'baseline {model_name}'].apply(lambda prompt: score_text(prompt, model, tokenizer))
       prompt_df.to_csv(script_args.out_file, index=False)
       del model

    # Classify prompts themselves
    if 'prompt sentiment' not in prompt_df.columns:
       model = LlamaForSequenceClassification.from_pretrained(script_args.sentiment_discrim, device_map='cuda')
       model.eval()
       print("classifying sentiment of prompts")
       prompt_df['prompt sentiment'] = prompt_df['prompt'].apply(lambda prompt: score_text(prompt, model, tokenizer))
       prompt_df.to_csv(script_args.out_file, index=False)
       del model

    if 'prompt formality' not in prompt_df.columns:
       print("classifying formality of prompts")
       model = LlamaForSequenceClassification.from_pretrained(script_args.formality_discrim, device_map='cuda')
       model.eval()
       prompt_df['prompt formality'] = prompt_df['prompt'].apply(lambda prompt: score_text(prompt, model, tokenizer))
       prompt_df.to_csv(script_args.out_file, index=False)
       del model

    seed()
    base_model_name = script_args.model_name
    adapter_model_name = script_args.trained_model

    # Generate outputs from trained style control model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_model_name)
    model = model.merge_and_unload().to('cuda')

    print('Generating prompt responses')
    prompt_df[f'{script_args.model_type} text'] = prompt_df['prompt'].progress_apply(lambda prompt: get_output(prompt, model, tokenizer))
    prompt_df.to_csv(script_args.out_file, index=False)
