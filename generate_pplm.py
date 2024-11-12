import pandas as pd
from PPLMDecoder import PPLMDecoder
from PPLMShallowFusionDecoder import PPLM_SFDecoder
from peft import PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import AutoTokenizer, LlamaForSequenceClassification, AutoModelForCausalLM, HfArgumentParser
from typing import Optional
from dataclasses import dataclass, field
import torch
from tqdm import tqdm
import numpy as np
import json
from utils import seed

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the base model name"})
    sentiment: Optional[str] = field(default=None, metadata={"help": "what sentiment, if any (positive, negative)"})
    formality: Optional[str] = field(default=None, metadata={"help": "what formality, if any (formal, informal)"})
    prompt_file: Optional[str] = field(default='eval_prompts.csv', metadata={"help": "CSV file containing eval prompts"})
    out_file: Optional[str] = field(default='out_gen.csv', metadata={"help": "path to write outputs to"})
    use_sf: Optional[bool] = field(default=False, metadata={"help": "whether to use shallow fusion PPLM"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

def get_prompt(idx, df):
    return df._get_value(idx, 'prompt')

def get_len(idx, df):
    return len(df._get_value(idx, 'full_text').split(' '))

def baseline_output(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    generation_kwargs = {
        'min_length': -1,
        'top_p': 1.0,
        'top_k': 0,
        'pad_token_id': tokenizer.eos_token_id,
        'do_sample': True,
        'max_length': 50
    }
    with torch.no_grad():
        out = model.generate(input_ids, **generation_kwargs)
        return tokenizer.decode(out[0])

def score_text(text, model, tokenizer):
    token_ids = tokenizer.encode(text, padding='max_length', max_length=2046, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(token_ids)
        return out.logits.tolist()

if __name__ == '__main__':
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    seed()
    prompt_df = pd.read_csv(script_args.prompt_file)
    model_name = 'llama2' if 'llama' in script_args.model_name else ''
    model = LlamaForCausalLM.from_pretrained(script_args.model_name, device_map="cuda").to_bettertransformer()
    model.config.use_cache = True
    tokenizer.pad_token = tokenizer.eos_token
    discriminators = []
    labels = []
    col_str = 'PPLM_'
    if script_args.sentiment in ['positive', 'negative']:
        discriminators.append('sentiment')
        labels.append(script_args.sentiment)
        col_str += script_args.sentiment
    if script_args.formality in ['formal', 'informal']:
        discriminators.append('formality')
        labels.append(script_args.formality)
        col_str += script_args.formality
    
    if script_args.use_sf:
        decoder = PPLM_SFDecoder(tokenizer, model, discriminators=discriminators, labels=labels)
    else:
        decoder = PPLMDecoder(tokenizer, model, discriminators=discriminators, labels=labels)
    decoder.sample = True
    generations = []
    tqdm.pandas()
    prompt_df[col_str] = prompt_df['prompt'].progress_apply(lambda prompt: decoder.decode(prompt))
    print('writing results to', script_args.out_file)
    prompt_df[col_str].to_csv(script_args.out_file, index=False)   
