import boto3
import os
from io import BytesIO
from contextlib import contextmanager
import huggingface_hub
import torch
from typing import List
import numpy as np

SMALL_CONST = 1e-15
@contextmanager
def s3_fileobj(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    yield BytesIO(obj['Body'].read())

def get_model_file(path_to_model, bucket='ctrldecoder'):
    """Load a model at the given S3 path. It is assumed that your model is stored at the key:"""
    temppath = f'.tmp/{path_to_model}'
    if not os.path.exists(temppath):
        print('Fetching weights from S3 bucket; this may take a moment.')
        with s3_fileobj(bucket, path_to_model) as model_weights:
            with open(temppath, 'wb') as out:
                out.write(model_weights.read())

    return temppath

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator. Originally in PPLM: https://github.com/uber-research/PPLM/blob/master/run_pplm.py
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def old_to_new(old_past_key_values: List[torch.FloatTensor]):
    """Takes old transformers key/values data structure (ie transformers version <4.x
    and converts to the new transformers key value data structure.
    old: List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`)
    new: Tuple of length `config.n_layers` tuple each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    """
    new_past_key_values = []

    for tensor in old_past_key_values:
        # each tensor is 2 x batch_size x heads x seq x embed_size
        tup = (tensor[0], tensor[1])
        new_past_key_values.append(tup)

    return tuple(new_past_key_values)


def new_to_old(new_past_key_values):
    """Takes new transformers key/values data structure (ie transformers version 4+
    and converts to the old transformers key value data structure.
    This is for convenience, as the old version is a list of tensors and we can keep track
    of the gradient for each tensor.
    old: List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`)
    new: Tuple of length `config.n_layers` tuple each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    """
    old_past_key_values = []
    for tup in new_past_key_values:
        K, V = tup
        old_past_key_values.append(torch.cat((K.unsqueeze(0), V.unsqueeze(0)), dim=0))
    return old_past_key_values

def hf_login():
    assert 'HF_TOKEN' in os.environ, 'Please set your HF_TOKEN environment variable to your huggingface token.\
                                      This is necessary for using the Llama2 model.'
    print('HF token', os.environ['HF_TOKEN'])
    huggingface_hub.login(token=os.environ['HF_TOKEN'])

def seed(val=0):
    torch.manual_seed(val)
    np.random.seed(val)
