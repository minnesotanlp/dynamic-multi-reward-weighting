# Abstract class for controlled decoding
from abc import ABC, abstractmethod

class CtrlDecoder(ABC):
    @abstractmethod
    def __init__(self, tokenizer, model, discriminators, labels, verbosity=None):
        #TODO: add other params like topk, topp, etc?
        pass

    @abstractmethod
    def decode(self, prompt, txt_len=60):
        """"responds to a prompt, according to the inheritor's decoding strategy"""
        pass

    @abstractmethod
    def decode_next_token(self, inputs, curr_idx):
        """decode the next token, according to the inheritor's decoding strategy"""
        pass
