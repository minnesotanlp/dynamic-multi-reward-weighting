from PPLMDecoder import PPLMDecoder, new_to_old, old_to_new
import torch
from utils import top_k_filter
import torch.nn.functional as F

class PPLM_SFDecoder(PPLMDecoder):
    def __init__(self, tokenizer, model_str, bow=None, discriminators=['sentiment', 'clickbait'],
                 labels=['very_positive', 'clickbait']):
        if len(labels) != 2:
            raise ValueError('need exactly two labels for now')
        assert len(discriminators) == len(labels) == 2, 'Need exactly 2 discrims (and assoc labels) for this class'
        print('discrimssf', discriminators)
        print('labelssf', labels)
        print([discriminators[0]])
        print([labels[0]])
        # set up two different attribute models; we'll use these to get separate perturbed probabilities
        # that are combined post-norm
        self.attribute1_model = PPLMDecoder(tokenizer, model_str, discriminators=[discriminators[0]], labels=[labels[0]])
        self.attribute2_model = PPLMDecoder(tokenizer, model_str, discriminators=[discriminators[1]], labels=[labels[1]])
        super().__init__(tokenizer, model_str, discriminators=discriminators, labels=labels)

    def reset(self):
        self.attribute1_model.reset()
        self.attribute2_model.reset()
        self.past = None
        self.accumulated_hidden = None
        self.grad_norms = None
        self.output_so_far = None
        self.device == 'cuda' and torch.cuda.empty_cache()

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name in ['output_so_far', 'accumulated_hidden'] and type(self) is PPLM_SFDecoder:
            self.attribute1_model.__setattr__(name, value)
            self.attribute2_model.__setattr__(name, value)

    def fuse_probs(self, unpert_probs, pert_probs1, pert_probs2):
        fused_probs = ((pert_probs1 ** self.gm_scale / 2) * (pert_probs2 ** self.gm_scale / 2) *
                      (unpert_probs ** (1 - self.gm_scale)))
        fused_probs = top_k_filter(fused_probs, k=self.top_k, probs=True)

        if torch.sum(fused_probs) <= 1:
            fused_probs = fused_probs / torch.sum(fused_probs)
        return fused_probs

    def decode_next_token(self, curr_idx):
        if self.output_so_far.shape[1] < 2:
            # no tokens yet, just generate one
            return self.unconditional_sample()

        if self.attribute1_model.past is None:
            # set self.past to be hidden state from model
            if self.output_so_far.shape[1] > 1:
                out = self.model(self.output_so_far[:, :-1])
                past = out.past_key_values
                self.attribute1_model.past = new_to_old(past)
                self.attribute2_model.past = new_to_old(past)

        last_output = self.output_so_far[:, -1:]

        out = self.model(self.output_so_far, output_hidden_states=True)
        unpert_logits = out.logits
        unpert_past = new_to_old(out.past_key_values)
        unpert_all_hidden = out.hidden_states
        # unpert_logits, unpert_past, unpert_all_hidden = self.model(self.output_so_far)
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
        unpert_last_hidden = unpert_all_hidden[-1]
        self.accumulated_hidden = torch.sum(
            unpert_last_hidden[:, :-1, :], dim=1
        )
        attribute1_pert_past = self.attribute1_model.perturb_past(last_output, unpert_past, unpert_logits)
        attribute1_pert_probs, attribute1_past = self.get_past_and_probs(attribute1_pert_past, last_output)

        attribute2_pert_past = self.attribute2_model.perturb_past(last_output, unpert_past, unpert_logits)
        attribute2_pert_probs, attribute2_past = self.get_past_and_probs(attribute2_pert_past, last_output)
        if type(attribute1_past[0]) is tuple:
            attribute1_past = new_to_old(attribute1_past)
            attribute2_past = new_to_old(attribute2_past)
        self.attribute1_model.past = attribute1_past
        self.attribute2_model.past = attribute2_past

        fused_probs = self.fuse_probs(unpert_probs, attribute1_pert_probs, attribute2_pert_probs)

        if self.sample:
            last = torch.multinomial(fused_probs, num_samples=1)
        else:  # greedy
            _, last = torch.topk(fused_probs, k=1, dim=-1)
        return last
