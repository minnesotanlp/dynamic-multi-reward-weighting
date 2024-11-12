import CtrlDecoder
import numpy as np
from operator import add
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from discrims import ClassifierHead, LABEL2IDX, DISCRIM_PATHS
from utils import new_to_old, old_to_new, SMALL_CONST, top_k_filter

def get_classifier(discrim: str, label: str, device: str):
    """Returns ClassifierHead for target discrim type and the label idx
    of target label."""
    if discrim not in ['sentiment', 'formality']:
        return NotImplementedError(f'No discriminator exists for given discrim type {discrim}')
    path = DISCRIM_PATHS[discrim]
    model = ClassifierHead(path, device=device)
    label2idx = LABEL2IDX['sentiment'] if discrim == 'sentiment' else LABEL2IDX['formality']
    label_idx = label2idx[label]
    return model, label_idx

class PPLMDecoder(CtrlDecoder.CtrlDecoder):
    def __init__(self, tokenizer=None, model=None, discriminators=None,
                 labels=None):
        assert len(discriminators) == len(labels), 'Discriminators and class labels must be the same length'
        self.tokenizer = tokenizer

        # Set up model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.config.use_cache = True
        self.model.config.output_hidden_states = True
        for param in self.model.parameters():  # freeze model weights
            param.requires_grad = False

        # Set up hyperparams
        # TODO: allow variation of the below hyperparams
        self.num_iterations = 5  # default is 3; paper says they like values 3-10
        self.gamma = 1
        self.horizon_length = 1
        self.stepsize = 0.05
        self.gm_scale = 0.8  # default is 0.9; paper says values 0.8-0.95 work well
        self.top_k = 10
        self.sample = True
        self.temperature = 1.0
        self.verbose = False
        self.kl_scale = 0.01  # default is 0.01; paper says 0.01 works best but some examples in
                              # colab notebook use 0.02

        # Set up instance variables for perturbing past
        self.past = None
        self.accumulated_hidden = None  # this is where the hidden state will live during iterations
        self.grad_norms = None
        self.output_so_far = None

        # Set up classifiers/labels
        self.discrims = []
        self.labels = []
        for discrim, label in zip(discriminators, labels):
            classifier, class_id = get_classifier(discrim, label, self.device)
            self.labels.append(class_id)
            self.discrims.append(classifier)

    def reset(self):
        """Does necessary clean up! (to be called after responding to a prompt)
        Reset instance variables and clears cuda cache if applicable."""
        self.past = None
        self.accumulated_hidden = None
        self.grad_norms = None
        self.output_so_far = None
        self.device == 'cuda' and torch.cuda.empty_cache()

    def generate_window_mask(self, window_length=0):
        """generates mask to look only at most recent window_length part of past.
        If window-length=0, we look at the whole past (i.e. the mask is all ones)"""
        _, _, _, curr_length, _ = self.past[0].shape
        if curr_length < window_length or window_length == 0:
            # No window needed, just return all ones
            return torch.ones_like(self.past[0]).to(self.device)
        else:
            # Create a window for the desired window_length
            ones_shape = (tuple(self.past[0].shape[:-2]) +
                          tuple([window_length]) +
                          tuple(self.past[0].shape[-1:]))
            zeros_shape = (tuple(self.past[0].shape[:-2]) +
                           tuple([curr_length - window_length]) +
                           tuple(self.past[0].shape[-1:]))
            return torch.cat((torch.ones(ones_shape), torch.zeros(zeros_shape)),
                             dim=-2).to(self.device)

    def set_up_tensor(self, t):
        """Prepares tensor for gradient computations during perturbation"""
        t.requires_grad_()  # by default, requires grad should be true, but we want to ensure it anyway
        return t.to(self.device)

    def perturb_past(self, last_token, unpert_past, unpert_logits):
        """Perturbs the past key values according to loss function"""
        grad_accumulator = [np.zeros(p.shape).astype('float32') for p in self.past]  # this will accum the grad across
                                                                                     # iterations
        assert self.accumulated_hidden is not None, 'Acc Hidd should never be none in perturb_past'
        window_mask = self.generate_window_mask()  # TODO: implement window? right now this is just all 1s

        for _ in range(self.num_iterations):  # Default num iter is 3
            # Add tensors that will store the past for the perturbed past
            # to pytorch's autograd graph (they will respond to the loss later)
            self.verbose and print('Iteration', i)
            curr_perturbation = [torch.tensor(p_, requires_grad=True, device=self.device) for p_ in grad_accumulator]
            perturbed_past = list(map(add, self.past, curr_perturbation))

            # Compute hidden state using perturbed past
            out = self.model(last_token, past_key_values=old_to_new(perturbed_past), output_hidden_states=True)
            all_logits = out.logits
            all_hidden = out.hidden_states
            last_hidden = all_hidden[-1]
            new_accum_hidden = self.accumulated_hidden + torch.sum(last_hidden, dim=1).detach()
            last_logits = all_logits[:, -1, :]  # logits shape is batch_size, sequence_length, vocab_size
            probs = F.softmax(last_logits, dim=-1)

            # Compute loss
            loss = 0.0
            loss_list = []
            ce_loss = torch.nn.CrossEntropyLoss()
            curr_unpert_past = unpert_past
            unsqueezed_pert_probs = torch.unsqueeze(probs, dim=1)
            wte = self.model.get_input_embeddings()

            for _ in range(self.horizon_length):  # default horizon length is 1
                input_embeddings = torch.matmul(unsqueezed_pert_probs, wte.weight.data)
                out = self.model(inputs_embeds=input_embeddings, past_key_values=old_to_new(curr_unpert_past), output_hidden_states=True)
                curr_unpert_past = out["past_key_values"]
                curr_all_hidden = out["hidden_states"]
                curr_last_hidden = curr_all_hidden[-1]
                new_accum_hidden += torch.sum(curr_last_hidden, dim=1)

            # calculate loss wrt classifier(s)
            _, _, _, curr_length, _ = self.past[0].shape
            discrim_loss = 0
            for discrim, label in zip(self.discrims, self.labels):
                prediction = discrim.pplm_forward(new_accum_hidden / (curr_length + 1 + self.horizon_length))
                label = torch.tensor([label], #todo adjust for different batch sizes ig
                                     device=self.device)
                l = ce_loss(prediction.unsqueeze(dim=0), label)
                discrim_loss += l
            loss += discrim_loss
            self.verbose and print('discrim loss', discrim_loss.data.cpu().numpy())
            loss_list.append(discrim_loss)

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            kl = self.calculate_kl_loss(unpert_probs, probs)
            self.verbose and print('KL loss', kl.data.cpu().numpy())
            loss += kl

            loss.backward()

            # Normalize gradients
            if self.grad_norms is not None:
                self.grad_norms = [
                    torch.max(self.grad_norms[index], torch.norm(p_.grad * window_mask))
                    for index, p_ in enumerate(curr_perturbation)
                ]
            else:
                self.grad_norms = [
                    (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                    for index, p_ in enumerate(curr_perturbation)
                ]
            grad = [
                -self.stepsize *
                (p_.grad * window_mask / self.grad_norms[
                    index] ** self.gamma).data.cpu().numpy()
                for index, p_ in enumerate(curr_perturbation)
            ]

            grad_accumulator = list(map(add, grad, grad_accumulator))

            # reset gradients; remove past from graph
            for p_ in curr_perturbation:
                p_.grad.data.zero_()
            new_past = [p_.detach() for p_ in self.past]
            self.past = new_past

        # Iterations complete -- add accumulated grads and return perturbed past
        grad_accumulator = [
            torch.tensor(p_, requires_grad=True, device=self.device) for p_ in grad_accumulator
        ]
        pert_past = list(map(add, self.past, grad_accumulator))
        return pert_past

    def calculate_kl_loss(self, unpert_probs, pert_probs):
        unpert_probs = (
                unpert_probs + SMALL_CONST *
                (unpert_probs <= SMALL_CONST).float().to(self.device).detach()
        )
        correction = SMALL_CONST * (pert_probs <= SMALL_CONST).float().to(
            self.device).detach()
        corrected_pert_probs = pert_probs + correction.detach()
        kl_loss = self.kl_scale * (
            (corrected_pert_probs * (corrected_pert_probs / unpert_probs).log()).sum()
        )
        return kl_loss

    def calculate_ppl(self, probs):
        next_token = torch.multinomial(probs, num_samples=1)
        inputs = torch.cat((self.output_so_far, next_token), dim=1)
        out = self.model(inputs, labels=inputs)
        return out.loss.item()

    def fuse_probs(self, unpert_probs, pert_probs):
        pert_probs = ((pert_probs ** self.gm_scale) * (
                unpert_probs ** (1 - self.gm_scale)))
        pert_probs = top_k_filter(pert_probs, k=self.top_k, probs=True)

        # make sure probs sum to 1
        if torch.sum(pert_probs) <= 1:
            pert_probs = pert_probs / torch.sum(pert_probs)
        return pert_probs

    def get_past_and_probs(self, pert_past, last_token):
        """Get past and probs for current output, except for the last token
        (GPT-2 takes past and current token as inputs)"""
        out = self.model(last_token, past_key_values=pert_past)
        pert_logits = out["logits"][:, -1, :] / self.temperature
        pert_probs = F.softmax(pert_logits, dim=-1)
        past = out["past_key_values"]
        return pert_probs, past

    def decode(self, prompt, txt_len=20):
        """Returns generation corresponding to a given prompt"""
        inputs = self.encode_prompt(prompt)
        print('inputs for decoding', inputs)
        context = inputs
        while len(context.shape) < 2:
            context = context.unsqueeze(0)

        self.output_so_far = context
        for i in range(txt_len):  # decode token by token
            if self.past is None:
                # set self.past to be hidden state from model
                if self.output_so_far.shape[1] > 1:
                    past = self.model(self.output_so_far[:, :-1])['past_key_values']
                    past = new_to_old(past)
                    self.past = past
            last = self.decode_next_token(i).to(self.output_so_far.device)
            if last[0].item() == self.tokenizer.eos_token_id:
                continue
            #other_last = self.decode_next_tokentmp(i)  # this was just to see the dif
            self.output_so_far = torch.cat((self.output_so_far, last), dim=1)
            self.verbose and print(self.tokenizer.decode(self.output_so_far.tolist()[0]))

        result = self.tokenizer.decode(self.output_so_far.tolist()[0])
        self.verbose and print(result)
        self.reset()  # need to reset state after completing the generation
        return result

    def decode_next_token(self, curr_idx):
        if self.output_so_far.shape[1] < 2:
            # no tokens yet, just generate one
            return self.unconditional_sample()

        last_output = self.output_so_far[:, -1:]

        # unpert_logits, unpert_past, unpert_all_hidden = self.model(self.output_so_far)
        out = self.model(self.output_so_far, output_hidden_states=True)
        unpert_logits = out.logits
        unpert_all_hidden = out.hidden_states
        unpert_past = out.past_key_values
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
        unpert_last_hidden = unpert_all_hidden[-1]
        self.accumulated_hidden = torch.sum(
            unpert_last_hidden[:, :-1, :], dim=1
        )

        pert_past = self.perturb_past(last_output, unpert_past, unpert_logits)
        pert_probs, past = self.get_past_and_probs(pert_past, last_output)
        total_l1_distance = 0
        for i in range(len(pert_past)):
            l1_distance = torch.norm(pert_past[i] - self.past[i], p=1)
            total_l1_distance += l1_distance

        self.past = new_to_old(past)
        fused_probs = self.fuse_probs(unpert_probs, pert_probs)
        if self.sample:
            last = torch.multinomial(fused_probs, num_samples=1)
        else:  # greedy
            _, last = torch.topk(fused_probs, k=1, dim=-1)
        return last

    def encode_prompt(self, prompt):
        return self.tokenizer.encode(
            self.tokenizer.bos_token + prompt,
            add_special_tokens=False, return_tensors='pt'
        ).to(self.device)

    def unconditional_sample(self):
        no_context = torch.tensor(self.encode_prompt(''), device=self.device)
        logits = self.model(no_context).logits
        # logits, _, _ = self.model(no_context)
        probs = F.softmax(logits)
        probs = top_k_filter(probs, k=10, probs=True)
        if self.sample:
            last = torch.multinomial(probs, num_samples=1)
        else:  # greedy
            _, last = torch.topk(probs, k=1, dim=-1)
        return last



if __name__ == '__main__':
    pretrained_model = 'meta-llama/Llama-2-7b-hf'
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LlamaForCausalLM.from_pretrained(pretrained_model).to('cuda')
    model.config.use_cache = True
    model.config.coutput_hidden_states = True
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token_id = 2
    prompt = 'This movie is'
    inputs = tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False, return_tensors='pt')
    print("inputs", inputs)
    print('Original generation: ')
    context = torch.tensor(inputs, device='cuda', dtype=torch.long)
    while len(context.shape) < 2:
        context = context.unsqueeze(0)
    for i in range(10):
        out = model(context, output_hidden_states=True)
        logits = out['logits']
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, torch.multinomial(probs, num_samples=1)), dim=1)
    print(tokenizer.decode(context[0]))
    print('\nThe controlled generation:')
    torch.manual_seed(seed)
    np.random.seed(seed)
    decoder = PPLMDecoder(tokenizer, model, discriminators=['sentiment', 'formality'], labels=['positive', 'informal'])
    res = decoder.decode(prompt)
    print(res)
