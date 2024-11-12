import torch

class ClassifierHead:
    def __init__(self, classifier_path, num_labels=2, device='cuda'):
        self.score = torch.nn.Linear(4096, num_labels, bias=False)
        self.score.load_state_dict(torch.load(classifier_path, map_location=torch.device('cuda:0')), strict=False)
        self.score.to(device)

    def forward_hidden(self, hidden, input_ids, pad_token_id):
        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.ne(input_ids, pad_token_id).sum(-1) - 1).to(hidden.device)
        self.score = self.score.to(hidden.device)
        logits = self.score(hidden)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits

    def pplm_forward(self, hidden):
        batch_size = 1
        sequence_lengths = torch.arange(2)
        self.score = self.score.to(hidden.device)
        logits = self.score(hidden)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits
