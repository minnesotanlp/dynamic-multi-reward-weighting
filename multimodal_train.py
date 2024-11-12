from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from math import sqrt
from accelerate import Accelerator
from datasets import load_dataset
import datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, LlamaTokenizer, LlamaForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline, T5ForConditionalGeneration, T5Config
from transformers.utils.hub import cached_file
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import wandb
import os
import sys
from discrims import ClassifierHead, DISCRIM_PATHS, LABEL2IDX
from utils import hf_login
import pandas as pd
tqdm.pandas()

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    p_adjust: Optional[bool] = field(default=True, metadata={"help": "adjust p"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    sentiment: Optional[str] = field(default="", metadata={"help": "the target sentiment: one of (pos, neg)"})
    formality: Optional[str] = field(default="", metadata={"help": "the target formality: one of (formal, informal)"})
    dynamic_reward: Optional[bool] = field(default=False, metadata={"help": "Experimental dynamic reward"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    grad_wt: Optional[bool] = field(default=False, metadata={"help": "whether to use grad weighting"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "num gradient accumulation steps"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=9, metadata={"help": "kl target for early stopping"})
    kl_coef: Optional[float] = field(default=0.2, metadata={"help": "initial kl coef"})
    softmax: Optional[bool] = field(default=True, metadata={"help": "whether to use softmax on the logits from discrims"})
    binary: Optional[bool] = field(default=False, metadata={"help": "whether to use binary reward formulation"})
    third_att: Optional[str] = field(default="None", metadata={"help":"third attribute (supported: irony|neutrality|toxicity)"})
    third_att_label: Optional[int] = field(default=1, metadata={"help":"third attribute target label idx (1 for toxicity/irony)"})
    score_scaling: Optional[bool] = field(default=True, metadata={"help": "to scale or not scale rewards"})
    score_norming: Optional[bool] = field(default=True, metadata={"help": "to norm or not norm rewards"})
    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=5, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="checkpoints/", metadata={"help": "output directory for storing checkpoints"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})

parser = HfArgumentParser(ScriptArguments)
args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

train_df = pd.read_csv('train_texts.csv')
train_dataset = datasets.Dataset.from_pandas(train_df)

tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

def get_dataset(tokenizer):
    # TODO: do we want to use different datasets for prompts this time around?
    rotten_dataset = load_dataset("rotten_tomatoes").remove_columns(['label'])
    yelp_dataset = load_dataset("yelp_review_full").remove_columns(['label'])
    train_dataset = datasets.concatenate_datasets([rotten_dataset['train'], yelp_dataset['train']], axis=0)
    test_dataset = datasets.concatenate_datasets([rotten_dataset['test'], yelp_dataset['test']], axis=0)
    return build_dataset(tokenizer, test_dataset)

def build_dataset(
    tokenizer, dataset, input_min_text_length=5, input_max_text_length=9
):
    ds = dataset
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    def tokenize(sample):
        words = sample["text"].split()
        next_idx = 1
        while next_idx < len(words) and words[next_idx] == '@user':
            next_idx += 1
        text = [words[i] for i in [0] + list(range(next_idx, len(words)))]
        sample["text"] = ' '.join(text)
        sample["input_ids"] = tokenizer.encode(sample["text"])[: input_size()]
        sample["prompt"] = tokenizer.decode(sample["input_ids"])
        return sample
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(tokenizer, train_dataset)

if __name__ == '__main__':
    hf_login()
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.binary and not args.softmax:
        raise NotImplementedError("We do not compute a binary reward without softmax")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        init_kl_coef = args.kl_coef,
        target_kl = 10,
        #target_kl=args.target_kl,
        score_scaling = args.score_scaling,
        use_score_norm = args.score_norming,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        peft_config=lora_config,
        device_map='cuda'
    )

    set_seed(config.seed)
    optimizer = None
    if args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

def build_pipeline(tokenizer, 
                   style='sentiment',  # one of sentiment|formality|tpxicity|irony|neutral
                   get_grads=False,  # whether to use gradient weighting technique
                   use_softmax=False,  # whether to use softmax (vs raw logits)
                   label=0):
    """Returns a pipeline function that takes text as input, and outputs classification score from target style discriminator."""
    
    #set up classifier
    style = style.lower()
    if style not in DISCRIM_PATHS.keys():
        raise ValueError(f'Input for target styles must be one of {list(DISCRIM_PATHS.keys())}')
    model_path = DISCRIM_PATHS[style]

    if 'neutral' == style:
        model_path = DISCRIM_PATHS['emotion']
        num_logits = 7
    else:
        num_logits = 2

    classifier = ClassifierHead(model_path, num_labels=num_logits)

    def pipeline(input_sentence):
        token_ids = tokenizer.batch_encode_plus(input_sentence, return_tensors='pt', truncation=True, padding=True)
        token_ids = token_ids.to('cuda')

        if get_grads: # compute gradient norm for gradient weighting
            loss_fct = CrossEntropyLoss()
            with model.pretrained_model.disable_adapter():
                out = model.pretrained_model.base_model.base_model(**token_ids).last_hidden_state
                output = classifier.forward_hidden(out, token_ids['input_ids'], tokenizer.pad_token_id)
            logits = output.view(-1, num_logits)
            labels = torch.tensor(label).to('cuda')
            score_gradients = [torch.autograd.grad(loss_fct(outpt, labels), classifier.score.weight, retain_graph=True) for outpt in logits]
            grad_norms = [torch.norm(s[0]) for s in score_gradients]
            torch.autograd.grad(loss_fct(logits[0], labels), classifier.score.weight)  # just to free tensors :/ ideally we want retain_graph=False for last elt of above
            if use_softmax:
                probs = [torch.nn.functional.softmax(o) for o in output]
            return probs, grad_norms
        else: 
            with torch.no_grad():
                with model.pretrained_model.disable_adapter():  # this way we can do inference on the original Llama model, which the discrim was trained on
                    out = model.pretrained_model.base_model.base_model(**token_ids).last_hidden_state
                    output = classifier.forward_hidden(out, token_ids['input_ids'], tokenizer.pad_token_id)
            if use_softmax:
                probs = [torch.nn.functional.softmax(o) for o in output]
            else:
                probs = output
            return probs, output
    return pipeline

if args.sentiment:
    sent_label = LABEL2IDX['sentiment'][args.sentiment]
    sentiment_pipe = build_pipeline(tokenizer, style='sentiment', get_grads=args.grad_wt, calibrate=args.calibrate, use_softmax=args.softmax, label=sent_label)
if args.formality:
    form_label = LABEL2IDX['formality'][args.formality]
    formality_pipe = build_pipeline(tokenizer, style='formality', get_grads=args.grad_wt, calibrate=args.calibrate, use_softmax=args.softmax, label=form_label)
if args.third_att:
    target_label = LABEL2IDX[args.third_att][args.third_att_label]
    other_pipe = build_pipeline(tokenizer, style=args.third_att, get_grads=args.grad_wt, calibrate=args.calibrate, use_softmax=args.softmax, label=target_label)
    # other_pipe is generic third attribute pipe for either irony, toxicity, or neutrality

generation_kwargs = {
    "min_new_tokens": 15,
    "top_k": 0.0,
    "repetition_penalty": 1.2,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

output_min_length = 20
output_max_length = 50
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]
    appended_question_tensors = []
    if args.dynamic_reward:
        for i, qt in enumerate(question_tensors):
             sig_tok = signal_tokens[i % len(signal_tokens)].view(-1)
             appended_qt = torch.cat((qt[0].view(1), sig_tok.view(1), qt[1:]), dim=0)
             appended_question_tensors.append(appended_qt)
        response_tensors = [ppo_trainer.generate(q, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs).squeeze() for q in question_tensors]
    else:
        response_tensors = [ppo_trainer.generate(q, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs).squeeze() for q in question_tensors]
    response_tensors = [r if r.nelement() > 0 and r.dim() > 0 else r.view(-1, 1) for r in response_tensors]
    if len(response_tensors) != len(question_tensors):
        print(f'only{len(response_tensors)} responses! skipping')
        continue
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    batch["query"] = tokenizer.batch_decode(question_tensors, skip_special_tokens=True)
    
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    batch["target_form"] = [0 for _ in texts]
    batch["target_sent"] = [0 for _ in texts]
    batch["target_toxicity_grad"] = [0 for _ in texts]
    batch["target_toxicity"] = [0 for _ in texts]
    if args.formality:
        formality_idx = 0 if args.formality == 'informal' else 1
        formality_outputs, wts = formality_pipe(texts)
        formality_idx = 0 if args.formality == 'informal' else 1
        formality_outputs = [output[formality_idx] for output in formality_outputs]
        if args.binary:
            formality_outputs = [1 if output >= 0.5 else -1 for output in formality_outputs]
            if args.grad_wt:
                formality_weights = [wt.item() for wt in wts]
        batch["target_form"] = [output for output in formality_outputs]
        batch["target_form_grad"] = wts
        
        if args.grad_wt and not args.binary:
            formality_weights = [wt.item() for wt in wts]
            formality_outputs = [output for output in formality_outputs]
        formality_outputs = [max(-9, min(9, o)) for o in formality_outputs]
        
        formality_rewards = [torch.tensor(output, dtype=torch.float) for output in formality_outputs]
        rewards = formality_rewards
        wandb.log({'env/form_reward_mean': torch.mean(torch.tensor(rewards)).cpu().numpy().item()})

    if args.third_att in ['toxicity', 'neutral', 'irony']:
        # TODO: variable named toxicity but stands for any third attribute; rename
        toxicity_idx = 1
        toxicity_outputs, wts = toxicity_pipe(texts)
        toxicity_outputs = [output[toxicity_idx] for output in toxicity_outputs]
        batch["target_toxicity"] = toxicity_outputs
        wandb.log({'env/tox_reward_mean': torch.mean(torch.tensor(toxicity_outputs, dtype=torch.float)).cpu().numpy().item()})
        batch["toxicity_grad"] = wts
        if args.grad_wt:
            toxicity_weights = [wt.item() for wt in wts]

    if args.sentiment:
        sentiment_idx = 0 if 'neg' in args.sentiment else 1
        sentiment_outputs, logits = sentiment_pipe(texts)
        sentiment_outputs = [output[sentiment_idx] for output in sentiment_outputs]
        batch["target_sent"] = [output for output in sentiment_outputs]
        batch["target_sent_grad"] = logits
        
        if args.binary:
            sentiment_outputs = [1 if output >= 0.5 else -1 for output in sentiment_outputs]
            if args.grad_wt:
                sentiment_weights = [wt.item() for wt in logits]

        if args.grad_wt and not args.binary:
            sentiment_outputs = [output for output in sentiment_outputs]
            sentiment_weights = [wt.item() for wt in logits]

        sentiment_outputs = [min(9, max(o, -9)) for o in sentiment_outputs] # clip outputs
        sentiment_rewards = [torch.tensor(output, dtype=torch.float) for output in sentiment_outputs]
        rewards = sentiment_rewards
        wandb.log({'env/sent_reward_mean': torch.mean(torch.tensor(rewards)).cpu().numpy().item()})
        if args.formality and args.toxicity:  # combine rewards
            rewards = [0.5 * formality + 0.5 * sentiment for formality, sentiment in zip(formality_rewards, sentiment_rewards)]
            if args.grad_wt:
                norm_sent_wt = [float(sent_wt)/(float(form_wt) + float(sent_wt) + float(tox_wt)) for form_wt, sent_wt, tox_wt in zip(formality_weights, sentiment_weights, toxicity_weights)]
                norm_form_wt = [float(form_wt)/(float(form_wt) + float(sent_wt) + float(tox_wt)) for form_wt, sent_wt, tox_wt in zip(formality_weights, sentiment_weights, toxicity_weights)]
                norm_tox_wt = [float(tox_wt)/(float(form_wt) + float(sent_wt) + float(tox_wt)) for form_wt, sent_wt, tox_wt in zip(formality_weights, sentiment_weights, toxicity_weights)]
                max_tox_wt = max(toxicity_weights)
                max_form_wt = max(formality_weights)
                max_sent_wt = max(sentiment_weights)
                norm_form_wt = [float(form_wt)/float(max_form_wt) for form_wt in formality_weights]
                norm_sent_wt = [float(sent_wt)/float(max_sent_wt) for sent_wt in sentiment_weights]
                norm_tox_wt = [float(tox_wt)/float(max_tox_wt) for tox_wt in toxicity_weights]
                batch["target_sent"] = sentiment_outputs
                batch["target_form"] = formality_outputs
                sentiment_outputs = [max(-5, (1 - o) * -wt * 5) if o < 0.5 else min(5, o * wt * 5) for o, wt in zip(sentiment_outputs, norm_sent_wt)]
                formality_outputs = [min(5, o * wt * 5) if o >= 0.5 else max(-5, (1 - o) * -wt * 5) for o, wt in zip(formality_outputs, norm_form_wt)]
                toxicity_outputs = [min(5, o * wt * 5) if o >= 0.5 else max(-5, (1 - o) * -wt * 5) for o, wt in zip(toxicity_outputs, norm_tox_wt)]
                batch["target_sent_grad"] = norm_sent_wt
                batch["target_form_grad"] = norm_form_wt
                batch["target_toxicity_grad"] = norm_tox_wt
                sentiment_rewards = [torch.tensor(output, dtype=torch.float) for output in sentiment_outputs]
                formality_rewards = [torch.tensor(output, dtype=torch.float) for output in formality_outputs]
                toxicity_rewards = [torch.tensor(output, dtype=torch.float) for output in toxicity_outputs]
                rewards = [(1/3) * formality + (1/3) * sentiment + (1/3) * toxicity for formality, sentiment, toxicity in zip(formality_rewards, sentiment_rewards, toxicity_outputs)]
    rewards = [torch.tensor(0, dtype=torch.float) if (resp_tensor.nelement() < 12 and reward > 0) else torch.tensor(reward, dtype=torch.float) for reward, resp_tensor in zip(rewards, response_tensors)] 
   
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "target_sent", "target_form", "target_sent_grad", "target_form_grad", "target_toxicity_grad", "target_toxicity"])

    if args.save_freq and epoch and epoch % args.save_freq == 0:
        ppo_trainer.save_pretrained(args.output_dir + '/' + f"step_{epoch}")

ppo_trainer.save_pretrained(args.output_dir + '/' + 'final')
