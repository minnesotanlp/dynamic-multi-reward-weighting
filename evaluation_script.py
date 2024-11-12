import argparse
import json, torch, os, sys, time
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import pipeline, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast, GPT2Config
from transformers.file_utils import cached_path
import torch
from torch.nn import CrossEntropyLoss
from nltk.tokenize.treebank import TreebankWordDetokenizer

sys.path.insert(1, './PPLM')
from run_pplm_discrim_train import Discriminator, evaluate_performance, predict, get_idx2class
from run_pplm import DISCRIMINATOR_MODELS_PARAMS, get_classifier, generate_text_pplm
from pplm_classification_head import ClassificationHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length_seq = 100
       
def main(eval_file, ctl_attr, mode, save):
    print("Preprocessing {} dataset...".format(str(eval_file)))
    start = time.time()

    assert type(mode) == str
    pred_labels = None
    if os.path.exists(eval_file) != True:
        print("please make sure the decoded file exists")

    sentences = []
    with open(eval_file, 'r') as f:
        data = json.load(f)
    num_idx = len(data)
    
    gen_dict = {}
    class_labels = []
    # compositional control
    if len(ctl_attr.split(',')) > 1:
        # keys: 'very_positive, informal', 'very_positive, formal', 'very_negative, informal', 'very_negative, formal'
        ctl_attr = 'sentiment, formality'
    for key, value in data['0'][ctl_attr].items():
        # strip "[, ]"
        key = key.strip("['")
        key = key.strip("']")
        gen_dict[key] = []
        class_labels.append(key)
    
    for i in range(num_idx):
        for key, value in data[str(i)][ctl_attr].items():
            # strip "[, ]"
            key = key.strip("['")
            key = key.strip("']")
            gen_dict[key].extend(value)

    for class_label in class_labels:
        print('class_label: '+str(class_label))
        sentences, results, pred_labels = eval(gen_dict[class_label], ctl_attr, class_label, mode)
        if save == True:
            # save eval results into file
            path = './eval_results'
            isExist = os.path.exists(path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path)
                print("The eval_results directory is created!")
            if mode in ['acc', 'precision', 'f1', 'recall']:
                #assert len(results) == 3
                with open(path+'/eval_'+mode+'_'+ctl_attr+'_'+class_label+'.json', 'w') as f:
                    result_dict = {}
                    result_dict['eval_file'] = str(eval_file)
                    result_dict['ctl_attr'] = str(ctl_attr)
                    result_dict['target_label'] = str(class_label)
                    result_dict['acc'] = results[0]
                    result_dict['f1'] = results[1]
                    result_dict['recall'] = results[2]
                    json.dump(result_dict, f, indent = 4)
            elif mode == 'ppl':
                assert len(sentences) == len(results)
                with open(path+'/eval_'+mode+'_'+ctl_attr+'_'+class_label+'.json', 'w') as f:
                    for i in range(len(sentences)):
                        # save to json line by line
                        result_dict = {}
                        result_dict['idx'] = str(i)
                        result_dict['ctl_attr'] = str(ctl_attr)
                        result_dict['target_label'] = str(class_label)
                        result_dict['sentence'] = sentences[i]
                        result_dict[mode] = results[i]
                        json.dump(result_dict, f, indent = 4)
            else:
                assert len(sentences) == len(results)
                with open(path+'/eval_'+mode+'_'+ctl_attr+'_'+class_label+'.json', 'w') as f:
                    for i in range(len(sentences)):
                        # save to json line by line
                        result_dict = {}
                        result_dict['idx'] = str(i)
                        result_dict['ctl_attr'] = str(ctl_attr)
                        result_dict['target_label'] = str(class_label)
                        result_dict['pred_label'] = pred_labels[i]
                        result_dict['sentence'] = sentences[i]
                        result_dict[mode] = results[i]
                        json.dump(result_dict, f, indent = 4)
    end = time.time()
    print("Evaluation took: {:.3f}s".format(end - start))
    print("Evaluation Done!")
    return results

def eval(sentences, ctl_attr, class_label, mode):
    # eval results
    results = []
    if mode == 'ppl':
        pred_labels = None
        model_id = "gpt2-large"
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        model.to(device).eval()
        for i in range(len(sentences)):
            ppl = get_ppl(sentences[i], model, tokenizer)
            results.append(ppl)
            print(sentences[i]+"...perplexity: "+str(ppl))

    elif mode == 'pred_scores':
        pred_labels, pred_scores, ground_truths = get_pred_results(sentences, ctl_attr, class_label)
        results = pred_scores
    elif mode in ['acc', 'precision', 'f1', 'recall']:
        pred_labels, pred_scores, ground_truths = get_pred_results(sentences, ctl_attr, class_label)
        results.append(accuracy_score(ground_truths, pred_labels))
        results.append(f1_score(ground_truths, pred_labels, average='macro'))
        results.append(recall_score(ground_truths, pred_labels, average="macro"))
    else:
        print("please make sure to input correct eval modes")
    return sentences, results, pred_labels

# ppl
def get_ppl(sentence, model, tokenizer):
      input_ids = torch.tensor(tokenizer.encode(sentence.strip('<|endoftext|>'))).unsqueeze(0) 
      input_ids = input_ids.to(device)
      with torch.no_grad():
          outputs = model(input_ids, labels=input_ids)
      loss, logits = outputs[:2]
      return math.exp(loss)

# classifier-based
def get_pred_results(sentences, ctl_attr, class_label):                
    pred_labels = []
    pred_scores = []
    ground_truths = []
    if ctl_attr.lower() == 'hatespeech':
        hatespeech_analysis = pipeline(model="Hate-speech-CNERG/bert-base-uncased-hatexplain")
        ground_truths = ['hatespeech'] * len(sentences)
        for i in range(len(sentences)):
            result = hatespeech_analysis(sentences[i])[0]
            pred_label = result['label']
            pred_labels.append(pred_label)
            if pred_label == 'hatespeech':
                pred_scores.append(result['score'])
            else:
                pred_scores.append(1 - result['score'])
    elif len(ctl_attr.split(',')) > 1:
        sentiment_pred_labels = []
        sentiment_pred_scores = []
        formality_pred_labels = []
        formality_pred_scores = []
        # 'very_positive, informal', 'very_positive, formal', 'very_negative, informal', 'very_negative, formal'
        if class_label == 'very_positive, informal':
            ground_truths = ['positive, informal'] * len(sentences)
            sentiment_class_label = 'very_positive'
            formality_class_label = 'informal'
        elif class_label == 'very_positive, formal':
            ground_truths = ['positive, formal'] * len(sentences)
            sentiment_class_label = 'very_positive'
            formality_class_label = 'formal'
        elif class_label == 'very_negative, informal':
            ground_truths = ['negative, informal'] * len(sentences)
            sentiment_class_label = 'very_negative'
            formality_class_label = 'informal'
        elif class_label == 'very_negative, formal':
            ground_truths = ['negative, formal'] * len(sentences)
            sentiment_class_label = 'very_negative'
            formality_class_label = 'formal'
        else:
            print("please make sure the labels for compositional controllable generations are correct")
        # sentiment
        idx2class = ["positive", "negative", "very_positive", "very_negative",
                     "neutral"]
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS['sentiment'][
                    "pretrained_model"
                ]
        discrim = class_label
        pretrained_model = discriminator_pretrained_model
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True
        )
        model.to(device)
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        for param in model.parameters():
            param.requires_grad = False
        classifier, class_id = get_classifier(
            'sentiment',
            sentiment_class_label,
            device
        )
        if device == 'cuda':
            torch.cuda.empty_cache()
        for i in range(len(sentences)):
            raw_text = sentences[i].strip('<|endoftext|>')
            tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + raw_text,
                add_special_tokens=False
            )
            output_so_far = None
            context = tokenized_cond_text
            context_t = torch.tensor(context, device=device, dtype=torch.long)
            while len(context_t.shape) < 2:
                context_t = context_t.unsqueeze(0)
            output_so_far = context_t
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]
            logits = classifier(torch.mean(unpert_last_hidden, dim=1))
            predictions = torch.nn.functional.softmax(logits, dim=1)
            preds = predictions.data.cpu().numpy().flatten().tolist()
            if sentiment_class_label in ['positive', 'very_positive']:
                pred_score = preds[idx2class.index('positive')] + preds[idx2class.index('very_positive')]
            elif sentiment_class_label in ['negative', 'very_negative']:
                pred_score = preds[idx2class.index('negative')] + preds[idx2class.index('very_negative')]
            else:
                pred_score = preds[idx2class.index('neutral')]
            sentiment_pred_scores.append(pred_score)

            pred_label = idx2class[preds.index(max(preds))]
            if pred_label == 'very_positive':
                sentiment_pred_labels.append('positive')
            elif pred_label == 'very_negative':
                sentiment_pred_labels.append('negative')
            else:
                sentiment_pred_labels.append(pred_label)
        
        # formality
        idx2class = ["formal", "informal"]
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS['formality'][
                    "pretrained_model"
                ]
        discrim = class_label
        pretrained_model = discriminator_pretrained_model
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True
        )
        model.to(device)
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        for param in model.parameters():
            param.requires_grad = False
        classifier, class_id = get_classifier(
            'formality',
            formality_class_label,
            device
        )
        if device == 'cuda':
            torch.cuda.empty_cache()
        for i in range(len(sentences)):
            raw_text = sentences[i].strip('<|endoftext|>')
            tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + raw_text,
                add_special_tokens=False
            )
            output_so_far = None
            context = tokenized_cond_text
            context_t = torch.tensor(context, device=device, dtype=torch.long)
            while len(context_t.shape) < 2:
                context_t = context_t.unsqueeze(0)
            output_so_far = context_t
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]
            logits = classifier(torch.mean(unpert_last_hidden, dim=1))
            predictions = torch.nn.functional.softmax(logits, dim=1)
            preds = predictions.data.cpu().numpy().flatten().tolist()
            formality_pred_scores.append(preds[idx2class.index(formality_class_label)])
            pred_label = idx2class[preds.index(max(preds))]
            formality_pred_labels.append(pred_label)
        
        # compositional results
        for i in range(len(sentences)):
            print("Ground truth: "+ground_truths[i])
            print("Prediction labels: "+str(sentiment_pred_labels[i])+', '+str(formality_pred_labels[i]))
            pred_scores.append(str(sentiment_pred_scores[i])+str(formality_pred_scores[i]))
            pred_labels.append(str(sentiment_pred_labels[i])+', '+str(formality_pred_labels[i]))
    
    elif ctl_attr.lower() == 'sentiment':
        idx2class = ["positive", "negative", "very_positive", "very_negative",
                     "neutral"]
        # label scaling
        if class_label == 'very_positive':
            ground_truths = ['positive'] * len(sentences)
        elif class_label == 'very_negative':
            ground_truths = ['negative'] * len(sentences)
        else:
            ground_truths = [class_label] * len(sentences)
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS['sentiment'][
                    "pretrained_model"
                ]
        discrim = class_label
        pretrained_model = discriminator_pretrained_model

        # load pretrained model
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True
        )
        model.to(device)
        model.eval()

        # load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

        # Freeze GPT-2 weights
        for param in model.parameters():
            param.requires_grad = False

        classifier, class_id = get_classifier(
            'sentiment',
            class_label,
            device
        )
        if device == 'cuda':
            torch.cuda.empty_cache()
        for i in range(len(sentences)):
            # encode
            raw_text = sentences[i].strip('<|endoftext|>')
            tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + raw_text,
                add_special_tokens=False
            )
            output_so_far = None
            context = tokenized_cond_text
            context_t = torch.tensor(context, device=device, dtype=torch.long)
            while len(context_t.shape) < 2:
                context_t = context_t.unsqueeze(0)
            output_so_far = context_t
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]
            logits = classifier(torch.mean(unpert_last_hidden, dim=1))
            predictions = torch.nn.functional.softmax(logits, dim=1)
            preds = predictions.data.cpu().numpy().flatten().tolist()
            print("Predictions:", ", ".join(
                    "{}: {:.4f}".format(c, pred) for c, pred in
                    zip(idx2class, preds)
                ))
            # label scaling
            if class_label in ['positive', 'very_positive']:
                pred_score = preds[idx2class.index('positive')] + preds[idx2class.index('very_positive')]
            elif class_label in ['negative', 'very_negative']:
                pred_score = preds[idx2class.index('negative')] + preds[idx2class.index('very_negative')]
            else:
                pred_score = preds[idx2class.index('neutral')]
            pred_scores.append(pred_score)

            pred_label = idx2class[preds.index(max(preds))]
            # label scaling
            if pred_label == 'very_positive':
                pred_labels.append('positive')
            elif pred_label == 'very_negative':
                pred_labels.append('negative')
            else:
                pred_labels.append(pred_label)

    elif ctl_attr.lower() == 'formality':
        idx2class = ["formal", "informal"]
        ground_truths = [class_label] * len(sentences)
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS['formality'][
                    "pretrained_model"
                ]
        discrim = class_label
        pretrained_model = discriminator_pretrained_model

        # load pretrained model
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True
        )
        model.to(device)
        model.eval()

        # load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

        # Freeze GPT-2 weights
        for param in model.parameters():
            param.requires_grad = False

        classifier, class_id = get_classifier(
            'formality',
            class_label,
            device
        )
        if device == 'cuda':
            torch.cuda.empty_cache()
        for i in range(len(sentences)):
            # encode
            raw_text = sentences[i].strip('<|endoftext|>')
            tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + raw_text,
                add_special_tokens=False
            )
            output_so_far = None
            context = tokenized_cond_text
            context_t = torch.tensor(context, device=device, dtype=torch.long)
            while len(context_t.shape) < 2:
                context_t = context_t.unsqueeze(0)
            output_so_far = context_t
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]
            logits = classifier(torch.mean(unpert_last_hidden, dim=1))
            predictions = torch.nn.functional.softmax(logits, dim=1)
            preds = predictions.data.cpu().numpy().flatten().tolist()
            print("Predictions:", ", ".join(
                    "{}: {:.4f}".format(c, pred) for c, pred in
                    zip(idx2class, preds)
                ))
            pred_scores.append(preds[idx2class.index(class_label)])
            pred_label = idx2class[preds.index(max(preds))]
            pred_labels.append(pred_label)
    else:
        print("please make sure to input correct control attribute")
    return pred_labels, pred_scores, ground_truths

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='eval args.')
    parser.add_argument('--eval_file', type=str, required=True, help='path to the file of decoded texts')
    parser.add_argument('--ctl_attr', type=str, required=True, help='controlled attribute during generating process')
    #parser.add_argument('--class_label', type=int, required=True, help='class label for the controlled attribute')
    parser.add_argument('--mode', type=str, required=True, help='evaluation metrics, ex. ppl -- perplexity, \
                                                        pred_scores -- prediction scores for the tartget attribute from pre-trained classifier,\
                                                        acc, precision, recall, f1.')
    parser.add_argument('--save', type=bool, default=True, help='save the eval results')

    args = parser.parse_args()
    main(args.eval_file, args.ctl_attr, args.mode, args.save)
