import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import numpy as np

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from nltk.translate.chrf_score import corpus_chrf
import joblib
import warnings

warnings.filterwarnings('ignore')
import nltk
from nltk.translate import meteor
from nltk.translate import bleu_score
from nltk import word_tokenize


def load_model(model_name=None, model=None, tokenizer=None,
               model_class=AutoModelForSequenceClassification, use_cuda=True):
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')
        model = model_class.from_pretrained(model_name)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def count_STA(generated_sents, sta_model):
    probs_max = sta_model.predict_proba(generated_sents).max(axis=1)
    classes = sta_model.predict(generated_sents)  # 1 - officialese, 0 -neutral
    probs = np.zeros(len(generated_sents))
    for i, p in enumerate(probs_max):
        if classes[i] == 1:
            probs[i] = p
        else:
            probs[i] = 1 - p
    return probs, classes


def encode_cls(texts, model, tokenizer, batch_size=32, verbose=False):
    results = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        with torch.no_grad():
            out = model(**tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device))
            embeddings = out.pooler_output
            embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
            results.append(embeddings)
    return np.concatenate(results)


def count_CP(
        meaning_model,
        meaning_tokenizer,
        original_texts,
        rewritten_texts,
        batch_size=32,
        verbose=False,
):
    scores = (
            encode_cls(original_texts, model=meaning_model, tokenizer=meaning_tokenizer, batch_size=batch_size,
                       verbose=verbose)
            * encode_cls(rewritten_texts, model=meaning_model, tokenizer=meaning_tokenizer, batch_size=batch_size,
                         verbose=verbose)
    ).sum(1)
    return scores


def count_METEOR(original_texts, generated_texts):
    ans = np.zeros(len(original_texts))
    for i, orig in enumerate(original_texts):
        ans[i] = meteor([word_tokenize(orig)], word_tokenize(generated_texts[i]))
    return ans


def count_BLEU(original_texts, generated_texts):
    ans = np.zeros(len(original_texts))
    for i, orig in enumerate(original_texts):
        ans[i] = bleu_score.sentence_bleu([orig], generated_texts[i])
    return ans


mname = 'sberbank-ai/rugpt3large_based_on_gpt2'
fl_tokenizer = AutoTokenizer.from_pretrained(mname)
fl_model = AutoModelForCausalLM.from_pretrained(mname)
if torch.cuda.is_available():
    fl_model.cuda()


def calc_gpt2_ppl_corpus(test_sentences, aggregate=False, sep='\n'):
    """ Calculate average perplexity per token and number of tokens in each text."""
    lls = []
    weights = []
    #     for text in tqdm(test_sentences):
    for text in test_sentences:
        encodings = fl_tokenizer(f'{sep}{text}{sep}', return_tensors='pt')
        input_ids = encodings.input_ids.to(fl_model.device)
        target_ids = input_ids.clone()

        w = max(0, len(input_ids[0]) - 1)
        if w > 0:
            with torch.no_grad():
                outputs = fl_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0]
                ll = log_likelihood.item()
        else:
            ll = 0
        lls.append(ll)
        weights.append(w)
    likelihoods, weights = np.array(lls), np.array(weights)
    if aggregate:
        return sum(likelihoods * weights) / sum(weights)
    return likelihoods, weights


def count_FL(original_texts,
             rewritten_texts):
    p1, w1 = calc_gpt2_ppl_corpus(original_texts)
    p2, w2 = calc_gpt2_ppl_corpus(rewritten_texts)
    perp_diff = p1 - p2
    perp_mean = (p1 * w1 + p2 * w2).sum() / (w1 + w1).sum()
    if perp_diff.sum() == 0:
        return perp_diff + 1e-7, perp_mean
    else:
        return perp_diff, perp_mean


def count_GM(STA, CP, FL):
    return STA * CP * FL


CLS_ALGORITHM = 'MultinomialNB'
sta_model = joblib.load(f'../models/{CLS_ALGORITHM}.pkl')

meaning_model, meaning_tokenizer = load_model('cointegrated/LaBSE-en-ru', use_cuda=False,
                                              model_class=AutoModel)


def count_metrics(original_texts, rewritten_texts, ALL=False):
    if ALL:
        STA, _ = count_STA(rewritten_texts,
                           sta_model)
        CP = count_CP(model=meaning_model,
                      tokenizer=meaning_tokenizer,
                      original_texts=original_texts,
                      rewritten_texts=rewritten_texts,
                      batch_size=1,
                      verbose=False)
        FL, perp_mean = count_FL(original_texts,
                                 rewritten_texts)
        BLEU = count_BLEU(original_texts,
                          rewritten_texts)
        METEOR = count_METEOR(original_texts,
                              rewritten_texts)
        GM = count_GM(STA, CP, FL)
        return STA, CP, FL, perp_mean, BLEU, METEOR, GM

    else:
        probs, classes = count_STA(rewritten_texts,
                                   sta_model)
        STA = classes.mean()
        print('STA', STA)
        CP = count_CP(meaning_model=meaning_model,
                      meaning_tokenizer=meaning_tokenizer,
                      original_texts=original_texts,
                      rewritten_texts=rewritten_texts,
                      batch_size=1,
                      verbose=False).mean()
        print('CP', CP)
        diffs, FL = count_FL(original_texts,
                             rewritten_texts)
        print('FL', FL)
        BLEU = count_BLEU(original_texts,
                          rewritten_texts).mean()
        print('BLEU', BLEU)
        METEOR = count_METEOR(original_texts,
                              rewritten_texts).mean()
        print('METEOR', METEOR)
        GM = count_GM(STA, CP, FL)
        return STA, CP, FL, BLEU, METEOR, GM