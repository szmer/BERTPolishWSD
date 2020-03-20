from itertools import chain

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from wsd.krnnt import tag_nkjp

def sublist_index(list1, list2, num=1):
    """
    Find if list2 is contained in the list1 and return the index (or -1). The keyword num can be
    used to find subsequent occurences of list2 (one-based).
    """
    num_found = 0
    for i in range(len(list1)):
        if list1[i:i+len(list2)] == list2:
            num_found += 1
            if num_found == num:
                return i
    return -1

class LemmaNotFoundError(ValueError):
    pass

class CantMatchBERTTokensError(ValueError):
    pass

def bert_tokenizer(tokenizer_name):
    return BertTokenizer.from_pretrained(tokenizer_name)

def bert_model(path):
    return BertModel.from_pretrained(path)

def embedded(text, model, tokenizer):
    """
    Return embedding matrix for the whole text.
    """
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        return model(input_ids)[0]

def form_tokenization_indices(form, sent, tokenizer, num=1):
    """
    Find boundary indices for tokens corresponding to the form in the whole sentence's BERT
    tokenization. If there are less occurences than requested num, CantMatchBERTTokensError is
    raised. The function assumes that the form will be tokenized the same way in isolation as in
    the sentence.
    """
    form_tokenization = tokenizer.tokenize(form)
    sent_tokenization = tokenizer.tokenize(sent)
    sub_idx = sublist_index(sent_tokenization, form_tokenization, num=num)
    if sub_idx == -1:
        raise CantMatchBERTTokensError('Cannot find {}th {} in {}'.format(
            num, form_tokenization, sent_tokenization))
    return sub_idx, sub_idx+len(form_tokenization)

# NOTE unused?
def lemma_form_in_sent(lemma, sent, num=1):
    """
    Return the num'th form corresponding to the lemma in the sentence, and the number of that form
    amongst the identical forms (one-based). Here num refers to the lemma's occurence according to
    NKJP-style tokenization achieved with krnnt.tag_nkjp function.
    """
    sents = tag_nkjp(sent)
    if len(sents) == 0:
        raise RuntimeError('No sentences found in text: {}'.format(len(sents), sent))
    # Even if there are more that one sentence, collapse them.
    nkjp_interp = list(chain.from_iterable(sents))
    matching_tokens = [(tok, tok_n) for (tok_n, tok) in enumerate(nkjp_interp) if tok[1] == lemma]
    if len(matching_tokens) < num:
        raise LemmaNotFoundError('Cannot find {}th lemma {} in {}'.format(num, lemma, sent))
    form = matching_tokens[num-1][0][0] # 0 0 to select the form and the token, not its n
    tok_n = matching_tokens[num-1][1]
    return form, len([tok for tok in nkjp_interp[:tok_n] if tok[0] == form])+1

def form_embeddings_in_sent(form, sent, model, tokenizer, num=1):
    """
    Return the embeddings corresponding to the num'th occurence of a the form in sent's
    embeddings. This number refers to the occurence in the sentence, which can fail due to BERT
    tokenization discrepancy (CantMatchBERTTokensError, compare form_tokenization_indices).
    """
    # NOTE CantMatchBERTTokensError is not catched.
    # Get the indices of the token 
    tok_idcs = form_tokenization_indices(form, sent, tokenizer, num=num)
    sent_word_embeddings = embedded(sent, model, tokenizer)
    word_embeddings = sent_word_embeddings[:, tok_idcs[0]:tok_idcs[1], :]
    return word_embeddings.numpy()

def lemma_embeddings_in_sent(lemma, sent, model, tokenizer, num=1):
    """
    Return the embeddings corresponding to the num'th occurence of a lemma's form in sent's
    embeddings. This number refers to the occurence in the sentence, which can fail due to BERT
    tokenization discrepancy (CantMatchBERTTokensError, compare form_tokenization_indices).
    """
    form, form_n = lemma_form_in_sent(lemma, sent, num=num)
    return form_embeddings_in_sent(form, sent, model, tokenizer, num=form_n)

def average_embedding_matrix(embeddings):
    """
    Return the average Numpy embedding vector, given a matrix of a couple embeddings for separate
    tokens as a Numpy array.
    """
    average_array = embeddings[0]
    for other_tensor in embeddings[1:]:
        average_array += other_tensor
    return average_array.mean(axis=0)

def average_embedding_list(embeddings):
    """
    Return the average Numpy embedding vector, given a list of embeddings to be averaged.
    """
    if len(embeddings) == 0:
        raise ValueError('An empty embeddings list')
    average_array = np.array(embeddings)
    return average_array.mean(axis=0)
