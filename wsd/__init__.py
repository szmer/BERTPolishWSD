import csv
import re
import http.client
import urllib.parse

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

class NetworkError(Exception):
    pass

class LemmaNotFoundError(ValueError):
    pass

class LemmaAmbiguousError(ValueError):
    pass

class CantMatchBERTTokensError(ValueError):
    pass

def is_nonalphabetic(text):
    return len(text) > 0 and len(re.sub('[^\\W0-9]', '', text)) / len(text) >= 0.65

def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def bert_model(path):
    return BertModel.from_pretrained(path)

def sublist_index(list1, list2):
    for i in range(len(list1)):
        if list1[i:i+len(list2)] == list2:
            return i
    return -1

def tag_nkjp(text):
    """
    Tag text with NKJP tagset. Returns a list of sentences as lists of (form, lemma, interp).
    """
    params = urllib.parse.urlencode({'text': text})
    headers = {'Content-type': 'application/x-www-form-urlencoded',
            'Accept': 'text/plain'}
    conn = http.client.HTTPConnection('localhost:9003') # where KRNNT resides
    conn.request('POST', '', params, headers)
    response = conn.getresponse()
    if response.status != 200:
        raise NetworkError('Cannot connect to KRNNT: {} {}, is the container running?'.format(
            response.status, response.reason))
    resp_html = response.read().decode('utf-8') # we get a HTML page and need to strip tags
    lines = resp_html[resp_html.index('<pre>')+len('<pre>')
            :resp_html.index('</pre>')].strip().split('\n')
    sents = [[]]
    current_token = None
    for line_n, line in enumerate(lines):
        # Next sentence.
        if len(line) == 0:
            sents.append([])
            continue
        # A form line - assign the form.
        if line_n % 2 == (0 if (len(sents) % 2 == 1) else 1):
            current_token = line.split('\t')[0]
        else:
            interp_data = line.split('\t')
            current_token = (current_token, interp_data[1], interp_data[2])
            sents[-1].append(current_token)
    return sents

def embedded(text, model, tokenizer):
    """
    Return embedding matrix for the whole text.
    """
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        return model(input_ids)[0]

def form_tokenization_indices(form, sent, tokenizer):
    """
    Find boundary indices for tokens corresponding to the form in the whole sentence's tokenization.
    """
    form_tokenization = tokenizer.tokenize(form)
    sent_tokenization = tokenizer.tokenize(sent)
    sub_idx = sublist_index(sent_tokenization, form_tokenization)
    if sub_idx == -1:
        raise CantMatchBERTTokensError('Cannot find {} in {}'.format(
            form_tokenization, sent_tokenization))
    return sub_idx, sub_idx+len(form_tokenization)

def lemma_form_in_sent(lemma, sent):
    """
    Return the form corresponding to the lemma in the sentence.
    """
    sents = tag_nkjp(sent)
    if len(sents) > 1:
        raise RuntimeError('Many sentences found in text: {}'.format(sent))
    nkjp_interp = sents[0]
    matching_tokens = [tok for tok in nkjp_interp if tok[1] == lemma]
    if len(matching_tokens) == 0:
        raise LemmaNotFoundError('Cannot find lemma {} in {}'.format(lemma, sent))
    elif len(matching_tokens) > 1:
        raise LemmaAmbiguousError('Found {} in {} {} times'.format(
            lemma, sent, len(matching_tokens)))
    return matching_tokens[0][0]

def lemma_embeddings_in_sent(lemma, sent, model, tokenizer):
    """
    Return the embeddings corresponding to the lemma in sent's embeddings.
    """
    try:
        form = lemma_form_in_sent(lemma, sent)
    except LemmaNotFoundError:
        return False
    # NOTE CantMatchBERTTokensError is not catched.
    tok_idcs = form_tokenization_indices(form, sent, tokenizer)
    sent_word_embeddings = embedded(sent, model, tokenizer)
    word_embeddings = sent_word_embeddings[:, tok_idcs[0]:tok_idcs[1], :]
    return word_embeddings.numpy()

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
    if len(embeddings) == 0:
        raise ValueError('An empty embeddings list')
    average_array = np.array(embeddings)
    return average_array.mean(axis=0)

def load_wn3_corpus(annot_sentences_path, test_ratio=8):
    """
    Returns a list of sentences, as list of lemmas, and a set of words. The first has pairs: (form,
    lemma, true_sense, tag). Every (test_ratio)th sentence will be treated as belonging to the test
    set.
    """
    train_sents = [] # pairs: (word lemma, lexical unit id [or None])
    test_sents = []
    train_words = set() # all unique words that are present
    test_words = set()
    with open(annot_sentences_path, newline='') as annot_file:
        annot_reader = csv.reader(annot_file)
        sent = []
        sent_n = 0
        for row in annot_reader:
            form, lemma, tag, true_sense = row[0], row[1], row[2], row[3]
            if form == '&' and lemma == '\\&\\':
                if sent_n % test_ratio == 0:
                    test_sents.append(sent)
                else:
                    train_sents.append(sent)
                sent = []
                sent_n += 1
            else:
                if re.match('\\d+', true_sense):
                    sent.append((form, lemma.lower(), '_'+true_sense, tag))
                else:
                    sent.append((form, lemma.lower(), None, tag))
                if sent_n % test_ratio == 0:
                    test_words.add(lemma.lower())
                else:
                    train_words.add(lemma.lower())
    return train_sents, train_words, test_sents, test_words
