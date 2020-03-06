import csv
from itertools import chain
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

class CantMatchBERTTokensError(ValueError):
    pass

def is_nonalphabetic(text):
    return len(text) > 0 and len(re.sub('[^\\W0-9]', '', text)) / len(text) >= 0.65

def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def bert_model(path):
    return BertModel.from_pretrained(path)

def sublist_index(list1, list2, num=1):
    num_found = 0
    for i in range(len(list1)):
        if list1[i:i+len(list2)] == list2:
            num_found += 1
            if num_found == num:
                return i
    return -1

def krnnt_response_lines(text):
    """
    Get KRNNT's analysis of text as a list of lines.
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
    return lines

def tag_nkjp(text):
    """
    Tag text with NKJP tagset. Returns a list of sentences as lists of (form, lemma, interp).
    """
    lines = krnnt_response_lines(text)
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
    sents = [sent for sent in sents if len(sent) > 0]
    return sents

def split_sents_krnnt(text, strip_sents=True):
    """
    Use KRNNT to split the text into a list of sentence strings.
    """
    lines = krnnt_response_lines(text)
    sents = ['']
    for line_n, line in enumerate(lines):
        # Next sentence.
        if len(line) == 0:
            sents.append('')
            continue
        # A form line - grow the sentence.
        if line_n % 2 == (0 if (len(sents) % 2 == 1) else 1):
            form_data = line.split('\t')
            form = form_data[0]
            if form_data[1] == 'space':
                preceding_sep = ' '
            elif form_data[1] == 'none':
                preceding_sep = ''
            elif form_data[1] == 'newline':
                preceding_sep = '\n'
            sents[-1] += preceding_sep + form
    if strip_sents:
        sents = [sent.strip() for sent in sents]
    sents = [sent for sent in sents if len(sent) > 0]
    return sents

def embedded(text, model, tokenizer):
    """
    Return embedding matrix for the whole text.
    """
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        return model(input_ids)[0]

def form_tokenization_indices(form, sent, tokenizer, num=1):
    """
    Find boundary indices for tokens corresponding to the form in the whole sentence's tokenization.
    """
    form_tokenization = tokenizer.tokenize(form)
    sent_tokenization = tokenizer.tokenize(sent)
    sub_idx = sublist_index(sent_tokenization, form_tokenization, num=num)
    if sub_idx == -1:
        raise CantMatchBERTTokensError('Cannot find {}th {} in {}'.format(
            num, form_tokenization, sent_tokenization))
    return sub_idx, sub_idx+len(form_tokenization)

def lemma_form_in_sent(lemma, sent, num=1):
    """
    Return the num'th form corresponding to the lemma in the sentence, and the number of that form
    amongst the identical forms (one-based).
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
    # NOTE CantMatchBERTTokensError is not catched.
    tok_idcs = form_tokenization_indices(form, sent, tokenizer, num=num)
    sent_word_embeddings = embedded(sent, model, tokenizer)
    word_embeddings = sent_word_embeddings[:, tok_idcs[0]:tok_idcs[1], :]
    return word_embeddings.numpy()

def lemma_embeddings_in_sent(lemma, sent, model, tokenizer, num=1):
    """
    Return the embeddings corresponding to the num'th occurence of a lemma's form in sent's
    embeddings.
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
    if len(embeddings) == 0:
        raise ValueError('An empty embeddings list')
    average_array = np.array(embeddings)
    return average_array.mean(axis=0)

def load_wn3_corpus(annot_sentences_path, test_ratio=8):
    """
    Returns a list of sentences, as list of lemmas, and a set of words. The first has pairs: (form,
    lemma, tag, true_sense). Every (test_ratio)th sentence will be treated as belonging to the test
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
                    sent.append((form, lemma.lower(), tag, true_sense))
                else:
                    sent.append((form, lemma.lower(), tag, None))
                if sent_n % test_ratio == 0:
                    test_words.add(lemma.lower())
                else:
                    train_words.add(lemma.lower())
    return train_sents, train_words, test_sents, test_words
