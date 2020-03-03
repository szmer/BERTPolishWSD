import csv
import re

import torch
from transformers import BertModel, BertTokenizer

def is_nonalphabetic(text):
    return len(text) > 0 and len(re.sub('[^\\W0-9]', '', text)) / len(text) >= 0.65

def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def bert_model(path):
    return BertModel.from_pretrained(path)

def embedded(text, model, tokenizer, max_length=16):
    """
    Return embedding matrix for the whole text.
    """
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True,
        max_length=max_length, pad_to_max_length=True)])
    with torch.no_grad():
        return model(input_ids)[0]

def average_embedding(sense_entry):
    """
    Return two values. The first one is the average embedding (or False), and the second if the
    averaging was only now performed (and the word_senses dictionary needs to be updated).
    """
    if not isinstance(sense_entry, list):
        return sense_entry, True
    # If we have no embeddings to average, we have to give up on this sense.
    elif not sense_entry:
        return False, False
    else:
        average_array = sense_entry[0].numpy()
        for other_tensor in sense_entry[1:]:
            average_array += other_tensor.numpy()
        return average_array/len(sense_entry), False

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
