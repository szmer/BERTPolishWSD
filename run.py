from wsd import (
        is_nonalphabetic, load_wn3_corpus,
        bert_model, bert_tokenizer, 
        average_embedding_matrix, average_embedding_list, lemma_embeddings_in_sent
        )

from lxml import etree
import numpy as np
import tokenizesentences

bert_model_path = 'bg_cs_pl_ru_cased_L-12_H-768_A-12_pt/'
pl_wordnet_path = 'plwordnet-3.1.xml'
corpus_path = 'plwordnet3-ipi-corp-annot.csv'

splitter = tokenizesentences.SplitIntoSentences()
model = bert_model(bert_model_path)
tokenizer = bert_tokenizer()

##
## Load the corpus.
##
train_sents, train_words, test_sents, test_words = load_wn3_corpus(corpus_path, test_ratio=7)

##
## Collect the sense information from Wordnet.
##
# Word -> (sense -> list of embedding matrices)
# We expect to perform the actual averaging later on the fly.
word_senses = dict()
wordnet_xml = etree.parse(pl_wordnet_path)
for lex_unit in wordnet_xml.iterfind('lexical-unit'):
    lemma = lex_unit.get('name').lower()
    variant = lex_unit.get('variant')
    gloss = lex_unit.get('desc')
    if lemma in train_words:
        if not lemma in word_senses:
            word_senses[lemma] = dict()
        if not variant in word_senses[lemma]:
            word_senses[lemma][variant] = []
        # Ignore mostly non-alphabetic sentences (symbols etc.).
        for sentence in [s for s in splitter.split_into_sentences(gloss) if not is_nonalphabetic(s)]:
            word_senses[lemma][variant].append(
                    average_embedding_matrix(lemma_embeddings_in_sent(
                        lemma, sentence, model, tokenizer)))

##
## (Preliminary stats).
##
print('Collecting stats...')
print(len(word_senses))
words = 0
words_with_no_avg = 0
all_ss = 0
ss_with_avg = 0
for word in word_senses:
    words += 1
    avg_found = False
    for sense in word_senses[word]:
        all_ss += 1
        avg, _ = average_embedding_list(word_senses[word][sense])
        if isinstance(avg, np.ndarray):
            ss_with_avg += 1
            avg_found = True
    if not avg_found:
        words_with_no_avg += 1
print(words, words_with_no_avg, all_ss, ss_with_avg)
