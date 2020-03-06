from wsd import (
        is_nonalphabetic, load_wn3_corpus,
        bert_model, bert_tokenizer, 
        average_embedding_matrix, average_embedding_list, form_embeddings_in_sent,
        lemma_embeddings_in_sent, split_sents_krnnt, LemmaNotFoundError, CantMatchBERTTokensError
        )

from lxml import etree
from scipy.spatial import distance

bert_model_path = 'bg_cs_pl_ru_cased_L-12_H-768_A-12_pt/'
pl_wordnet_path = 'plwordnet-3.1.xml'
corpus_path = 'plwordnet3-ipi-corp-annot.csv'

model = bert_model(bert_model_path)
tokenizer = bert_tokenizer()

##
## Load the corpus.
##
train_sents, train_words, test_sents, test_words = load_wn3_corpus(corpus_path, test_ratio=7)

##
## Collect the sense information from Wordnet.
##
print('Collecting senses data...')
# Word -> (sense -> list of embedding matrices)
# We expect to perform the actual averaging later on the fly.
word_senses = dict()
wordnet_xml = etree.parse(pl_wordnet_path)
no_matches = 0
no_matches_senses = 0
for lex_unit in wordnet_xml.iterfind('lexical-unit'):
    lemma = lex_unit.get('name').lower()
    variant = lex_unit.get('variant')
    gloss = lex_unit.get('desc')
    if lemma in train_words:
        bert_fail = False
        gloss_sents = [s for s in split_sents_krnnt(gloss) if not is_nonalphabetic(s)]
        lemma_fails = 0
        # Ignore mostly non-alphabetic sentences (symbols etc.).
        for sentence in gloss_sents:
            try:
                # Only now guarantee the dictionary entries to be able to count words with actual
                # embeddings.
                if not lemma in word_senses:
                    word_senses[lemma] = dict()
                if not variant in word_senses[lemma]:
                    word_senses[lemma][variant] = []
                word_senses[lemma][variant].append(
                        average_embedding_matrix(lemma_embeddings_in_sent(
                            lemma, sentence, model, tokenizer)))
            except LemmaNotFoundError:
                lemma_fails += 1
                continue
            except CantMatchBERTTokensError:
                print('Cannot match the form for {} in {}'.format(lemma, sentence))
                no_matches += 1
                if not bert_fail:
                    no_matches_senses += 1
                else:
                    bert_fail = True
        if lemma_fails == len(gloss_sents):
            print('Lemma {} not found in {}'.format(lemma, gloss))
print('{} BERT tokenization alignment failures for {} senses.'.format(
    no_matches, no_matches_senses))
print('{} words have some embeddings out of {} needed'.format(len(word_senses), len(train_words)))
##
## Check accuracy.
##
unembedded_words = 0
unembedded_senses = 0

def prepare_senses_entry(lemma):
    global unembedded_words, unembedded_senses
    has_embedding = False
    if lemma in word_senses:
        for sense in word_senses[lemma]:
            if isinstance(word_senses[lemma][sense], list):
                if len(word_senses[lemma][sense]) > 0:
                    word_senses[lemma][sense] = average_embedding_list(word_senses[lemma][sense])
                    has_embedding = True
                else:
                    unembedded_senses += 1
                    word_senses[lemma][sense] = False
    if not has_embedding:
        unembedded_words += 1

print('Evaluating accuracy...')
print()
correct_n = 0
all_n = 0
for sent_n, sent in enumerate(train_sents):
    print('{}/{}'.format(sent_n, len(train_sents)), end='\r') # overwrite the number
    sent_forms_str = ' '.join([entry[0] for entry in sent])
    # Count how many times forms and lemmas already appeared in the sentence.
    form_counts = dict()
    lemma_counts = dict()
    for form, lemma, tag, true_sense in sent:
        # Advance the counters if needed.
        if not form in form_counts:
            form_counts[form] = 1
        else:
            form_counts[form] += 1
        if not lemma in lemma_counts:
            lemma_counts[lemma] = 1
        else:
            lemma_counts[lemma] += 1
        # Try to disambiguate if we know the true sense.
        if true_sense is not None:
            token_embedding = average_embedding_matrix(
                    form_embeddings_in_sent(form, sent_forms_str, model, tokenizer,
                        num=form_counts[form]))
            prepare_senses_entry(lemma)
            best_sense = False
            lowest_distance = float('inf')
            if lemma in word_senses:
                for sense in word_senses[lemma]:
                    if not sense:
                        continue
                    local_distance = distance.cosine(word_senses[lemma][sense], token_embedding)
                    if local_distance < lowest_distance:
                        best_sense = sense
                        lowest_distance = local_distance
            if best_sense == true_sense:
                correct_n += 1
            all_n += 1
print('{} words and {} senses with no embeddings'.format(unembedded_words, unembedded_senses))
print('{} from {} correct ({})'.format(correct_n, all_n, 0 if all_n == 0 else correct_n/all_n))
