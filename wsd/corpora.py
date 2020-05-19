import copy
import csv
import os
import re

from lxml import etree

from wsd import is_nonalphabetic
from wsd.krnnt import tag_nkjp, split_sents_krnnt

class AnnotatedCorpus(object):
    def __init__(self):
        # List of lists (sentences) of tokens as tuples:
        # Long version (disambiguated): (form, lemma, interp, lexical unit id [or None])
        # Short version (ambiguous): (form, lemma, interp, lexical unit id [or None])
        self.parsed_sents = []
        # Sentences as whole strings.
        self.raw_sents = []
        # All unique words that are present, lowercased.
        self.lemmas = set()

    def get_ambiguous_version(self):
        """
        Get an ambiguous version of a sense-annotated corpus (with sense information stripped).
        """
        new_corp = AnnotatedCorpus()
        new_corp.lemmas = copy.copy(self.lemmas)
        new_corp.raw_sents = copy.copy(self.raw_sents)
        for sent in self.parsed_sents:
            new_corp.parsed_sents.append([])
            for (form, lemma, interp, variant) in sent:
                new_corp.parsed_sents[-1].append((form, lemma, interp))
        return new_corp

    def is_ambiguous(self):
        return (len(self.parsed_sents[0][0]) == 3) if len(self.parsed_sents) > 0 else True

def load_annotated_corpus(annot_sentences_path, test_ratio=8):
    """
    Returns a list of sentences, as list of lemmas, and a set of words (only ones with sense
    annotations). The first has pairs: (form, lemma, tag, true_sense). Every (test_ratio)th sentence
    will be treated as belonging to the test set.
    """
    train_corp, test_corp = AnnotatedCorpus(), AnnotatedCorpus()
    with open(annot_sentences_path, newline='') as annot_file:
        annot_reader = csv.reader(annot_file)
        sent = []
        sent_n = 0
        for row in annot_reader:
            form, lemma, tag, true_sense = row[0], row[1], row[2], row[3]
            if form == '&' and lemma == '\\&\\':
                if sent_n % test_ratio == 0:
                    test_corp.parsed_sents.append(sent)
                    test_corp.raw_sents.append(' '.join([data[0] for data in sent]))
                else:
                    train_corp.parsed_sents.append(sent)
                    train_corp.raw_sents.append(' '.join([data[0] for data in sent]))
                sent = []
                sent_n += 1
            else:
                if not re.match('\\d+', true_sense):
                    true_sense = None
                sent.append((form, lemma.lower(), tag, true_sense))
                if true_sense is not None:
                    if sent_n % test_ratio == 0:
                        test_corp.lemmas.add(lemma.lower())
                    else:
                        train_corp.lemmas.add(lemma.lower())
    return train_corp, test_corp

def wordnet_corpus_for_lemmas(wordnet_path, lemmas, model, tokenizer):
    """
    Make a corpus of glosses from the wordnet XML for all senses of the lemmas.
    """
    wordnet_xml = etree.parse(wordnet_path)
    corpus = AnnotatedCorpus()
    corpus.lemmas = copy.copy(lemmas)
    for lex_unit in wordnet_xml.iterfind('lexical-unit'):
        lemma = lex_unit.get('name').lower()
        variant = lex_unit.get('variant')
        gloss = lex_unit.get('desc')
        if lemma in lemmas:
            parsed_sents = tag_nkjp(gloss)
            # Collect the non-alphabetic indices to also remove them from the raw sentences.
            removed_ids = set()
            for sent_n, sent in enumerate(parsed_sents):
                if is_nonalphabetic(' '.join([tok[0] for tok in sent])):
                        removed_ids.add(sent_n)
            parsed_sents = [sent for sent_n, sent in enumerate(parsed_sents)
                    if not sent_n in removed_ids]
            raw_sents = [sent for sent_n, sent in enumerate(split_sents_krnnt(gloss))
                    if not sent_n in removed_ids]
            assert len(parsed_sents) == len(raw_sents)
            # Add the sense annotations.
            for sent_n, sent in enumerate(parsed_sents):
                parsed_sents[sent_n] = [(form, encountered_lemma, interp, variant)
                        if encountered_lemma == lemma
                        else (form, encountered_lemma, interp, None)
                        for (form, encountered_lemma, interp) in sent]
            corpus.parsed_sents += parsed_sents
            corpus.raw_sents += raw_sents
    return corpus

# This is probably a bad idea, since many forms can contain frequent affixes not present in lemmas.
###-def all_wordnet_lemmas(wordnet_path, lowercase=True):
###-    """
###-    Extract all lemmas from the wordnet.
###-    """
###-    lemmas = set()
###-    wordnet_xml = etree.parse(wordnet_path)
###-    for lex_unit in wordnet_xml.iterfind('lexical-unit'):
###-        lemma = lex_unit.get('name')
###-        if lowercase:
###-            lemma = lemma.lower()
###-        lemmas.add(lemma)
###-    return lemmas

def load_nkjp_ambiguous(nkjp_path, lowercase=True):
    corpus = AnnotatedCorpus()
    some_text_read = False
    for dir_path, dirs, files in os.walk(nkjp_path):
        if 'ann_morphosyntax.xml' in files: # recognize corpus folders
            tree = etree.parse(dir_path+'/ann_morphosyntax.xml')
            some_text_read = True
            # tag is namespaced, .// for finding anywhere in the tree
            for sent_subtree in tree.iterfind(
                    './/{http://www.tei-c.org/ns/1.0}s[@{http://www.w3.org/XML/1998/namespace}id]'):
                parsed_sent = []
                for token in sent_subtree.iterfind('.//{http://www.tei-c.org/ns/1.0}seg'):
                    # Collect information about the token position.
                    for field in token.iterfind('.//{http://www.tei-c.org/ns/1.0}f[@name]'):
                        # Correct sentence graph paths.
                        if field.attrib['name'] == 'disamb':
                            # Lemma and tag.
                            token_data = field.find(
                                    './/{http://www.tei-c.org/ns/1.0}string').text.split(':')
                            lemma = token_data[0]
                            if lowercase:
                                lemma = lemma.lower()
                            interp = ':'.join(token_data[1:])
                        if field.attrib['name'] == 'orth':
                            form = field.find('.//{http://www.tei-c.org/ns/1.0}string').text
                            if lowercase:
                                form = form.lower()
                    parsed_sent.append((form, lemma, interp))
                    corpus.lemmas.add(lemma)
                raw_sent = ' '.join([token[0] for token in parsed_sent])
                corpus.parsed_sents.append(parsed_sent)
                corpus.raw_sents.append(raw_sent)
    if not some_text_read:
        raise FileNotFoundError('Cannot find NKJP files in the path {}'.format(nkjp_path))
    return corpus
