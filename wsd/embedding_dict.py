from collections import defaultdict
from logging import error

from scipy.spatial import distance

from wsd.corpora import AnnotatedCorpus
from wsd.bert import (
        average_embedding_matrix, average_embedding_list, form_embeddings_in_sent,
        tokenization_freqlist, weight_wordpieces
        )

class EmbeddingDict(object):
    def __init__(self):
        # lemma -> sense -> a list of embeddings
        self.embedding_lists = dict()
        self.unified_embeddings = dict()
        self.model = 'no model yet'
        self.tokenizer = 'no tokenizer yet'
        self.cut_wordpieces = False

    def set_wordpiece_weighting(self, corpus):
        """
        Count wordpiece occurences from the corpus and use them to weight wordpieces during mapping
        BERT to NKJP-style forms.
        """
        corpus_forms = set([token_data[0]
            for sent in corpus.parsed_sents
            for token_data in sent])
        self.wordpiece_freqlist = tokenization_freqlist(corpus_forms, self.tokenizer)

    def set_cut_wordpieces(self):
        """
        Make the embedding dictionaries ignore all wordpieces beyond the fourth one in each form.
        """
        self.cut_wordpieces = True

    def embedding_for_sense(self, lemma, sense):
        """
        Return the embedding for the lemma's sense of the given id, according to the example
        embeddings that we currently have stored.
        """
        return average_embedding_list(self.embedding_lists[lemma][sense]) 

    def predict_sense_for_embedding(self, lemma : str, token_embedding, case='average'):
        """
        Get prediction for the token_embedding assuming the lemma, or None if we have no info for
        the lemma. The 'average' case compares the embedding to average embedding of each sense;
        the 'best' one selects the sense with one nearest embedding; the 'worst' case selects the
        sense where the farther embedding is still nearer than farther ones of other senses.
        """
        if not lemma in self.embedding_lists:
            return None
        best_sense = False
        lowest_distance = float('inf')
        for sense in self.embedding_lists[lemma]:
            if case == 'average':
                sense_embedding = self.embedding_for_sense(lemma, sense)
                local_distance = distance.cosine(token_embedding, sense_embedding)
                if local_distance < lowest_distance:
                    lowest_distance = local_distance
                    best_sense = sense
            elif case == 'best':
                for sense_embedding in self.embedding_lists[lemma][sense]:
                    local_distance = distance.cosine(token_embedding, sense_embedding)
                    if local_distance < lowest_distance:
                        lowest_distance = local_distance
                        best_sense = sense
            elif case == 'worst':
                all_better = True
                for sense_embedding in self.embedding_lists[lemma][sense]:
                    local_distance = distance.cosine(token_embedding, sense_embedding)
                    if local_distance >= lowest_distance:
                        all_better = False
                if all_better:
                    lowest_distance = local_distance
                    best_sense = sense
        return best_sense

    def form_embedding_in_sent(self, form, raw_sent, num=1):
        embeddings = form_embeddings_in_sent(form, raw_sent, self.model, self.tokenizer, num=num)
        if self.cut_wordpieces:
            if len(embeddings.shape) == 3 and embeddings.shape[0] == 1:
                embeddings = embeddings[0][:4]
            elif len(embeddings.shape) == 2:
                embeddings = embeddings[:4]
            else:
                raise ValueError('got bad embeddings dimensions')
        if not hasattr(self, 'wordpiece_freqlist'):
            return average_embedding_matrix(embeddings)
        else:
            return average_embedding_matrix(embeddings,
                    weights=(weight_wordpieces(
                        self.tokenizer.tokenize(form)
                        if not self.cut_wordpieces
                        else weight_wordpieces(self.tokenizer.tokenize(form)[:4])),
                        self.wordpiece_freqlist))

    def predict_sense_for_token(self, form, lemma, raw_sent, form_num=1, case='average'):
        if not lemma in self.embedding_lists:
            return None
        token_embedding = self.form_embedding_in_sent(form, raw_sent, num=form_num)
        return self.predict_sense_for_embedding(lemma, token_embedding, case=case)

    def extend_with_ambiguous_corpus(self, corpus : AnnotatedCorpus, incremental=False,
            catch_errors=True):
        """
        Extend the embeddings lists with examples taken from the corpus that has no sense
        information (i.e., an AnnotatedCorpus with short tuples).
        """
        # TODO incremental
        errors_encountered = defaultdict(lambda: 0)
        # lemma -> sense -> a list of embeddings
        additional_embeddings = {lemma: defaultdict(list) for lemma in self.embedding_lists}
        for sent_n, sent in enumerate(corpus.parsed_sents):
            raw_sent = corpus.raw_sents[sent_n]
            # Keeping track of possible occurences of the same forms or lemmas in the sentence. Note
            # that this may lead to discrepancy with BERT tokenization; compare notes for the
            # bert.form_tokenization_indices function.
            form_counts = { data[0]: 1 for data in sent }
            for (form, lemma, interp) in sent:
                if lemma in self.embedding_lists:
                    try:
                        token_embedding = self.form_embedding_in_sent(form, raw_sent,
                                num=form_counts[form])
                        best_sense = self.predict_sense_for_embedding(lemma, token_embedding)
                    except Exception as e:
                        if catch_errors:
                            error('Error {} encountered for {} in {}'.format(e, form, raw_sent))
                            errors_encountered[type(e).__name__] += 1
                        else:
                            raise e
                    if not incremental:
                        additional_embeddings[lemma][best_sense].append(token_embedding)
                    # In the incremental case, add the embedding example directly to the dictionary.
                    else:
                        self.embedding_lists[lemma][best_sense].append(token_embedding)
                form_counts[form] += 1
        if not incremental:
            for lemma in additional_embeddings:
                for sense in additional_embeddings[lemma]:
                    self.embedding_lists[lemma][sense] += additional_embeddings[lemma][sense]
        if len(errors_encountered) > 0:
            error('Encountered errors: {}'.format([(key, value) for key, value
                in errors_encountered.items()]))

    def predict(self, corpus, case='average'):
        """
        Predict senses for a whole ambiguous corpus. Return a list of sentences in the disambiguated
        corpus format (tuples: form, lemma, NKJP interp, sense id).
        """
        assert corpus.is_ambiguous()
        disamb_sents = []
        for sent_n, sent in enumerate(corpus.parsed_sents):
            disamb_sents.append([])
            raw_sent = corpus.raw_sents[sent_n]
            # Keeping track of possible occurences of the same forms or lemmas in the sentence. Note
            # that this may lead to discrepancy with BERT tokenization; compare notes for the
            # bert.form_tokenization_indices function.
            form_counts = { data[0]: 1 for data in sent }
            for (form, lemma, interp) in sent:
                best_sense = self.predict_sense_for_token(form, lemma, raw_sent,
                        form_num=form_counts[form], case=case)
                disamb_sents[-1].append((form, lemma, interp, best_sense))
        return disamb_sents

def build_embedding_dict(model, tokenizer, *corpora, catch_errors=True,
        count_wordpieces_in=None, cut_wordpieces=False):
    errors_encountered = defaultdict(lambda: 0)
    emb_dict = EmbeddingDict()
    emb_dict.model = model
    emb_dict.tokenizer = tokenizer
    if count_wordpieces_in is not None:
        emb_dict.set_wordpiece_weighting(count_wordpieces_in)
    if cut_wordpieces:
        emb_dict.set_cut_wordpieces()
    # Prepare the dictionary. We want to devote a full loop pass to it not to overwrite some
    # important values with update()
    for corp in corpora:
        emb_dict.embedding_lists.update({lemma: defaultdict(list) for lemma in corp.lemmas})
    for corp in corpora:
        for sent_n, sent in enumerate(corp.parsed_sents):
            raw_sent = corp.raw_sents[sent_n]
            # Keeping track of possible occurences of the same forms or lemmas in the sentence. Note
            # that this may lead to discrepancy with BERT tokenization; compare notes for the
            # bert.form_tokenization_indices function.
            form_counts = { data[0]: 1 for data in sent }
            for form, lemma, tag, true_sense in sent:
                if true_sense is not None:
                    try:
                        token_embedding = emb_dict.form_embedding_in_sent(form, raw_sent,
                                num=form_counts[form])
                        emb_dict.embedding_lists[lemma][true_sense].append(token_embedding)
                    except Exception as e:
                        if catch_errors:
                            error('Error {} encountered for {} in {}'.format(e, form, raw_sent))
                            errors_encountered[type(e).__name__] += 1
                        else:
                            raise e
                form_counts[form] += 1
    if len(errors_encountered) > 0:
        error('Encountered errors: {}'.format([(key, value) for key, value
            in errors_encountered.items()]))
    return emb_dict
