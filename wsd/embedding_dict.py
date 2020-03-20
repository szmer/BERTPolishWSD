from collections import defaultdict
from logging import error

from scipy.spatial import distance

from wsd.corpora import AnnotatedCorpus
from wsd.bert import (
        average_embedding_matrix, average_embedding_list, form_embeddings_in_sent,
        )

class EmbeddingDict(object):
    def __init__(self):
        # lemma -> sense -> a list of embeddings
        self.embedding_lists = dict()
        self.unified_embeddings = dict()
        self.model = 'no model yet'
        self.tokenizer = 'no tokenizer yet'

    def embedding_for_sense(self, lemma, sense):
        """
        Return the embedding for the lemma's sense of the given id, according to the example
        embeddings that we currently have stored.
        """
        return average_embedding_list(self.embedding_lists[lemma][sense]) 

    def predict_sense_for_embedding(self, lemma : str, token_embedding):
        """
        Get prediction for the token_embedding assuming the lemma, or None if we have no info for
        the lemma.
        """
        if not lemma in self.embedding_lists:
            return None
        best_sense = False
        lowest_distance = float('inf')
        for sense in self.embedding_lists[lemma]:
            sense_embedding = self.embedding_for_sense(lemma, sense)
            local_distance = distance.cosine(token_embedding, sense_embedding)
            if local_distance < lowest_distance:
                lowest_distance = local_distance
                best_sense = sense
        return best_sense

    def form_embedding_in_sent(self, form, raw_sent, num=1):
        return average_embedding_matrix(
                form_embeddings_in_sent(form, raw_sent, self.model, self.tokenizer, num=num))

    def predict_sense_for_token(self, form, lemma, raw_sent, form_num=1):
        token_embedding = self.form_embedding_in_sent(form, raw_sent, num=form_num)
        return self.predict_sense_for_embedding(lemma, token_embedding)

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

def build_embedding_dict(model, tokenizer, *corpora, catch_errors=True):
    errors_encountered = defaultdict(lambda: 0)
    emb_dict = EmbeddingDict()
    emb_dict.model = model
    emb_dict.tokenizer = tokenizer
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
                    except Exception as e:
                        if catch_errors:
                            error('Error {} encountered for {} in {}'.format(e, form, raw_sent))
                            errors_encountered[type(e).__name__] += 1
                        else:
                            raise e
                    emb_dict.embedding_lists[lemma][true_sense].append(token_embedding)
                form_counts[form] += 1
    if len(errors_encountered) > 0:
        error('Encountered errors: {}'.format([(key, value) for key, value
            in errors_encountered.items()]))
    return emb_dict
