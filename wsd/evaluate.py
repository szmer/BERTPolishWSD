from wsd.corpora import AnnotatedCorpus
from wsd.embedding_dict import EmbeddingDict

def embedding_dict_accuracy(emb_dict : EmbeddingDict, gold_corpus : AnnotatedCorpus):
    all_sense_cases_count = 0
    no_info_cases_count = 0
    correct_sense_cases_count = 0
    for sent_n, sent in enumerate(gold_corpus.parsed_sents):
        raw_sent = gold_corpus.raw_sents[sent_n]
        # Keeping track of possible occurences of the same forms or lemmas in the sentence. Note
        # that this may lead to discrepancy with BERT tokenization; compare notes for the
        # bert.form_tokenization_indices function.
        form_counts = { data[0]: 1 for data in sent }
        for (form, lemma, interp, true_sense) in sent:
            if true_sense is not None:
                all_sense_cases_count += 1
                best_sense = emb_dict.predict_sense_for_token(form, lemma, raw_sent,
                        form_num=form_counts[form])
                if best_sense is None:
                    no_info_cases_count += 1
                elif best_sense == true_sense:
                    correct_sense_cases_count += 1
            form_counts[form] += 1
    return {
            'all cases': all_sense_cases_count, 'correct cases': correct_sense_cases_count,
            'cases with no info': no_info_cases_count,
            'accuracy': (correct_sense_cases_count/all_sense_cases_count)
            # avoid division by zero:
            if all_sense_cases_count != 0 else 'n/a',
            'informed accuracy': (correct_sense_cases_count
            / (all_sense_cases_count-no_info_cases_count))
            if (all_sense_cases_count-no_info_cases_count) != 0 else 'n/a',
            }
