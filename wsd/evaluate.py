from wsd import sequence_differences
from wsd.corpora import AnnotatedCorpus
from wsd.embedding_dict import EmbeddingDict

def embedding_dict_accuracy(emb_dict : EmbeddingDict, gold_corpus : AnnotatedCorpus, case='average'):
    test_corpus = gold_corpus.get_ambiguous_version()
    prediction = emb_dict.predict(test_corpus, case=case)
    all_sense_cases_count = 0
    no_info_cases_count = 0
    correct_sense_cases_count = 0
    for sent_n, sent in enumerate(gold_corpus.parsed_sents):
        for tok_n, (form, lemma, interp, true_sense) in enumerate(sent):
            if true_sense is not None:
                all_sense_cases_count += 1
                predicted_sense = prediction[sent_n][tok_n][3]
                if predicted_sense is None:
                    no_info_cases_count += 1
                elif predicted_sense == true_sense:
                    correct_sense_cases_count += 1
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

def compare_predictions(prediction1, prediction2):
    diffs = []
    # Compare each sentence as a sequence, because the diffing function expects seqs of individual
    # elements to be compared.
    for sent_n, sent in enumerate(prediction1):
        diffs += sequence_differences(sent, prediction2[sent_n])
    return diffs
