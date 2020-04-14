import copy
from logging import info
import unittest

import numpy as np

from local_settings import bert_model_path, bert_tokenizer_name, pl_wordnet_path, annot_corpus_path
from wsd import is_nonalphabetic
from wsd.krnnt import tag_nkjp, split_sents_krnnt
from wsd.corpora import (
        AnnotatedCorpus, load_annotated_corpus, wordnet_corpus_for_lemmas, load_nkjp_ambiguous
        )
from wsd.bert import (
        average_embedding_list, average_embedding_matrix, bert_model, bert_tokenizer,
        embedded, lemma_form_in_sent, form_tokenization_indices, lemma_embeddings_in_sent,
        LemmaNotFoundError, CantMatchBERTTokensError, weight_wordpieces
        )
from wsd.embedding_dict import build_embedding_dict
from wsd.evaluate import embedding_dict_accuracy, compare_predictions

model = bert_model(bert_model_path)
tokenizer = bert_tokenizer(bert_tokenizer_name)

class TestWSDUtils(unittest.TestCase):

    def test_nonalphabetic(self):
        self.assertTrue(is_nonalphabetic('342244??!!'))
        self.assertFalse(is_nonalphabetic('I have exactly 72 elephants'))

    def test_averaging(self):
        example_matrix = np.array([[3, 3, 3, 4, 4, 4], [20, 20, 20, 100, 100, 100]])
        averaged_matrix = average_embedding_matrix(example_matrix)
        np.testing.assert_array_almost_equal(averaged_matrix,
                np.array([11.5, 11.5, 11.5, 52, 52, 52]))
        example_list = [np.array(elem)
                for elem in [[3, 3, 3, 4, 4, 4], [20, 20, 20, 100, 100, 100]]]
        averaged_list = average_embedding_list(example_list)
        np.testing.assert_array_almost_equal(averaged_list,
                np.array([11.5, 11.5, 11.5, 52, 52, 52]))

    def test_weighted_averaging(self):
        example_matrix = np.array([[3, 3, 3, 4, 4, 4], [20, 20, 20, 100, 100, 100]])
        averaged_matrix = average_embedding_matrix(example_matrix, weights=[2, 0.5])
        # numpy does scaling here - (2÷2,5)×3+(0,5÷2,5)×20 etc.
        np.testing.assert_array_almost_equal(averaged_matrix,
                np.array([6.4, 6.4, 6.4, 23.2, 23.2, 23.2]))
        example_list = [np.array(elem)
                for elem in [[3, 3, 3, 4, 4, 4], [20, 20, 20, 100, 100, 100]]]
        averaged_list = average_embedding_list(example_list, weights=[2, 0.5])
        np.testing.assert_array_almost_equal(averaged_list,
                np.array([6.4, 6.4, 6.4, 23.2, 23.2, 23.2]))

    def test_text_embeddings(self):
        sample_embedding = embedded('Witamy w jaskini', model, tokenizer)
        self.assertIsNotNone(sample_embedding)

        average = average_embedding_matrix(sample_embedding.numpy())
        self.assertTrue(isinstance(average, np.ndarray))
        self.assertEqual(list(average.shape), [768])

        average2 = average_embedding_list([average, average])
        self.assertTrue(isinstance(average2, np.ndarray))
        self.assertEqual(list(average2.shape), [768])

        with self.assertRaises(ValueError):
            average_embedding_list([])

    def test_nkjp(self):
        nkjp_interp = tag_nkjp('Ala ma kota.')
        self.assertEqual(nkjp_interp, [[
            ('Ala', 'Ala', 'subst:sg:nom:f'),
            ('ma', 'mieć', 'fin:sg:ter:imperf'),
            ('kota', 'kot', 'subst:sg:acc:m2'),
            ('.', '.', 'interp')]])
        nkjp_interp = tag_nkjp('')
        self.assertEqual(nkjp_interp, [])

    def test_nkjp_form_finding(self):
        form, form_n = lemma_form_in_sent('piec', 'Koło pieca postawiono miotłę')
        self.assertEqual(form, 'pieca')
        self.assertEqual(form_n, 1)
        form, form_n = lemma_form_in_sent('piec', 'Koło pieca postawiono miotłę i piec', num=2)
        self.assertEqual(form, 'piec')
        self.assertEqual(form_n, 1)
        with self.assertRaises(LemmaNotFoundError):
            lemma_form_in_sent('piec', 'Koło pieca postawiono miotłę', num=2)

    def test_krnnt_splitting(self):
        text = 'Ala ma kota. A kot ma Alę.'
        sents = split_sents_krnnt(text)
        self.assertEqual(sents, ['Ala ma kota.', 'A kot ma Alę.'])
        text = ''
        sents = split_sents_krnnt(text)
        self.assertEqual(sents, [])

    def test_tokens_matching(self):
        token_boundaries = form_tokenization_indices('jaskini', 'Witamy w jaskini Raj',
                tokenizer)
        self.assertEqual(token_boundaries, (4,7))
        token_boundaries = form_tokenization_indices('jaskini',
                'Witamy w jaskini Raj, najlepszej jaskini',
                tokenizer, num=2)
        # ['wit', '##amy', 'w', 'ja', '##skin', '##i', 'raj', ',', 'na', '##j', '##le', '##ps',
        # '##ze', '##j', 'ja', '##skin', '##i']
        self.assertEqual(token_boundaries, (15,18))
        with self.assertRaises(CantMatchBERTTokensError):
            form_tokenization_indices('jaskini', 'Witamy w jaskini Raj', tokenizer, num=2)

    def test_lemma_embeddings_in_sent(self):
        lemma_embeddings = lemma_embeddings_in_sent('jaskinia', 'Witamy w jaskini Raj', model,
                tokenizer)
        self.assertTrue(isinstance(lemma_embeddings, np.ndarray))
        self.assertEqual(list(lemma_embeddings.shape), [1, 3, 768])
        lemma_embeddings = lemma_embeddings_in_sent('jaskinia',
                'Witamy w jaskini Raj, najlepszej jaskini', model,
                tokenizer, num=2)
        self.assertTrue(isinstance(lemma_embeddings, np.ndarray))
        self.assertEqual(list(lemma_embeddings.shape), [1, 3, 768])

class TestCorporaAndDicts(unittest.TestCase):
    def is_good_annotated_corpus(self, corp, is_ambiguous=False):
        self.assertTrue(hasattr(corp, 'parsed_sents'))
        self.assertTrue(hasattr(corp, 'raw_sents'))
        self.assertTrue(hasattr(corp, 'lemmas'))
        self.assertTrue(isinstance(corp.parsed_sents, list))
        self.assertTrue(isinstance(corp.parsed_sents[0], list))
        self.assertTrue(isinstance(corp.parsed_sents[0][0], tuple))
        self.assertEqual(len(corp.parsed_sents[0][0]), 4 if not is_ambiguous else 3)
        self.assertTrue(isinstance(corp.raw_sents, list))
        self.assertTrue(isinstance(corp.raw_sents[0], str))
        self.assertTrue(isinstance(corp.lemmas, set))

    def make_sample_corp(self, sents):
        test_corp = AnnotatedCorpus()
        test_corp.raw_sents = sents
        for test_sent_n, test_sent in enumerate(sents):
            parsed_sent = tag_nkjp(test_sent)[0]
            # Add some fake sense information to the sentence.
            test_corp.parsed_sents.append([tok + (str(test_sent_n),)
                    for tok in parsed_sent])
            test_corp.lemmas = test_corp.lemmas.union(set([tok[1] for tok in parsed_sent]))
        return test_corp

    def make_sample_ambiguous_corp(self, sents):
        test_ambiguous_corp = AnnotatedCorpus()
        test_ambiguous_corp.raw_sents = sents
        test_ambiguous_corp.parsed_sents = tag_nkjp(sents)
        return test_ambiguous_corp

    def make_sample_emb_dict(self):
        # TODO test the error path
        test_corp = self.make_sample_corp(['Zimowe zające skaczą dzielnie po łące',
            # "dzielnie" here would actually be a slang noun, but probably the model will treat it
            # as an adverb still - completely ok for our purposes
            'Wojownik ulicy przemierzał dzielnie Warszawy'])
        test_emb_dict = build_embedding_dict(model, tokenizer, test_corp)
        return test_emb_dict

    def test_corp_ambiguous_version(self):
        test_corp = self.make_sample_corp(['Zimowe zające skaczą dzielnie po łące',
            'Wojownik ulicy przemierzał dzielnie Warszawy'])
        test_corp2 = test_corp.get_ambiguous_version()
        self.is_good_annotated_corpus(test_corp)
        self.is_good_annotated_corpus(test_corp2, is_ambiguous=True)
        self.assertFalse(test_corp.is_ambiguous())
        self.assertTrue(test_corp2.is_ambiguous())

    def test_load_annotated_corpus(self):
        info('Loading the annotated corpus...')
        train_corp, test_corp = load_annotated_corpus(annot_corpus_path)
        self.is_good_annotated_corpus(train_corp)
        self.is_good_annotated_corpus(test_corp)
        with self.assertRaises(Exception):
            load_annotated_corpus('nonsense')

    def test_nkjp_loading(self):
        nkjp_corpus = load_nkjp_ambiguous('test_resources/nkjp1m')
        self.is_good_annotated_corpus(nkjp_corpus, is_ambiguous=True)
        self.assertEqual(('o', 'o', 'interj'), nkjp_corpus.parsed_sents[0][0])
        self.assertEqual('o , nadzwyczajne wypadki .', nkjp_corpus.raw_sents[0])
        self.assertIn('o', nkjp_corpus.lemmas)
        with self.assertRaises(FileNotFoundError):
            load_nkjp_ambiguous('nonsense')

    def test_wordnet_corpus(self):
        sample_lemmas = {'pies', 'piec', 'koza'}
        info('Preparing a corpus from Wordnet...')
        wordnet_corp = wordnet_corpus_for_lemmas(pl_wordnet_path, sample_lemmas, model, tokenizer)
        self.is_good_annotated_corpus(wordnet_corp)
        with self.assertRaises(Exception):
            wordnet_corpus_for_lemmas('nonsense', sample_lemmas, model, tokenizer)

    def test_build_embedding_dict(self):
        test_emb_dict = self.make_sample_emb_dict()
        self.assertTrue(isinstance(test_emb_dict.embedding_lists, dict))
        self.assertTrue(isinstance(test_emb_dict.unified_embeddings, dict))
        self.assertIn('zając', test_emb_dict.embedding_lists)
        self.assertIn('0', test_emb_dict.embedding_lists['zając'])
        self.assertTrue(isinstance(test_emb_dict.embedding_lists['zając']['1'], list))

    def test_predict_sense_for_token(self):
        test_emb_dict = self.make_sample_emb_dict()
        pred1 = test_emb_dict.predict_sense_for_token('zająca', 'zając', 'Leśnik złapał zająca',
                form_num=1)
        pred2 = test_emb_dict.predict_sense_for_token('żubra', 'żubr', 'Leśnik złapał żubra',
                form_num=1)
        self.assertEqual(pred1, '0')
        self.assertIsNone(pred2)

    def test_predict(self):
        test_emb_dict = self.make_sample_emb_dict()
        nkjp_corpus = load_nkjp_ambiguous('test_resources/nkjp1m')
        prediction = test_emb_dict.predict(nkjp_corpus)
        self.assertTrue(isinstance(prediction, list))
        self.assertTrue(isinstance(prediction[0], list))
        self.assertTrue(isinstance(prediction[0][0], tuple))
        self.assertTrue(nkjp_corpus.is_ambiguous())
        prediction = test_emb_dict.predict(nkjp_corpus, case='best')
        self.assertTrue(isinstance(prediction, list))
        self.assertTrue(isinstance(prediction[0], list))
        self.assertTrue(isinstance(prediction[0][0], tuple))
        self.assertTrue(nkjp_corpus.is_ambiguous())
        prediction = test_emb_dict.predict(nkjp_corpus, case='worst')
        self.assertTrue(isinstance(prediction, list))
        self.assertTrue(isinstance(prediction[0], list))
        self.assertTrue(isinstance(prediction[0][0], tuple))
        self.assertTrue(nkjp_corpus.is_ambiguous())

    def test_compare_predictions(self):
        test_emb_dict = self.make_sample_emb_dict()
        nkjp_corpus = load_nkjp_ambiguous('test_resources/nkjp1m')
        prediction = test_emb_dict.predict(nkjp_corpus)
        prediction2 = copy.deepcopy(prediction)
        self.assertEqual(len(compare_predictions(prediction, prediction2)), 0)
        x = prediction2[0][0]
        prediction2[0][0] = (x[0], x[1], x[2], 'nonsense')
        self.assertEqual(len(compare_predictions(prediction, prediction2)), 1)

    def test_extend_with_ambiguous_corpus(self):
        test_emb_dict = self.make_sample_emb_dict()
        test_ambiguous_corp = self.make_sample_ambiguous_corp(
                ['Niedźwiedzie przemierzają dzielnie lasy'])
        test_emb_dict.extend_with_ambiguous_corpus(test_ambiguous_corp)
        self.assertTrue(len(test_emb_dict.embedding_lists['dzielnie']['0']) == 2
                or len(test_emb_dict.embedding_lists['dzielnie']['1']) == 2)

        # The incremental case.
        test_emb_dict = self.make_sample_emb_dict()
        test_emb_dict.extend_with_ambiguous_corpus(test_ambiguous_corp, incremental=True)
        self.assertTrue(len(test_emb_dict.embedding_lists['dzielnie']['0']) == 2
                or len(test_emb_dict.embedding_lists['dzielnie']['1']) == 2)

    def test_wordpiece_freqlist(self):
        test_emb_dict = self.make_sample_emb_dict()
        test_embedding1 = test_emb_dict.form_embedding_in_sent('przemierzają',
                'Niedźwiedzie przemierzają dzielnie lasy')
        nkjp_corpus = load_nkjp_ambiguous('test_resources/nkjp1m')
        test_emb_dict.set_wordpiece_weighting(nkjp_corpus)
        test_weights = weight_wordpieces(
                test_emb_dict.tokenizer.tokenize('Niedźwiedzie przemierzają'),
                test_emb_dict.wordpiece_freqlist)
        # We know that the first wordpiece, containing an uppercase letter, will have more weight
        # because of never appearing in the lowercase corpus
        self.assertGreater(test_weights[0], test_weights[1])
        test_embedding2 = test_emb_dict.form_embedding_in_sent('przemierzają',
                'Niedźwiedzie przemierzają dzielnie lasy')
        assert np.any(np.not_equal(test_embedding1, test_embedding2))

    def test_evaluate(self):
        test_emb_dict = self.make_sample_emb_dict()
        test_corp = self.make_sample_corp(['Zimowe zające skaczą dzielnie po łące',
            'Wojownik ulicy przemierzał dzielnie Warszawy'])
        result = embedding_dict_accuracy(test_emb_dict, test_corp)
        self.assertTrue(isinstance(result, dict))
        self.assertIn('all cases', result)
        self.assertIn('correct cases', result)
        self.assertIn('accuracy', result)

if __name__ == '__main__':
    unittest.main()
