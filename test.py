import unittest
import numpy as np
from wsd import (
        average_embedding_list, average_embedding_matrix, bert_model, bert_tokenizer,
        embedded, is_nonalphabetic,
        tag_nkjp, lemma_form_in_sent, form_tokenization_indices, lemma_embeddings_in_sent
        )

bert_model_path = 'bg_cs_pl_ru_cased_L-12_H-768_A-12_pt/'

class TestWSD(unittest.TestCase):
    model = bert_model(bert_model_path)
    tokenizer = bert_tokenizer()

    def test_nonalphabetic(self):
        self.assertTrue(is_nonalphabetic('342244??!!'))
        self.assertFalse(is_nonalphabetic('I have exactly 72 elephants'))

    def test_text_embeddings(self):
        sample_embedding = embedded('Witamy w jaskini', self.model, self.tokenizer)
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

    def test_nkjp_form_finding(self):
        form = lemma_form_in_sent('piec', 'Koło pieca postawiono miotłę')
        self.assertEqual(form, 'pieca')

    def test_tokens_matching(self):
        token_boundaries = form_tokenization_indices('jaskini', 'Witamy w jaskini Raj',
                self.tokenizer)
        self.assertEqual(token_boundaries, (3,6))

    def test_lemma_embeddings_in_sent(self):
        lemma_embeddings = lemma_embeddings_in_sent('jaskinia', 'Witamy w jaskini Raj', self.model,
                self.tokenizer)
        self.assertTrue(isinstance(lemma_embeddings, np.ndarray))
        self.assertEqual(list(lemma_embeddings.shape), [1, 3, 768])

if __name__ == '__main__':
    unittest.main()
