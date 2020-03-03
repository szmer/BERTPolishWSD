import unittest
import numpy as np
from wsd import (
        average_embedding, bert_model, bert_tokenizer,
        embedded, is_nonalphabetic, tag_nkjp
        )

bert_model_path = 'bg_cs_pl_ru_cased_L-12_H-768_A-12_pt/'

class TestWSD(unittest.TestCase):
    def test_nonalphabetic(self):
        self.assertTrue(is_nonalphabetic('342244??!!'))
        self.assertFalse(is_nonalphabetic('I have exactly 72 elephants'))

    def test_embeddings(self):
        model = bert_model(bert_model_path)
        tokenizer = bert_tokenizer()

        max_length = 10
        sample_embedding1 = embedded('Witamy w jaskini', model, tokenizer, max_length=max_length)
        sample_embedding2 = embedded('Witamy w hotelu', model, tokenizer, max_length=max_length)
        self.assertIsNotNone(sample_embedding1)
        self.assertIsNotNone(sample_embedding2)

        average1, is_done1 = average_embedding([])
        self.assertEqual(average1, False)
        self.assertEqual(is_done1, False)

        average2, is_done2 = average_embedding([sample_embedding1, sample_embedding2])
        self.assertTrue(isinstance(average2, np.ndarray))
        self.assertEqual(list(average2.shape), [1, max_length, 768])
        self.assertEqual(is_done2, False)

    def test_nkjp(self):
        nkjp_interp = tag_nkjp('Ala ma kota.')
        self.assertEqual(nkjp_interp, [[
            ('Ala', 'Ala', 'subst:sg:nom:f'),
            ('ma', 'mieć', 'fin:sg:ter:imperf'),
            ('kota', 'kot', 'subst:sg:acc:m2'),
            ('.', '.', 'interp')]])

if __name__ == '__main__':
    unittest.main()
