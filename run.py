import logging

from local_settings import (
        bert_model_path, bert_tokenizer_name, pl_wordnet_path, annot_corpus_path, nkjp_path,
        is_incremental
        )
from wsd.corpora import load_annotated_corpus, wordnet_corpus_for_lemmas, load_nkjp_ambiguous
from wsd.bert import bert_model, bert_tokenizer
from wsd.embedding_dict import build_embedding_dict
from wsd.evaluate import embedding_dict_accuracy

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

model = bert_model(bert_model_path)
tokenizer = bert_tokenizer(bert_tokenizer_name)

logging.info('Loading the annotated corpus...')
train_corp, test_corp = load_annotated_corpus(annot_corpus_path, test_ratio=7)
logging.info('Loading wordnet...')
wordnet_corp = wordnet_corpus_for_lemmas(pl_wordnet_path, train_corp.lemmas, model, tokenizer)
logging.info('Loading NKJP...')
nkjp_corp = load_nkjp_ambiguous(nkjp_path)

logging.info('Building the embedding dictionary...')
embeddings_dict = build_embedding_dict(model, tokenizer, train_corp, wordnet_corp)
logging.info('Extending embeddings with NKJP... (incremental: {})'.format(is_incremental))
embeddings_dict.extend_with_ambiguous_corpus(nkjp_corp, incremental=is_incremental)

logging.info('Accuracy evaluation...')
result = embedding_dict_accuracy(embeddings_dict, test_corp)
logging.info(result)
