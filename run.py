import argparse
import logging
import pickle

from local_settings import (
        bert_model_path, bert_tokenizer_name, pl_wordnet_path, annot_corpus_path, nkjp_path
        )
from wsd.corpora import load_annotated_corpus, wordnet_corpus_for_lemmas, load_nkjp_ambiguous
from wsd.bert import bert_model, bert_tokenizer
from wsd.embedding_dict import build_embedding_dict
from wsd.evaluate import embedding_dict_accuracy, compare_predictions

argparser = argparse.ArgumentParser(description=
        'Run WSD experiments with estimating senses from examples with BERT.')
args = argparser.add_argument('--load', help='Load a premade embeddings dictionary.')
args = argparser.add_argument('--load2', help=
        'Load a second premade embeddings dictionary. You have to also compare.')
args = argparser.add_argument('--save', help='Save embeddings dictionary to the file.')
args = argparser.add_argument('--dont_extend',
        help='Use incremental strategy when extending dictionaries with corpora of ambiguous texts.',
        action='store_true')
args = argparser.add_argument('--incremental',
        help='Use incremental strategy when extending dictionaries with corpora of ambiguous texts.',
        action='store_true')
args = argparser.add_argument('--compare',
        help='Compare a newly made embedding dictionary\'s predictions with the loaded one\'s',
        action='store_true')
args = argparser.add_argument('--case',
        help='Use the prediction with average (default), best or worst case. The "average" case '
        'compares the embedding to average embedding of each sense; the "best" one selects the sense '
        'with one nearest embedding; the "worst" case selects the sense where the farther embedding '
        'is still nearer than farther ones of other senses.',
        default='average')

args = argparser.parse_args()

if args.compare and not args.load:
    raise ValueError(
            'You cannot compare unless you load and train another dictionary at the same time.')
if args.dont_extend and args.incremental:
    raise ValueError('You cannot use the incremental strategy when not extending.')
if not args.compare and args.load and (args.dont_extend or args.incremental):
    raise ValueError(
            'You cannot use dont_extend and incremental options when only loading an embeddings'
            +' dictionary.')
if args.load2 and not args.compare:
    raise ValueError(
            'You have to compare embedding dictionaries when you load two of them.')
if args.case and not args.case in ['average', 'best', 'worst']:
    raise ValueError('The case has to be one of average, best and worst.')

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

model = bert_model(bert_model_path)
tokenizer = bert_tokenizer(bert_tokenizer_name)

logging.info('Loading the annotated corpus...')
train_corp, test_corp = load_annotated_corpus(annot_corpus_path, test_ratio=7)

# embeddings_dict1 is trained or load2 if present, embeddings_dict2 can be only from --load
embeddings_dict1, embeddings_dict2 = None, None

# Training a new embeddings dictionary.
if (not args.load or args.compare) and not args.load2:
    logging.info('Loading wordnet...')
    wordnet_corp = wordnet_corpus_for_lemmas(pl_wordnet_path, train_corp.lemmas, model, tokenizer)

    logging.info('Building the embedding dictionary...')
    embeddings_dict1 = build_embedding_dict(model, tokenizer, train_corp, wordnet_corp)

    # Extending, if desired.
    if not args.dont_extend:
        logging.info('Loading NKJP...')
        nkjp_corp = load_nkjp_ambiguous(nkjp_path)
        logging.info('Extending embeddings with NKJP... (incremental: {})'.format(args.incremental))
        embeddings_dict1.extend_with_ambiguous_corpus(nkjp_corp, incremental=args.incremental)


    # Saving the embedding dictionary.
    if args.save:
        logging.info('Building the dictionary to disk...')
        with open(args.save, 'wb') as out_file:
            pickle.dump(embeddings_dict1, out_file)

# Loading a premade embeddings dictionary.
else:
    logging.info('Building the premade dictionary...')
    with open(args.load, 'rb') as premade_file:
        embeddings_dict2 = pickle.load(premade_file)

# load2, if requested, takes the place of the dictionary that would be trained.
if args.load2:
    logging.info('Building the premade dictionary...')
    with open(args.load2, 'rb') as premade_file:
        embeddings_dict1 = pickle.load(premade_file)

# Normal accuracy evaluation.
if not args.compare:
    logging.info('Accuracy evaluation (the {} case variant)...'.format(args.case))
    result = embedding_dict_accuracy(
            embeddings_dict1 if embeddings_dict2 is None else embeddings_dict2,
            test_corp, case=args.case)
    logging.info(result)
# Comparing responses.
else:
    logging.info('Comparing {} (left) and {} (right), the {} case variant...'.format(
        args.load2 if args.load2 is not None else 'currently built',
        args.load, args.case))
    test_corp_ambiguous = test_corp.get_ambiguous_version()
    prediction1 = embeddings_dict1.predict(test_corp_ambiguous, case=args.case)
    prediction2 = embeddings_dict2.predict(test_corp_ambiguous, case=args.case)
    for item in compare_predictions(prediction1, prediction2):
        print(item)
