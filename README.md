# Overview

This package is for testing Polish word sense disambiguation with BERT.
Currently we're focusing on performing tests on the small plWordnet3-annotated corpus
made for [CoDeS](http://zil.ipipan.waw.pl/CoDeS).
We compare the BERT embedding of the token to disambiguate with embeddings
of tokens of the same lemma that we know are of certain sense (because they appear
in the reference corpus or Wordnet glosses).

This is a work in progress. It's also intended to deprecate the [gibber](https://github.com/szmer/gibber/) code down the line
(better code quality, models etc.).

# Installation

## Requirements
- Python 3.7 or newer
- pip
- virtualenv
- Docker

## Resources needed
- Slavic BERT files for pytorch from [DeepPavlov](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#bert)
- [Polish Wordnet](http://plwordnet.pwr.wroc.pl/wordnet/) (Słowosieć) XML file
- the [CoDeS small sense-annotated corpus](http://zil.ipipan.waw.pl/CoDeS?action=AttachFile&do=view&target=WSD-test-data.csv) of Polish
- optionally [NKJP1M](http://nkjp.pl/index.php?page=14&lang=1) (i.e., the 1-million subcorpus)
- [KRNNT](https://github.com/kwrobel-nlp/krnnt) (we install it below inside Docker)

## Installation process

```bash
docker pull djstrong/krnnt:1.0.1
virtualenv
source bin/activate
pip3 install -r requirements.txt # this may be just pip on some platforms
deactivate
```

## Running

In one terminal window:

```bash
docker run -p 9003:9003 -it djstrong/krnnt
# To kill, ctrl+c
```

In another terminal window:

```bash
source bin/activate
# After you review local_settings.py, run this to see the options:
python3 run.py --help
# (this may be just python instead of python3 on your machine)
# Plain `python3 run.py` will just train and test an embedding dictionary from Wordnet and the train corpus.
# After you're done:
deactivate
```

To test:
```
source bin/activate
python3 test.py
# After you're done:
deactivate
```
