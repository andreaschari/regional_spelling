#!/usr/bin/env python

# Setup
import argparse
import pyterrier as pt
pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
import pyterrier_dr
import pyt_splade
from breame.spelling import american_spelling_exists, british_spelling_exists, get_american_spelling, get_british_spelling
from breame.terminology import is_british_english_term, is_american_english_term
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--model", help="neural model", choices=['tct-colbert', 'tasb', 'splade', 'colbert', 'electra', 'monot5'])
argParser.add_argument("-idx_dir", "--index_directory", help="parent path to save index", type=str, required=True)

args = argParser.parse_args()

def ConvUS(query):
    tokens =[]
    # tokenize query
    for token in tokenizer(query):
        tokens.append(str(token))
        tokens.append(str(token.whitespace_))
    # change spelling
    for idx, token in enumerate(tokens):
        if not token.isspace() and token:
            if token.isupper():
                tokens[idx] = get_american_spelling(token).upper()
            elif token.istitle():
                tokens[idx] = get_american_spelling(token).title()
            else:
                tokens[idx] = get_american_spelling(token)
    #tokens = [s.translate(str.maketrans('', '', string.punctuation)) for s in tokens] #  remove punct
    return ''.join(tokens)

def ConvUK(query):
    tokens =[]
    # tokenize query
    for token in tokenizer(query):
        tokens.append(str(token))
        tokens.append(str(token.whitespace_))
    # change spelling
    for idx, token in enumerate(tokens):
        if not token.isspace() and token:
            if token.isupper():
                tokens[idx] = get_british_spelling(token).upper()
            elif token.istitle():
                tokens[idx] = get_british_spelling(token).title()
            else:
                tokens[idx] = get_british_spelling(token)
    #tokens = [s.translate(str.maketrans('', '', string.punctuation)) for s in tokens] #  remove punct
    return ''.join(tokens)

# Load Data
dataset = pt.get_dataset("irds:msmarco-passage/dev/judged")
topics = dataset.get_topics()

# # # # Index Data
## helper functions
def it_us():
  for doc in dataset.get_corpus_iter():
    # do what you want to the document here
    doc['text'] = ConvUS(doc['text'])
    yield doc

def it_uk():
  for doc in dataset.get_corpus_iter():
    # do what you want to the document here
    doc['text'] = ConvUK(doc['text'])
    yield doc

#Runs  Indexing
if args.model in ['monot5', 'electra']:
    print(f'Indexing: msmarco-passage')
    indexer = pt.IterDictIndexer(f'{args.idx_dir}/msmarco-passage')
    indexer.index(dataset.get_corpus_iter())

    indexer = pt.IterDictIndexer(f'{args.idx_dir}/msmarco-passage-ConvUK')
    indexer.index(it_uk())
    
    indexer = pt.IterDictIndexer(f'{args.idx_dir}/msmarco-passage-ConvUS')
    indexer.index(it_us())

if args.model in ['tasb','tct-colbert']:
    if args.model == 'tct-colbert':
        dense_model = pyterrier_dr.TctColBert(verbose=True)
    if args.model == 'tasb':
        dense_model = pyterrier_dr.TasB.dot(verbose=True)

    print(f'Indexing: msmarco-passage-dense-{args.model}')
    index = pyterrier_dr.FlexIndex(f"./indices/msmarco-passage-dense-{args.model}")
    idx_pipeline = dense_model >> index
    idx_pipeline.index(dataset.get_corpus_iter())

    index = pyterrier_dr.FlexIndex(f"{args.idx_dir}/msmarco-passage-dense-{args.model}-ConvUK")
    idx_pipeline = dense_model >> index
    idx_pipeline.index(it_uk())

    index = pyterrier_dr.FlexIndex(f"{args.idx_dir}/msmarco-passage-dense-{args.model}-ConvUS")
    idx_pipeline = dense_model >> index
    idx_pipeline.index(it_us())

elif args.model == 'splade':
    dense_model = pyt_splade.SpladeFactory()

    print(f'Indexing: msmarco-passage-dense-{args.model}')
    indexer = pt.IterDictIndexer(f"{args.idx_dir}/msmarco-passage-dense-{args.model}", pretokenised=True)
    indxr_pipe = dense_model.indexing() >> indexer
    index_ref = indxr_pipe.index(dataset.get_corpus_iter(), batch_size=128)

    indexer = pt.IterDictIndexer(f"{args.idx_dir}/msmarco-passage-dense-{args.model}-ConvUK", pretokenised=True)
    indxr_pipe = dense_model.indexing() >> indexer
    index_ref = indxr_pipe.index(it_uk(), batch_size=128)

    indexer = pt.IterDictIndexer(f"{args.idx_dir}/msmarco-passage-dense-{args.model}-ConvUS", pretokenised=True)
    indxr_pipe = dense_model.indexing() >> indexer
    index_ref = indxr_pipe.index(it_us(), batch_size=128)

elif args.model == 'colbert':
    checkpoint = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"

    print(f'Indexing: msmarco-passage-dense-{args.model}')
    indexer = ColBERTIndexer(checkpoint, "{args.idx_dir}/", f"msmarco-passage-dense-{args.model}", chunksize=3)
    indexer.index(dataset.get_corpus_iter())

    indexer = ColBERTIndexer(checkpoint, "{args.idx_dir}/", f"msmarco-passage-dense-{args.model}-ConvUK", chunksize=3)
    indexer.index(it_uk())

    indexer = ColBERTIndexer(checkpoint, "{args.idx_dir}/", f"msmarco-passage-dense-{args.model}-ConvUS", chunksize=3)
    indexer.index(it_us())