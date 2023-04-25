#!/usr/bin/env python

# Setup
import argparse
import os
import pyterrier as pt
pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
import pyterrier_dr
import pyt_splade
from pyterrier.measures import MRR, R
from breame.spelling import american_spelling_exists, british_spelling_exists, get_american_spelling, get_british_spelling
from breame.terminology import is_british_english_term, is_american_english_term
from pyterrier_colbert.ranking import ColBERTFactory
# from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer

argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--lang", help="query language (US or UK)", choices=['UK', 'US'])
argParser.add_argument("-m", "--model", help="Dense retriever", choices=['TCT-ColBERT', 'TasB', 'splade', 'colbert'])
argParser.add_argument("-s", "--set", help="query set to run experiments on (set1, set2, set3, set4)", choices=['set1','set2','set3','set4'])
argParser.add_argument("-f", "--force_rerank", help="forces re-ranking (True or False)", type=bool)
argParser.add_argument("-idx_dir", "--index_directory", help="index path", type=str, required=True)
argParser.add_argument("-out_dir", "--output_directory", help="path to save pyterrier output", type=str, required=True)

args = argParser.parse_args()

## HELPER Functions
def detect_terms_spell_any(query):
    '''
    Uses breame package to count number of UK/US spellings or language specific terms
    Input: query text
    Return: count of regional spelling and count of regional terms detected
    '''
    spell, term = 0, 0
    # tokenise query
    tokens = []
    for token in tokenizer(query):
        tokens.append(str(token))
        tokens.append(str(token.whitespace_))
    # detect dialect specific terms or spelling
    for token in tokens:
        if british_spelling_exists(token) or american_spelling_exists(token):
            spell += 1
        elif is_british_english_term(token) or is_american_english_term(token):
            term += 1
    return spell, term

def detect_spelling(query):
    '''
    Detects if text is using British or American spelling/terms and counts occurrences
    Input: query text
    Return: count of UK and US spelling/terms
    '''
    UK, US = 0, 0
    # tokenise query
    tokens = []
    for token in tokenizer(query):
        tokens.append(str(token))
        tokens.append(str(token.whitespace_))
    # detect dialect specific terms or spelling
    for token in tokens:
        if american_spelling_exists(token) or is_american_english_term(token):
            US += 1
        elif british_spelling_exists(token)or is_british_english_term(token):
            UK += 1
    return UK, US

def ConvUS(query):
    '''
    Uses breame to normalise query to American spelling
    Input: query text
    Return: query text normalised to American spelling
    '''
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
    return ''.join(tokens)

def ConvUK(query):
    '''
    Uses breame to normalise query to British spelling
    Input: query text
    Return: query text normalised to British spelling
    '''
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
    return ''.join(tokens)

if args.lang == 'UK':
    convDiff = ConvUS
    def detect_terms_spell(query):
        '''
        Detects if text is using language specific spelling and or terms
        '''
        spell, term = 0, 0
        # tokenise query
        tokens = []
        for token in tokenizer(query):
            tokens.append(str(token))
            tokens.append(str(token.whitespace_))
        # detect dialect specific terms or spelling
        for token in tokens:
            if british_spelling_exists(token):
                spell += 1
            elif is_british_english_term(token):
                term += 1
        return spell, term
elif args.lang == 'US':
    convDiff = ConvUK
    def detect_terms_spell(query):
        '''
        Detects if text is using language specific spelling and or terms
        '''
        spell, term = 0, 0
        # tokenise query
        tokens = []
        for token in tokenizer(query):
            tokens.append(str(token))
            tokens.append(str(token.whitespace_))
        # detect dialect specific terms or spelling
        for token in tokens:
            if american_spelling_exists(token):
                spell += 1
            elif is_american_english_term(token):
                term += 1
        return spell, term


# Load Data
dataset = pt.get_dataset("irds:msmarco-passage/dev/judged")
topics = dataset.get_topics()

# Query Filtering
## Filter Terms & Spelling
spellings, terms = [], []
for query in topics['query']:
    spell_temp, term_temp = detect_spelling(query)
    spellings.append(spell_temp)
    terms.append(term_temp)
topics['UK'] = spellings
topics['US'] = terms

# # Create Set 1 (Regional spelling & terms)
set1 = topics.copy()
if args.lang == 'UK':
    set1 = set1[(set1['UK'] > 0) & (set1['US'] == 0)]
elif args.lang == 'US':
    set1 = set1[(set1['US'] > 0) & (set1['UK'] == 0)]

spellings, terms = [], []
for query in set1['query']:
    spell_temp, term_temp = detect_terms_spell(query)
    spellings.append(spell_temp)
    terms.append(term_temp)
set1['spellings'] = spellings
set1['terms'] = terms

# # Create Set 2 (Only regional spelling)
set2 = set1.copy()
set2 = set2[(set2['terms'] == 0) & (set2['spellings'] > 0)]

# # Create Set 3 (Set1 normalised to opposite spelling)
set3 = set1.copy()
set3['query'] = set3['query'].apply(convDiff)

# # Create Set 4 (Set2 normalised to opposite spelling)
set4 = set2.copy()
set4['query'] = set4['query'].apply(convDiff)

if args.set == 'set1':
    SET = set1
elif args.set == 'set2':
    SET = set2
elif args.set == 'set3':
    SET = set3
elif args.set == 'set4':
    SET = set4

# Retrieval
## Setup
def convert_text(dataframe, converter):
    dataframe['text'] = dataframe['text'].apply(converter)
    return dataframe

if args.model in ['TCT-ColBERT', 'TasB']:
    if args.model == 'TCT-ColBERT':
        dense_model = pyterrier_dr.TctColBert(verbose=True)
    if args.model == 'TasB':
        dense_model = pyterrier_dr.TasB.dot(verbose=True)

    index = pyterrier_dr.FlexIndex(args.idx_dir)
    pipeline = dense_model >> index.np_retriever()

elif args.model == 'splade':
    factory = pyt_splade.SpladeFactory()
    pipeline = factory.query() >> pt.BatchRetrieve(args.idx_dir, wmodel='Tf')


elif args.model == 'colbert':
    index_path = "colbert" # YOU NEED TO CHANGE THIS to the index directory (and use idx_dir for the parent directory i.e /data/indices/)
    checkpoint = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    factory = ColBERTFactory(checkpoint, args.idx_dir, index_path, faiss_partitions=100,memtype='mem')
    factory.faiss_index_on_gpu = False
    pipeline = factory.end_to_end()


## Run Retrieval 
print()
print(f'Running DR Experiments for {args.lang} Queries, {args.set} with {args.index} Index')
if (not os.path.isfile(args.out_dir)) or args.force_rerank:
    print(f'Dense retrieval using {args.model} \n')
    dense_out = pipeline(SET)
    dense_res = pt.Experiment([dense_out], topics=SET, qrels=dataset.get_qrels().merge(SET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['DR pipeline'], verbose=True)
    print(dense_res)
    pt.io.write_results(dense_out, args.out_dir)
else:
    # read file and run experiments
    print(f'Loading DR Experiments Results for {args.lang} Queries, {args.set} with {args.index} Index')
    dense_out = pt.io.read_results(args.out_dir)
    dense_res = pt.Experiment([dense_out], topics=SET, qrels=dataset.get_qrels().merge(SET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['DR pipeline'], verbose=True)
    print(dense_res)