#!/usr/bin/env python

# Setup
import argparse
import os
import pyterrier as pt
pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
import pyterrier_dr
from pyterrier.measures import *
from breame.spelling import american_spelling_exists, british_spelling_exists, get_american_spelling, get_british_spelling
from breame.terminology import is_british_english_term, is_american_english_term
from pyterrier_t5 import MonoT5ReRanker
# from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import more_itertools
import torch
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer

class ElectraScorer(pt.transformer.TransformerBase):
    def __init__(self, model_name='crystina-z/monoELECTRA_LCE_nneg31', batch_size=16, text_field='text', verbose=True, cuda=True):
        super().__init__()
        import transformers
        self.model_name = model_name
        self.cuda = cuda
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

        self.tokeniser = transformers.AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        self.model = transformers.ElectraForSequenceClassification.from_pretrained(model_name).cuda()
        self.model.eval()
        if self.cuda:
            self.model = self.model.cuda()

    def transform(self, inp):
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='ELECTRA scoring')
        with torch.no_grad():
            for chunk in more_itertools.chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                toks = self.tokeniser(queries, texts, return_tensors='pt', padding=True, truncation=True)
                if self.cuda:
                    toks = {k: v.cuda() for k, v in toks.items()}
                scores.append(self.model(**toks).logits[:, 1].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res

argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--lang", help="query language (US or UK)", choices=['UK', 'US'])
argParser.add_argument("-m", "--model", help="neural model scorer", choices=['TCT_ColBERT', 'electra', 'monoT5', 'bm25'])
argParser.add_argument("-s", "--set", help="query set to run experiments on (set1, set2, set3, set4)", choices=['set1','set2','set3','set4'])
argParser.add_argument("-f", "--force_rerank", help="forces re-ranking (in-case result file already exists in target directory ) (True or False", action='store_true')
argParser.add_argument("-idx_dir", "--index_directory", help="index path", type=str, required=True)
argParser.add_argument("-out_dir", "--output_directory", help="path to save pyterrier output", type=str, required=True)



args = argParser.parse_args()
## Load Model
if args.model == 'TCT_ColBERT':
    second_stage_model = pyterrier_dr.TctColBert(verbose=True)
elif args.model == 'monoT5':
    second_stage_model = MonoT5ReRanker()
elif args.model == 'electra':
    second_stage_model = ElectraScorer(batch_size=1)

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

# RETRIEVAL EXPERIMENTS
## Setup
def convert_text(dataframe, converter):
    dataframe['text'] = dataframe['text'].apply(converter)
    return dataframe

# Load Index
index_ref = pt.IndexRef.of(args.idx_dir) # assumes you have already built an index if not use dialect_experiments_indexing.py script
if args.index == 'default':
    pipeline = pt.text.get_text(dataset, "text") >> second_stage_model
    DOC_NORM = ''
else:
    if args.index == 'ConvUK':
        pipeline = pt.text.get_text(dataset, "text") >> pt.apply.generic(lambda res : convert_text(res, ConvUK)) >> second_stage_model
    elif args.index =='ConvUS':
        pipeline = pt.text.get_text(dataset, "text") >> pt.apply.generic(lambda res : convert_text(res, ConvUS)) >> second_stage_model
    DOC_NORM = '_doc_norm'


bm25 = pt.BatchRetrieve(index_ref, wmodel='BM25',  verbose=True)
## Run Retrieval 
print()
print(f'Running Experiments for {args.lang} Queries, {args.set} with {args.index} Index')
# FIRST STAGE
print('Running first-stage retrieval')
first_stage_out = bm25(SET)
bm25_res = pt.Experiment([first_stage_out], topics=SET, qrels=dataset.get_qrels().merge(SET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['first-stage pipeline'],  verbose=True)
print(bm25_res)
# SECOND STAGE
if args.model != 'bm25':
    if (not os.path.isfile(args.out_dir)) or args.force_rerank:
        print(f'Running second-stage retrieval using {args.model}\n')
        second_stage_out = pipeline(first_stage_out)
        second_stage_res = pt.Experiment([second_stage_out], topics=SET, qrels=dataset.get_qrels().merge(SET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['second-stage pipeline (w doc norm)'], verbose=True)
        print(second_stage_res)
   
        print(f'Writing in {args.out_dir}')
        pt.io.write_results(second_stage_out, args.out_dir)
    else:
        # read file and run experiments
        print('Loading previously saved re-ranking results for {args.model}\n')
        second_stage_out = pt.io.read_results(args.out_dir)
        second_stage_res = pt.Experiment([second_stage_out], topics=SET, qrels=dataset.get_qrels().merge(SET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['second-stage pipeline (w doc norm)'], verbose=True)
        print(second_stage_res)