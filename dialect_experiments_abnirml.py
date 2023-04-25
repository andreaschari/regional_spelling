#!/usr/bin/env python

# Setup
import argparse
import os
import pyterrier as pt
pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
import pyterrier_dr
import pyt_splade
from pyterrier_t5 import MonoT5ReRanker
from pyterrier.measures import MRR, R
from breame.spelling import american_spelling_exists, british_spelling_exists, get_american_spelling, get_british_spelling
from breame.terminology import is_british_english_term, is_american_english_term
from pyterrier_colbert.ranking import ColBERTFactory, ColBERTModelOnlyFactory
# from nltk.tokenize import word_tokenize
import torch
import more_itertools
import numpy as np
import pandas as pd
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer

argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--lang", help="query language (US or UK)", choices=['UK', 'US'])
argParser.add_argument("-m", "--model", help="neural model scorer", choices=['tct-colbert', 'tasb', 'splade', 'colbert', 'electra', 'monot5'])
argParser.add_argument("-s", "--set", help="query set to run experiments on (set1, set2, set3, set4)", choices=['set1','set2','set3','set4'])
argParser.add_argument("-comp", "", help="whether to compare different or no normalisation", choices=['diff', 'nonorm'])

args = argParser.parse_args()

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

# # Create Set 1
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

# # Create Set 2
set2 = set1.copy()
set2 = set2[(set2['terms'] == 0) & (set2['spellings'] > 0)]

# # Create Set 3
set3 = set1.copy()
set3['query'] = set3['query'].apply(convDiff)

# # Create Set 4
set4 = set2.copy()
set4['query'] = set4['query'].apply(convDiff)

if args.set == 'set1':
    abnirml = set1.copy()
elif args.set == 'set2':
    abnirml = set2.copy()
elif args.set == 'set3':
    abnirml = set3.copy()
elif args.set == 'set4':
    abnirml = set4.copy()

## get qrels
qrels = dataset.get_qrels()
abnirml.drop(columns=['UK', 'US', 'spellings', 'terms'], inplace=True)
abnirml_w_qrels = abnirml.merge(qrels, on=['qid'])

# Retrieval
## Setup
def convert_text(dataframe, converter):
    dataframe['text'] = dataframe['text'].apply(converter)
    return dataframe

if args.model in ['tct-colbert', 'tasb']:
    if args.model == 'tct-colbert':
        scorer = pyterrier_dr.TctColBert()
    if args.model == 'tasb':
        scorer = pyterrier_dr.TasB.dot()

elif args.model == 'splade':
    factory = pyt_splade.SpladeFactory()
    scorer = factory.text_scorer()
elif args.model == 'colbert':
    checkpoint = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    factory = ColBERTModelOnlyFactory(checkpoint)    
    scorer = factory.text_scorer()

elif args.model == 'monot5':
    scorer = MonoT5ReRanker(verbose=False)

elif args.model == 'electra':
    scorer = ElectraScorer(batch_size=1, verbose=False)

print(f'Running ABNIRML experiments for {args.lang} Queries, {args.set} using {args.model} scorer')
# get ranking scores of query/qrels pairs norm. in same dialect and norm. in diff dialect
same_norm_pipeline = pt.text.get_text(dataset, "text") >> pt.apply.generic(lambda res : convert_text(res, convSame)) >> scorer
if  args.comp == 'diff':
    diff_norm_pipeline = pt.text.get_text(dataset, "text") >> pt.apply.generic(lambda res : convert_text(res, convDiff)) >> scorer #normalizes to opposite spelling
else:
    diff_norm_pipeline = pt.text.get_text(dataset, "text") >> scorer # no normalization

greater = 0
less = 0
for _, row in abnirml_w_qrels.iterrows():       
    data = [[row['qid'], row['query'], row['docno']]]
    df = pd.DataFrame(data, columns=['qid', 'query', 'docno'])
        
    out_same = same_norm_pipeline(df)
    out_diff = diff_norm_pipeline(df)

    if np.where(out_same['score'] > out_diff['score'], True, False):
        greater += 1        
    elif np.where(out_same['score'] < out_diff['score'], True, False):
        less += 1


print(f'ConvSame > {args.comp} Count: {greater}')
print(f'ConvSame < {args.comp} Count: {less}')
print(f"TOTAL Pairs : {abnirml_w_qrels.shape}")
