#!/usr/bin/env python

# Setup
import argparse
import pyterrier as pt
pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
from pyterrier.measures import MRR, R
from breame.spelling import american_spelling_exists, british_spelling_exists, get_american_spelling, get_british_spelling
from breame.terminology import is_british_english_term, is_american_english_term
# from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from scipy.stats import ttest_ind, ranksums, wilcoxon
from statsmodels.stats.weightstats import ttost_ind, ttost_paired
nlp = English()
tokenizer = nlp.tokenizer

argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--lang", help="query language (US or UK)", choices=['UK', 'US'])
argParser.add_argument("-m", "--model", help="Model", choices=['TCT_ColBERT', 'TasB', 'splade', 'colbert', 'electra', 'monoT5', 'bm25'])
argParser.add_argument("-s", "--set", help="query set to run experiments on (set1, set2, set3, set4)", choices=['set1','set2','set3','set4'])
argParser.add_argument("-pipeline_one_dir",  help="path of first pipeline results", type=str, required=True)
argParser.add_argument("-pipeline_two_dir", help="path of second pipeline results", type=str, required=True)

args = argParser.parse_args()

def detect_terms_spell_any(query):
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
    Detects if text is using British or American spelling
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
            #print('US: ' + token)
        elif british_spelling_exists(token)or is_british_english_term(token):
            UK += 1
            #print('UK: ' + token)
    return UK, US

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

if args.lang == 'UK':
    convDiff = ConvUS
    convSame = ConvUK
elif args.lang == 'US':
    convDiff = ConvUK
    convSame = ConvUS

def detect_terms_spell_cross(query):
    '''
    Detects if text is using language specific spelling and or terms
    '''
    if args.lang == 'US':
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
    elif args.lang == 'UK':
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

def detect_terms_spell(query):
    '''
    Detects if text is using language specific spelling and or terms
    '''
    if args.lang == 'UK':
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

def detect_subtoken_group(tokens, reranker):
    spell, token_len = 0, 0
    for token in tokens:
        if british_spelling_exists(token):
            spell += 1
            swapped_spelling = get_american_spelling(token)
            
            swapped_len = len(reranker.tokenizer(swapped_spelling)['attention_mask'])
            original_len = len(reranker.tokenizer(token)['attention_mask'])
            
            if swapped_len != original_len:
                token_len = 1
            else:
                token_len = 0
    return spell, token_len

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

# CREATE CROSS SETS
# # Create Set 1
xset1 = topics.copy()
if args.lang == 'UK':
    xset1 = xset1[(xset1['US'] > 0) & (xset1['UK'] == 0)]
elif args.lang == 'US':
    xset1 = xset1[(xset1['UK'] > 0) & (xset1['US'] == 0)]

spellings, terms = [], []
for query in xset1['query']:
    spell_temp, term_temp = detect_terms_spell_cross(query)
    spellings.append(spell_temp)
    terms.append(term_temp)
xset1['spellings'] = spellings
xset1['terms'] = terms

# # Create Set 2
xset2 = xset1.copy()
xset2 = xset2[(xset2['terms'] == 0) & (xset2['spellings'] > 0)]

# # Create Set 3
xset3 = xset1.copy()
xset3['query'] = xset3['query'].apply(convSame)

# # Create Set 4
xset4 = xset2.copy()
xset4['query'] = xset4['query'].apply(convSame)

if args.set == 'set1':
    SET = set1
    xSET = xset1
    normSET = set3
    normSETstr = "set3"
elif args.set == 'set2':
    SET = set2
    xSET = xset2
    normSET = set4
    normSETstr = "set4"
elif args.set == 'set3':
    SET = set3
    xSET = xset3
elif args.set == 'set4':
    SET = set4
    xSET = xset4

# read files and run experiments
print(f'Loading {args.model} Experiments Results from {args.pipeline_one_results}')
output_base = pt.io.read_results(args.pipeline_one_results)
print(f'Loading {args.model} Experiments Results from {args.pipeline_two_results}')
output_diff = pt.io.read_results(args.pipeline_two_results)

print('Running T-tests...')
# get per query mmr and r for each model output
output_base_per_query = pt.Experiment([output_base], perquery=True, topics=SET, qrels=dataset.get_qrels().merge(SET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['original pipeline'])
# filter individual recall and mrr in two lists
output_diff_per_query = pt.Experiment([output_diff], perquery=True, topics=normSET, qrels=dataset.get_qrels().merge(normSET, on=['qid']), eval_metrics=[MRR@10, R@1000], names=['diff pipeline'])
mrr_base = output_base_per_query[output_base_per_query['measure'] == 'RR@10']['value'].to_numpy()
recall_base = output_base_per_query[output_base_per_query['measure'] == 'R@1000']['value'].to_numpy()

mrr_diff = output_diff_per_query[output_diff_per_query['measure'] == 'RR@10']['value'].to_numpy()
recall_diff = output_diff_per_query[output_diff_per_query['measure'] == 'R@1000']['value'].to_numpy()
# Run Wilcoxon tests
mrr_wilcoxon = wilcoxon(mrr_base, mrr_diff) # Wilcoxon
print('MRR Wilcoxon')
print(mrr_wilcoxon)
recall_wilcoxon = wilcoxon(recall_base, recall_diff) #Wilcoxon
print('Recall Wilcoxon')
print(recall_wilcoxon)
# Run TOST
mrr_tost = ttost_paired(mrr_base, mrr_diff, -0.05, 0.05) # TOST
print('MRR TOST')
print(mrr_tost)
recall_tost = ttost_paired(recall_base, recall_diff, -0.05, 0.05) # TOST
print('Recall TOST')
print(recall_tost)
