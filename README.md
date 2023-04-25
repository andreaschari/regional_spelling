# On the Effects of Regional Spelling Conventions in Retrieval Models (SIGIR 2023)

One advantage of neural language models is that they are meant to
generalise well in situations of synonymity i.e. where two words
have similar or identical meanings. In this paper, we investigate and
quantify how well various ranking models perform in a clear-cut
case of synonymity: when words are simply expressed in different
surface forms due to regional differences in spelling conventions
(e.g., color vs colour). We explore the prevalence of American and
British English spelling conventions in datasets used for the pre-
training, training and evaluation of neural retrieval methods, and
find that American spelling conventions are far more prevalent. De-
spite these biases in the training data, we find that retrieval models
are in general robust in this case of synonymity generalisation. We
also explore document normalisation as a possible performance
improvement strategy and observe that all models are affected by
document normalisation. While they all experience a drop in per-
formance when normalised to a different spelling convention than
that of the query, we observe varied behaviour when the docu-
ment is normalised to share the query spelling convention with the
lexical models showing improvements, while the dense retrievers
remain unaffected and the neural re-rankers exhibit contradictory
behaviour.

This README contains the source code for all of our experiments report in the paper

# Dependencies

## Pyterrier
`pip install python-terrier`

## Pyterrier-_DR
Follow instructions here: https://github.com/terrierteam/pyterrier_dr

## Pyterrier_splade
Follow instructions here: https://github.com/cmacdonald/pyt_splade

## pyterrier_colbert
Follow instructions here: https://github.com/terrierteam/pyterrier_colbert#installation


## spaCy
`pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm`


## Other dependencies
` pip install numpy pandas breame scipy statsmodels`


# How-to
## Indexing
Before running any experiments we need to generate the appropriate index using Pyterrier:

`python dialect_experiments_indexing.py -m ['tct-colbert', 'tasb', 'splade', 'colbert', 'electra', 'monot5'] -idx_dir [path]`

This will load an instance of the model specified and generate the three indices used in the experiments:
1. default msmarco-passage
2. msmarco-passage normalised to American Spelling (ConvUS)
3. msmarco-passage normalised to British Spelling (ConvUK)

## Running Re-ranking experiments

To run the reranking model experiments please use `dialect_experiments.py`.

Usage: `python dialect_experiments.py -l [UK, US] -m ['TCT_ColBERT', 'electra', 'monoT5', 'bm25'] -s ['set1', 'set2', 'set3', 'set4'] --idx_dir [index path] --out_dir [path to store re-ranking results]`

This will generate the query sets and run a re-ranking pipeline using given input pyterrier index on specified query set (1,2,3,4) and report back first-stage (BM25) & second-stage (-m) model MMR@10 and R@1000 metrics for this query set.

## Running Dense Retrival experiments

To run the dense retrievsl model experiments please use `dialect_experiments_dense.py`.

Usage: `python dialect_experiments_dense.py -l [UK, US] -m []'TCT-ColBERT', 'TasB', 'splade', 'colbert'] -s ['set1', 'set2', 'set3', 'set4'] --idx_dir [index path] --out_dir [path to store dense retrieval results]`

This will generate the query sets and run a dense retrival pipeline using given input pyterrier index on specified query set (1,2,3,4) and report back Dense Retrival MRR@10 and R@1000 metrics for this query set.

## Running ABNIRML-style experiments

To run the ABNIRML-style experiments please use `dialect_experiments_abnirml.py`.

Usage: `python dialect_experiments_abnirml.py -l [UK, US] -m ['TCT_ColBERT', 'TasB', 'splade', 'colbert', 'electra', 'monoT5', 'bm25'] -s ['set1', 'set2', 'set3', 'set4'] -comp [diff, nonorm]`

This will generate the query sets and run the ABNIRML-style experiments for model (-m)  on specified query set (1,2,3,4) and report back number of query-document pairs where models prefers same normalisation vs different spelling normalisation/ no normalisation (-comp).

## Running Statistical tests

To run the statistical tests for dense or re-ranking output use `dialect_experiments_statistical_tests.py`

Usage: `python dialect_experiments_statistical_tests.py -l [UK,US] -m ['TCT_ColBERT', 'TasB', 'splade', 'colbert', 'electra', 'monoT5', 'bm25'] -s ['set1','set2','set3','set4'] -pipeline_one_dir (pipeline results path) -pipeline_two_dir (pipeline results path)`

This will load the results of pipeline one and two and will run Wilcoxon rank-signed and TOST on the MRR and Recall scores
