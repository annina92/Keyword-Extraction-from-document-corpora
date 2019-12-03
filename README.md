Ideas for keyword extraction:
- tfidf
- ngrams tfidf for tokenization optimization
- text rank
- Latent Dirichlet Allocation (LDA)
- Latent semantic analysis
- Frequent meta-words should be removed from keywords ("paper", "research", ...) (use only term frequency??)
- NEED OF UNSUPERVISED LEARNING
- SVM for supervised learning?

Ideas for clustering:
- ???

Ideas for further work:
- once defined keywords for research area, define words for each document (tfidf at doc level should be enough)



Content:

- get_iris_abs.py: extraction of abstract given prof names.

- utils_data_structures: functions to create several data structures to handle document processing
    - id -> abstract
    - id -> research group
    - id -> authors
    - authors -> research group
    - authors -> abstractId
    - research group -> authors
    - research group -> abstractId

- doc_pre-processing: function to pre process abstracts (tokenization, stopword removal) and tfidf functions (uni, bi and trigrams)
    - pos_docs: docs list with pos_tags (saved in pos_tag_docs.json)
    - tokenization: tokenization using NER to find common words that should be tokenized together. Numbers removal
    - text_cleaning: removal of special or useless characters, lemmatization(TODO)
    - tfidf: tfidf of pre-processed docs, extraction of most important words for each research group
    - tfidf_ngrams: tfidf using bigrams and trigrams, to find important groups of words inside docs, for a better keyword extraction

Basic idea: identify most important words (made of 1, 2 and 3 tokens) and use them as keywords.

Vocabulary size: 12184

N° docs: 1407

N° categories: 19



Used Libraries:
- NLTK
- SpaCy
- numpy
- sklearn
- pandas
