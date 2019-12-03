import json
import spacy
import pandas as pd
import numpy as np
import re
import nltk

from nltk.tokenize import MWETokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

spacy_nlp = spacy.load('en_core_web_sm')
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

tokenized_docs = {}

def dummy(doc):
    return doc

vectorizer_trigrams = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    use_idf=True,
    ngram_range=(3,3),
    sublinear_tf =True)  


punctuation_list = ["%","","$","&","£","\\","#","|","!",".",",", "[", "]", "(", ")",":",";","©", "`","``", "”", "\'", "\"","\{", "\}"]

with open("./data_structures/dict_id_abstract.json", 'r+') as myfile:
    data=myfile.read()
dict_id_abstract = json.loads(data)

ids = list(dict_id_abstract.keys())

print(len(ids))

def statistics():
    sum =0
    with open("./data_structures/dict_research_group_ids.json", 'r+') as myfile:
        data=myfile.read()
    dict_research_group_ids = json.loads(data)    

    for rs in dict_research_group_ids.keys():
        print(rs,": ",len(dict_research_group_ids[rs]))
        sum += len(dict_research_group_ids[rs])

    print("sum", sum)
    print("N° of docs: ",len(ids))

def tf():
    with open("./data_structures/for_test_lemmas.json", 'r+') as myfile:
        data=myfile.read()
    dict_docs = json.loads(data)
    ids = list(dict_docs.keys())    
    values = []

    dict_word_frequencies = {}
    for doc in dict_docs:
        document = dict_docs[doc]
        for word in document:
            values.append(tuple((word, 1)))

    d = {x:0 for x, _ in values} 
    for name, num in values: d[name] += num 
    
    # using map 
    result = list(map(tuple, d.items())) 
    result.sort(key=lambda tup: tup[1])        
    print(result)


def tfIdf():

    with open("./data_structures/for_test_lemmas.json", 'r+') as myfile:
        data=myfile.read()
    dict_docs = json.loads(data)
    ids = list(dict_docs.keys())

    with open("./data_structures/dict_research_group_ids.json", 'r+') as myfile:
        data=myfile.read()
    dict_research_group_ids = json.loads(data)

    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        use_idf=True,
        sublinear_tf =True)  

    list_docs = []
    for _id in ids:
        list_docs.append(dict_docs[_id])
    tfidf_vectors = vectorizer.fit_transform(list_docs)
    terms = vectorizer.get_feature_names()
    df = pd.DataFrame(tfidf_vectors.toarray(), columns= terms)
    print(df)
    m = tfidf_vectors.toarray()

    for rs in dict_research_group_ids.keys():
        print("Research group: ",rs)
        #index_tfidf contains indices of the rows to use in the matrix
        index_tfidf = []
        for _id in dict_research_group_ids[rs]:
            index_tfidf.append(ids.index(_id))

        matrix = np.zeros(shape=(len(index_tfidf), m.shape[1]))
        print(matrix.shape)

        for i in range(len(index_tfidf)):
            matrix[i] = m[index_tfidf[i], :]

        #matrix contains tfidf values of the documents of a single research group (truncated tfidf matrix)

        ######################
        #TEST 1: TAKE HIGHEST VALUE IN DECREASING ORDER OF SUM OF COLUMNS VALUES
        ######################
        print("TEST 1")

        array_feature_values_sum = np.sum(matrix, axis = 0)

#        for i in range(m.shape[1]):
#            if array_feature_values_sum[i]>0:
#                print(terms[i], ": ", array_feature_values_sum[i])

        result = np.where(array_feature_values_sum == np.amax(array_feature_values_sum))
        
        print('Word :', terms[result[0][0]])
        print('Score :', array_feature_values_sum[result[0]])
        

        ordered_indices = np.argpartition(-array_feature_values_sum,range(10))
        print (ordered_indices)
        for i in range (10):
            print (terms[ordered_indices[i]])

        print("\n\n ")


        ######################
        #TEST 2: TAKE MEAN AND ORDER BY DECREASING SCORES
        ######################
        print("TEST 2")
        array_feature_values_sum = np.sum(matrix, axis = 0)
        array_means = np.zeros(shape=len(array_feature_values_sum))

        for i in range(len(array_feature_values_sum)):
            array_means[i] = array_feature_values_sum[i]/matrix.shape[1]
        
        result = np.where(array_means == np.amax(array_means))
        
        print('Word :', terms[result[0][0]])
        print('Score :', array_means[result[0]])
        

        ordered_indices = np.argpartition(-array_means,range(10))
        print (ordered_indices)
        for i in range (10):
            print (terms[ordered_indices[i]])

        print("\n\n ")      
            
    
    #########################
    #META WORDS
    #########################
    print("META WORDS")
    common_words = np.sum(m, axis = 0)

#        for i in range(m.shape[1]):
#            if array_feature_values_sum[i]>0:
#                print(terms[i], ": ", array_feature_values_sum[i])

    result = np.where(common_words == np.amax(common_words))
    
    print('Word :', terms[result[0][0]])
    print('Score :', common_words[result[0]])
    

    ordered_indices = np.argpartition(-common_words,range(20))
    print (ordered_indices)
    for i in range (20):
        print (terms[ordered_indices[i]])


    #return tfidf_vectors

def tfidf_ngrams(ngrams):

    vectorizer_trigrams = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        use_idf=True,
        ngram_range=(ngrams,ngrams),
        sublinear_tf =True)  

    with open("./data_structures/for_test.json", 'r+') as myfile:
        data=myfile.read()
    dict_docs = json.loads(data)

    dict_docs_trigrams= {}

    ids = list(dict_docs.keys())

    list_docs = []
    for _id in ids:
        list_docs.append(dict_docs[_id])
    tfidf_vectors = vectorizer_trigrams.fit_transform(list_docs)

    terms = vectorizer_trigrams.get_feature_names()
    m = tfidf_vectors.toarray()
    print("shape: ",m.shape)
    print("type: ",type(m))
    print("itemsize: ",m.itemsize)
    for i in range(0,m.shape[0]):
        dict_docs_trigrams[ids[i]] = []
        for j in range(0,m.shape[1]):
         
            if m[i,j] >0:
                dict_docs_trigrams[ids[i]].append(tuple((terms[j], m[i,j])))
    print(dict_docs_trigrams)    
        
    df = pd.DataFrame(tfidf_vectors.toarray(), columns= terms)
    #print(df)

def word_statistics(abstract_list_id, dict_abstract):
    word_count = 0
    _max=0
    _min =100000
    for _id in abstract_list_id:
        abstract = dict_abstract[_id]
        abstract_lenght = len(abstract.split(" "))
        word_count+= abstract_lenght
        if abstract_lenght > _max :
            _max = abstract_lenght
        if abstract_lenght < _min:
            _min = abstract_lenght

    print("word count: ")
    print(word_count/len(abstract_list_id))
    print("max lenght: ")
    print(_max)
    print("min lenght: ")
    print(_min)

def pos_docs(docs):
    result = {}
    for doc in docs:
        text = word_tokenize(docs[doc]) 
        result[doc]=nltk.pos_tag(text)
    f = open("./data_structures/pos_tag_docs.json", "w+")
    json_data = json.dumps(result)
    f.write(json_data)
    f.close() 


def tokenization(docs):
    documents = {}

    for doc in docs:
        document_plain= docs[doc]
        document_plain = document_plain.replace("/", "").replace("-", " ")
        #re.sub(r'\([^)]*\)', '', document_plain)
        re.sub(r'\([0-9]*\)', '', document_plain)

        relevant_words = []
        mwetokenizer = MWETokenizer()
        document_ner = spacy_nlp(document_plain)

        for element in document_ner.ents:
        # don't consider numbers
            if element.label_ not in "CARDINAL":
                relevant_words.append(element)

        #for each relevant word, if whitespace is present, create a single token with all the words
        for word in relevant_words:
            token = str(word).split()
            if len(token)>1:
                move_data=[]
                for element in token:
                    move_data.append(element)
                tup = tuple(move_data)
                mwetokenizer.add_mwe(tup)
        
        document_tokenized = word_tokenize(document_plain)
        document_retokenized = mwetokenizer.tokenize(document_tokenized)
        
        documents[doc] = document_retokenized
    return documents

def text_cleaning(docs):
    documents = {}
    documents_without_verbs ={}

    for doc in docs:
        words = []
        for word in docs[doc]:
            word = word.lower()
            word.replace("|","").replace("\\","").replace("!","").replace("\"","").replace("£","").replace("$","").replace("%","").replace("&","").replace("","").replace("(","").replace(")","").replace("=","").replace("?","").replace("^","").replace(",","").replace(".","").replace("@","").replace("#","").replace("\'","").replace("~", "")
            if (word not in stopwords.words('english')) and (word not in punctuation_list) and (len(word)>1)and (not word.isdigit())and ("//" not in word):
                #TODO LEMMATIZATION        
                words.append(word)
        documents[doc] = words

        #bigram_fd = nltk.FreqDist(nltk.bigrams(words))
        #words è la lista di parole da modificare
        #document_tagged = nltk.pos_tag(words)
        #PRENDO SOSTANTIVI E AGGETTIVI SENZA VERBI E FACCIO TFIDF CON QUELLI
        #document_tagged = [(x,y) for (x,y) in document_tagged if (y in ('VB', 'NN', 'NNS', 'NNP', 'NNPS','VBD', "VBG", "VBN", "VBP", "VBZ"))or ("_" in x) ]

        #print (document_tagged)

    f = open("./data_structures/for_test_noverbs.json", "w+")
    json_data = json.dumps(documents_without_verbs)
    f.write(json_data)
    f.close() 

    f = open("./data_structures/for_test.json", "w+")
    json_data = json.dumps(documents)
    f.write(json_data)
    f.close() 
    
    return documents

def lemmatization(docs):
    documents = {}
    for doc in docs:
        words = []
        document_tagged = nltk.pos_tag(docs[doc])
        #PRENDO SOSTANTIVI E AGGETTIVI SENZA VERBI E FACCIO TFIDF CON QUELLI
        document_tagged = [(x,y) for (x,y) in document_tagged if (y in ("JJ", 'JJR','JJS','VB', 'NN', 'NNS', 'NNP', 'NNPS','VBD', "VBG", "VBN", "VBP", "VBZ"))or ("_" in x) ]

        for word, tag in document_tagged:
            if tag.startswith("NN"):
                lemmatized_word = lemmatizer.lemmatize(word, pos="n")
                words.append(lemmatized_word)
            if tag.startswith("VB"):
                lemmatized_word = lemmatizer.lemmatize(word, pos="v")
                words.append(lemmatized_word)
            if tag.startswith("JJ"):
                lemmatized_word = lemmatizer.lemmatize(word, pos="a")
                words.append(lemmatized_word) 
            else:
                print("left out: ", word)
        documents[doc] = words

    f = open("./data_structures/for_test_lemmas.json", "w+")
    json_data = json.dumps(documents)
    f.write(json_data)
    f.close() 
    print (documents)
  


def LSA(tfidf_matrix):
    lsa = TruncatedSVD(n_components=5, n_iter=20)
#    lsa.fit(tfidf_vectors)

    for i, comp in enumerate(lsa.components_):
 #       termsInComp=zip(terms, comp)
 #       sortedTerms = sorted(termsInComp, key=lambda x:x[1], reverse=True) [:10]
        print("concept", i)
#        for term in sortedTerms:
#            print (term[0])
        print("  ")


#tokenized_words = tokenization(dict_id_abstract)
#processed_words = text_cleaning(tokenized_words)
#lemmatization(processed_words)
#tfidf_ngrams(3)
#tfidf_ngrams(2)
tfIdf()

#word_statistics(ids, dict_id_abstract)