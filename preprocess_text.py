# preprocess_text
import nltk
import os
import codecs
from collections import Counter
import re

from collections import Counter
import re
import nltk
import pickle

url_pattern = '(?:(?:https?:\/\/|www\.)\w+[\.\w]*\.\w+)|(?:\w+\.(?:com|dk))|(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url_re = re.compile(url_pattern)
url_sub = ' __url__ '
digit_pattern = r"\d(?:[\.,\d]*\d+)|\d{2,}(?:[\.,\d]*\d+)?" # matching >1 figure digits
replace_regex = [(url_sub,url_re),(' __digit__ ',re.compile(digit_pattern))]

import numpy as np
def filter_words(words,stopwords):
    # filter non signs
    # words = [word for word in words if isalphanum(word)]
    # filter stopwords
    words = [word for word in words if word not in stopwords]
    return words
def isalphanum(string):
    if not string.isalnum():
        # find word with - between
        pattern = r'[a-zAZøæåÆØÅ]+\-[a-zAZøæåÆØÅ]*|__[a-zAZøæåÆØÅ]+_*'
        search = re.findall(pattern,string)
        if len(search)>0:
            return True
        else:
            return False
    return True
def mark_sent(string,sent_sep = ' __sent__ '):
	return sent_sep.join(nltk.tokenize.sent_tokenize(string))

tweet_tokenize = nltk.tokenize.casual.TweetTokenizer()
def process_documents(text,stopwords,tokenizer = 'tweet' , sentence = False,pre_clean=lambda x: x,post_clean=lambda x: x):#= nltk.tokenize.casual.TweetTokenizer().tokenize):
    text = ''.join(text)#.copy(
    text = pre_clean(text)
    if tokenizer == 'tweet':
        tokenizer = tweet_tokenize.tokenize
    else:
        tokenizer = nltk.tokenize.word_tokenize
    # replace known patterns.
#    for sub_pattern,regex in replace_regex:
#        text = regex.sub(sub_pattern,text)
    if sentence:
        text = mark_sent(text)
    text = tokenizer(text)#nltk.tokenize.word_tokenize(text)
    # lower
    text = [i.lower() for i in text]
    # remove stopwords
    text = filter_words(text,stopwords)
    text = post_clean(text)
    return text

from nltk.corpus import stopwords
stopwords = set(nltk.corpus.stopwords.words('danish'))#|set(open('danish_stopwords.txt','r').read().split('\n'))|set(['__digit__']) # danish


### Function for creating sparse matrices

def to_dense(corpus,vocab_size):
    X = np.zeros((len(corpus),vocab_size),dtype=np.int32)
    for num in range(len(corpus)):
        bow = corpus[num]
        for w,count in bow.items():
            try:
                X[num][w]=count
            except:
                pass
    return X
import scipy.sparse as sp
def to_sparse(corpus,vocab_size):
    X = sp.dok_matrix((len(corpus),vocab_size), dtype=np.int32)
    for num in range(len(corpus)):
        bow = corpus[num]
        for w,count in bow.items():
            X[num,w]=count
    print(X.shape)
    X = X.tocsr()
    print(X.shape)
    return X

def to_index(text,d):
    return [d[i] for i in text if i in d]
def to_bow(texts,d,ngram=3,sparse = True):
    vocab_size = len(d)
    w2id = {w:num for num,w in enumerate(d)}
    bows = [Counter(to_index(get_ngram(text,ngram),w2id)) for text in texts]
    if sparse:
        return to_sparse(bows,vocab_size)
    else:
        return bows
def get_ngram(doc,n=2):
    grams = doc.copy()

    for gram in range(2,n+1):
        grams+=['_'.join(doc[i:i+gram]) for i in range(len(doc)+1-gram)]
    return grams

def make_index(texts,ngram=False,cutoff=5,max_words=100000,ngram_only=False):
    c = Counter()
    if ngram==False:
        for doc in texts:
            for w in doc:
                c[w]+=1
    else:
        for doc in texts:
            grams = get_ngram(doc,ngram)
            for w in grams:
                if ngram_only:
                    if w.count('_')!=ngram:
                        continue
                c[w]+=1
    return [w for w,count in c.most_common(max_words) if count>=cutoff]

def process_docs(texts,tokenizer=nltk.word_tokenize,stopwords=set(),**kwargs):
    "Function for tokenizing and simple cleaning of docs"
    docs = [process_documents(str(text),stopwords=stopwords,tokenizer=tokenizer,**kwargs) for text in texts]
    return docs

###
def make_bows(docs,ngram=3,kwargs={}):
    Index = make_index(docs,ngram=ngram,**kwargs)
    bows = to_bow(docs,Index,ngram=ngram)
    return Index,bows


class ngrams():
  def __init__(self,phrasers=[]):
    self.phrasers = phrasers
  def get_phrase(self,sent,n=False):
    if n!=False:
      phrasers = self.phrasers[0:n-1]
    else:
      phrasers = self.phrasers
    for phraser in phrasers:
      sent = phraser[sent]
    return sent
from gensim.models.phrases import Phrases
def make_phraser(docs,phrase_length=3):
    """Input tokenized docs. Estimates collocations using the gensim.Phrases
    Returns ngram class. Use .get_phrase(sent,n) to transform tokenized word lists to phrases of length n"""
    models = []
    for i in range(phrase_length-1):
        model = Phrases(docs)
        docs = [model[sent] for sent in docs]
        docs = [[i.replace(' ','_') for i in doc] for doc in docs]
        models.append(model)
    models = [model.freeze() for model in models]
    final_phraser = ngrams(models)
    return final_phraser
