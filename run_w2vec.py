
import gensim
from gensim.models.phrases import Phrases
import logging

from gensim.models.word2vec import Word2Vec
from collections import Counter
import random
import nltk
nltk.download('punkt')
def calculate_w2vec_size(count):
  count = count/1000000
  base_size = 64
  return min(max(32,base_size*(int(np.log2(count)-1))),256)

def replace_phrases(text,phrases):
    for i in phrases:
        text = text.replace(i,'_'.join(i.split()))
    return text
def run_w2vec(texts,emb_size=False,known_phrases=[], return_phrased=False,return_counter=False,kwargs={},phrases=True):
    if type(texts[0])==str:


        docs = [nltk.word_tokenize(replace_phrases(i.lower(),known_phrases)) for i in texts]
        print('Tokenizing...')
    else:
        docs = texts
    if phrases:
        print('Locating collocations...')
        phrase_model_bi = Phrases(docs)

        phrase_docs_bi = [phrase_model_bi[sent] for sent in docs]
        phrase_model = Phrases(phrase_docs_bi)
        phrase_docs = [phrase_model[sent] for sent in phrase_docs_bi]
    # train w2v
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    ############
    ## missing
    #####load pretrained and just retrain
    ###############
    ### Define parameters for the model ###
    max_words = 100000
    size=emb_size # Size of Embedding.
    workers=4 # Number CPU Cores to use for training in Parallel.
    iter_= 50 # Depending on the size of your data you might want to run through your data more than once.
    window=6 # How much Context
    min_count=5 # Number of Occurrences to be kept in the Vocabulary
    count = 0
    ws = Counter()
    if not emb_size:
        count = 0
        for doc in docs:
            count+=len(doc)
        emb_size = calculate_w2vec_size(count)
    if phrases:
        new_docs = docs+phrase_docs_bi+phrase_docs
    else:
        new_docs = docs
    random.shuffle(new_docs)
    for i in new_docs:
        count+=len(i)
        for w in i:
            ws[w]+=1
    print('%d unique words in corpus and %d count'%(len(ws),count))
    ws = dict(ws.most_common(max_words))
    min_words = 2000000
    iter_ = min_words//count
    iter_ = max(iter_,3)
    w2v = Word2Vec(vector_size=size,workers=workers,negative=10,window=window,min_count=min_count) # max_final_vocab = max_words
    w2v.build_vocab_from_freq(ws)
    print('Getting ready to train')
    #print(w2v.corpus_count)
    w2v.train(new_docs,total_words = count,epochs=iter_)

    if return_phrased:
        if return_counter:
            return w2v,docs,phrase_docs_bi,phrase_docs,ws
        else:
             w2v,docs,phrase_docs_bi,phrase_docs
    if return_counter:
        w2v,ws
    return w2v

def process_document(text,tokenizer = nltk.tokenize.word_tokenize):
    text = ''.join(text)
    text = tokenizer(text)
    # lower
    text = [i.lower() for i in text]
    return text
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
    X = X.tocsr()
    return X

def to_index(text,d):
    return [d[i] if i in d else 0 for i in text]
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

def make_index(texts,ngram=False,cutoff=5,max_words=100000):
    c = Counter()
    if ngram==False:
        for doc in texts:
            for w in doc:
                c[w]+=1
    else:
        for doc in texts:
            grams = get_ngram(doc,ngram)
            for w in grams:
                c[w]+=1
    return ['__out__']+[w for w,count in c.most_common(max_words) if count>=cutoff]
def get_dtm(texts):
    docs = [process_document(text) for text in texts]
def dtm_tfidf(dtm):
    # Document frequency
    df = np.asarray(dtm.sign().sum(axis=0))[0,:]
    # Inverse document frequency
    idf =-np.log(df/dtm.shape[0])
    # Combined term frequence and inverse document frequency
    tfidf = dtm.multiply(idf)
    return tfidf.tocsr()
from scipy.stats import entropy
def run_entropy(bow,path):
    tfidf = dtm_tfidf(bow)
    ents = []
    for i in tqdm.tqdm(np.arange(bow.shape[1])):
        idx = (np.asarray(bow[:,i].sum(axis=1)).flatten()>0)*1
        s = np.asarray(tfidf[idx].sum(axis=0)).flatten()
        ent = entropy(s[s>0])
        ents.append(ent)
    ents = np.array(ents)
    pickle.dump(ents,open(path+'entropy_w','wb'))
