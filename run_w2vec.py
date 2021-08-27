import nltk
import gensim
nltk.download('punkt')
from gensim.models.phrases import Phrases
import logging

from gensim.models.word2vec import Word2Vec
from collections import Counter

def replace_phrases(text,phrases):
    for i in phrases:
        text = text.replace(i,'_'.join(i.split()))
    return text
def run_w2vec(texts,emb_size=64,known_phrases=[], return_phrased=False,return_counter=False,kwargs={}):
    if type(texts[0])==str:
        docs = [nltk.word_tokenize(replace_phrases(i.lower(),known_phrases)) for i in texts]
        print('Tokenizing...')
    else:
        docs = texts
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
    new_docs = docs+phrase_docs_bi+phrase_docs
    for i in new_docs:
        count+=len(i)
        for w in i:
            ws[w]+=1
    print('%d unique words in corpus and %d count'%(len(ws),count))
    ws = dict(ws.most_common(max_words))
    min_words = 2000000
    iter_ = min_words//count
    iter_ = max(iter_,5)
    w2v = Word2Vec(vector_size=size,workers=workers,negative=10,window=window,min_count=min_count) # max_final_vocab = max_words
    w2v.build_vocab_from_freq(ws)
    print('Getting ready to train')
    #print(w2v.corpus_count)
    new_docs = docs+phrase_docs_bi+phrase_docs
    import random
    random.shuffle(new_docs)
    w2v.train(new_docs,total_words = count,epochs=iter_)

    if return_phrased:
        if return_counter:
            return w2v,docs,phrase_docs_bi,phrase_docs,ws
        else:
             w2v,docs,phrase_docs_bi,phrase_docs
    if return_counter:
        w2v,ws
    return w2v
