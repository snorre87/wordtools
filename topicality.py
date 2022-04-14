
import gensim
from gensim.models.phrases import Phrases
import logging
import nltk
nltk.download('punkt')
from gensim.models.word2vec import Word2Vec
from collections import Counter
import random
import scipy.sparse as sp
import sklearn.cluster
import matplotlib.pyplot as plt

class Topicality():
    def __init__(self,texts,tokenizer=nltk.word_tokenize,ngram=3,min_count=5,max_words=100000):
        self.texts = texts
        print('tokenizing')
        self.docs = [[i.lower() for i in nltk.word_tokenize(text)] for text in texts]
        print('making index')
        self.Index = make_index(self.docs,ngram=ngram,cutoff=min_count,max_words=max_words)
        self.w2i = {w:num for num,w in enumerate(self.Index)}
        print('Transforming to DocTermMatrix')
        self.dtm = to_bow(self.docs,self.w2i,ngram=ngram)
        print('Running Entropy/Topicality detector')
        self.entropy_w = run_entropy(self.dtm)
        print('Running w2vec')
        self.w2v = run_w2vec(self.docs)
        ## Normalize word embeddings
        self.w2v_m,self.w2w_std = model.wv.vectors.mean(axis=0),model.wv.vectors.std(axis=0)

    def get_word_vector(self,w,normed=True,lower=True):

        """Get Word2Vec vector from input.
        Input either a word or a sequence of words.
        example: 'hello_world' or 'hello world'
         """
        if lower:
            w = w.lower()
        if w in w2vec_w2id:
            vec = model.wv.get_vector(w)
        else:
            ws = re.split('_| ',w)
            start,end = 0,len(ws)
            n = len(ws)
            vec = []
            while start<n and start<end:
                seq = '_'.join(ws[start:end])
                if seq in w2vec_w2id:
                    vec.append(model.wv.get_vector(seq))
                    start = end
                    end = len(ws)
                else:
                    end-=1
            if len(vec)==0:
                return self.w2v_m
            vec = np.array(vec)
            vec = vec.mean(axis=0)
            if normed:
                vec = (vec-self.w2v_m)/self.w2v_std
        return vec
    ## Visualization
    def visualize_wordspace(self,words,topn=250,w2group=False,cmap=plt.cm.gist_ncar,return_2d = False,clustering = sklearn.cluster.KMeans(n_clusters=20),reducer='umap'):
        """Visualize words in 2d space with labels.
        Groups are defined by the w2group, input > dict
        topn> int, keeping only the n most occurring words, normalzed by     count in each group.
        Choose between clustering algorithms: 'kmeans'
        Choose 2d reducer: 'umap','tsne','pca's"""
        dtm = self.dtm
        w2id = self.w2i
        words = list(words)
        dat = []
        mat = []
        for w in words:
            if not w in w2id:
                continue
            idx = w2id[w]
            d = {'w':w,'count':dtm[:,idx].sum()}
            mat.append(get_word_vector(w))
            dat.append(d)
        df = pd.DataFrame(dat)
        mat = np.array(mat)

        embedding = get_2d_embedding(mat,reducer=reducer)
        embedding = reducer.fit_transform(mat)
        df['x_w'] = embedding[:,0]
        df['y_w'] = embedding[:,1]
        if not w2group:
            # Doing agglomerative clustering on words.
            #partition = dict(df[['w','category']].values)
            # Consider clustering
            vectors = []
            for w in words:
                vectors.append(get_word_vector(w))
            labels = clustering.fit_predict(np.array(vectors))
            w2group = {words[i]:labels[i] for i in range(len(labels))}
        #w2group = {w:num for num,w in enumerate(words)}
        groups = set(w2group.values())
        # gist rainbow or tab20 or spectral gist_ncar
        g2color = {i:cmap(num/len(groups)) for num,i in enumerate(groups)}
        df['color'] = [g2color[w2group[w]] for w in df['w'].values]
        df['category'] = df.w.apply(lambda x: w2group[x])
        #import re
        #df['color'] = df.color.apply(lambda x: tuple(map(float,re.findall('[0-9.]+',x))) if type(x)==str else x)
        def relative_count(groupdf):
            # tradeoff between
            groupdf['rel_count'] = groupdf['count']/groupdf['count'].median()+groupdf['count']/groupdf['count'].max()
            return groupdf
        df = df.drop_duplicates(['w'],keep='first')
        df = df.groupby('category').apply(relative_count)
        fig,ax = plt.subplots()
        fig.set_size_inches(18,12)
        for group,groupdf in df.groupby('category'):
            color = groupdf.color.values[0]
            plt.scatter(groupdf.x_w,groupdf.y_w,color=color)
        pos_umap = {n:tuple(df.iloc[num][['x_w','y_w']]) for num,n in enumerate(df.w.values)}
        if topn:
            df = df.sort_values('rel_count').tail(topn)
        texts = []
        for i in df.w.values:
          texts.append(plt.text(pos_umap[i][0],pos_umap[i][1],i))
        print('Adjusting texts: May take some time...')
        from adjustText import adjust_text
        adjust_text(texts)
        #nx.draw_networkx_labels(nx.subgraph(g,top), pos);
        ax.set_axis_off()
        #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        if return_2d:
            return df,fig,ax
        return fig,ax


    def visualize_topical_words(self,nwords=1000,topn_visible=300,add_k_neighbors=0,norm_window = 15,return_data=False,freq_cut=0.5,topicality_cut_quantile=0.25
    ,clustering = sklearn.cluster.KMeans(n_clusters=20),reducer='umap',remove_duplicate_phrases=False):
        """Function for visualizing topical words in w2vec reduced 2d space. Topical words are based on the relative (to neighbors of similar occurrence of each word) entropy of the weighted and tfidf normalized co-occurrence network.
        Choose number of words to include (nwords), and how many should be visible (topn_visible). Choose to add neighbors using w2vec for attenuating clusters.
        Return fig,ax, and optionally a dataframe consisting of 2d coordinates and cluster categories.
        Choose between clustering algorithms: 'agglomerative','kmeans','networkbased'
        remove_duplicate_phrases will remove multiwords that share stems, keeping only the most topical.
        """
        ents = self.entropy_w
        w = norm_window
        ents_normed = np.array([ents[i]/ents[max(i-w,0):min(i+w,len(ents))].mean() for i in range(len(ents))])
        topicality_cut = np.quantile(ents_normed,topicality_cut_quantile)
        sort = ents_normed.argsort()
        bow = self.dtm
        c = np.asarray(bow.sum(axis=0)).flatten()
        ps = c/bow.shape[0]
        Index = self.Index
        nodes = set([Index[i] for i in sort[0:nwords] if ps[nwords]<freq_cut and ents_normed[i]<topicality_cut])
        if add_k_neighbors>0:
            # locate meaningful cut
            sort = ents_normed.argsort()
            k = 250
            kn = 10
            scores = []
            for i in tqdm.tqdm(sort[:k]):
                if ps[i]>freq_cut:
                    continue
                w = Index[i]
                vec = get_word_vector(w)
                ns = model.wv.most_similar_cosmul(vec,topn=kn)
                for n2,score in ns:
                    if not n2== w:
                        scores.append(score)
            cut = np.median(scores)
            kn = add_k_neighbors
            for w in nodes:
                vec = get_word_vector(w)
                ns = model.wv.most_similar_cosmul(vec)
                count = 0
                for n2,score in ns:

                    if score<cut:
                        continue
                    if n2==w:
                        continue
                    pair = sorted([w,n2],key=len)
                    if pair[0] == pair[-1][:len(pair[0])]:
                        #print(w,pair)
                        continue
                    count+=1
                    nodes.add(n2)
                    if count==kn:
                        break
        return self.visualize_wordspace(nodes,topn=topn_visible,return_2d=return_data,clustering=clustering,reducer=reducer)


## Visualization
def reduce_mat(mat,reducer='umap'):
    if reducer=='umap':
        import umap
        reducer = umap.UMAP()
    elif reducer=='tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE()
    elif reducer=='pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, svd_solver='full')
    embedding = reducer.fit_transform(mat)
    return embedding

def get_2d_embedding(mat,reducer='umap'):
    "Reduction models include: TSNE UMAP and PCA,'tsne','umap','pca'"
    print('Reduce to 2d via: %s'%reducer.upper())
    embedding = reduce_mat(mat,reducer=reducer)
    return embedding

## Preprocessing
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
def to_bow(texts,w2id,ngram=3,sparse = True):
    vocab_size = len(w2id)
    bows = [Counter(to_index(get_ngram(text,ngram),w2id)) for text in texts]
    if sparse:
        return to_sparse(bows,vocab_size)
    else:
        return bows

def calculate_w2vec_size(count):
  count = count/1000000
  base_size = 64
  return min(max(32,base_size*(int(np.log2(count)-1))),256)


def replace_phrases(text,phrases):
    for i in phrases:
        text = text.replace(i,'_'.join(i.split()))
    return text
def run_w2vec(texts,emb_size=64,known_phrases=[], return_phrased=False,return_counter=False,kwargs={},phrases=True):
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
def run_entropy(dtm):
    tfidf = dtm_tfidf(dtm)
    ents = []
    for i in tqdm.tqdm(np.arange(bow.shape[1])):
        idx = (np.asarray(bow[:,i].sum(axis=1)).flatten()>0)*1
        s = np.asarray(tfidf[idx].sum(axis=0)).flatten()
        ent = entropy(s[s>0])
        ents.append(ent)
    ents = np.array(ents)
    return ents
