import os
try:
    import community
except:
    os.system('pip install python-louvain')
    import community
try:
    import gensim
except:
    os.system('pip install gensim -U --quiet')
    import gensim
try:
    import run_w2vec as W2V
except:
    os.system('curl https://raw.githubusercontent.com/snorre87/wordviz/main/run_w2vec.py > run_w2vec.py')
    import run_w2vec as W2V
import pickle
from collections import Counter
import numpy as np
import tqdm
import os
    ## edge based jaccard
def jaccard_d(d,d2):
    nom = len(d&d2)
    return nom/len(d|d2)

from community import community_louvain
def add_community_relative_degree(g):
    part = community_louvain.best_partition(g)
    # calculate community degree to weigh labels
    com2n = {p:[] for p in part.values()}
    for n,p in part.items():
        g.nodes[n]['community'] = str(p)
        com2n[p].append(n)
    for p,nodes in com2n.items():
        degs = np.array([len(g[n]) for n in nodes])
        ma = max(degs)
        m = np.mean(degs)
        degs_sqrt = np.sqrt(degs)
        rel_deg = degs/ma
        for n,rel_d in zip(nodes,rel_deg):
            g.nodes[n]['relative_degree'] = rel_d
    return g
def generate_pmi_network(docs,min_cut = 10,maximum_nodes = 10000,w2vec_pretrained=False,w2vec_path=False,clean=lambda x:x, pmi_smoothing=10,topn = 100000,return_knn=False,add_community=False):
    cut = min_cut
    c = Counter()
    c2 = Counter()
    for doc in docs:
        for w in doc:
            w = clean(w)
            if len(w)==0:
                continue
            c[w]+=1
            c2[w.lower()]+=1
    # Remove duplicates from different spellings and lowercasing
    g = nx.Graph()
    for e in c:
        j = e.lower()
        if len(e)==0:
            continue
        if e[0].isupper():
            if c2[j]>c[e]:
                g.add_edge(j,e)

    e2e = {}
    for e in c:
        e2 = e.lower()
        if g.has_node(e2):
            ent = list(g[e2].keys())[0]
            e2e[e]=ent
    def resolve_ent(e,e2e=e2e):
        if e in e2e:
            return e2e[e]
        return e
    print(len(g),len(e2e))
    del c2
    c = Counter()
    docs2 = []
    for doc in docs:
        l = []
        for w in doc:
            w = clean(w)
            if len(w)==0:
                continue
            w = resolve_ent(w,e2e)

            l.append(w)
            c[w]+=1
        docs2.append(l)

    keep = set([i for i,count in c.most_common(maximum_nodes) if count>=cut])
    print('%d nodes are kept'%len(keep))
    docs = docs2
    if type(w2vec_pretrained) == type(False):
        if type(w2vec_pretrained)==str:
            ent2v = pickle.load(open(w2vec_pretrained,'rb'))
        else:
            if os.path.isfile(w2vec_path):
                ent2v = pickle.load(open(w2vec_path,'rb'))
            else:
                print('W2VEC training')
                ent2v = W2V.run_w2vec(docs,phrases=False,emb_size=128)
                if type(w2vec_path)==str:
                    pickle.dump(ent2v,open(w2vec_path,'wb'))
    else:
        ent2v = w2vec_pretrained
    n_docs = len(l)
    print('Start characterizing edges')
    edges = []
    for doc in docs:
        ents = [i for i in doc if i in keep]
        for i in range(len(ents)-1):
            n = ents[i]
            for j in range(i+1,len(ents)):
                n2 = ents[j]
                edges.append(tuple(sorted([n,n2])))
    edge_c = Counter(edges)
    pmis = {}
    alpha = pmi_smoothing # smoothing term
    for edge,count in edge_c.items():
        n,n2 = edge
        p = (c[n]+alpha)/n_docs
        p2 = (c[n2]+alpha)/n_docs
        m = count/n_docs
        pmis[edge] = m/(p*p2)
    pmis = Counter(pmis)

    # W2vec distance
    edge2dist = {}
    error= 0
    for n,n2 in tqdm.tqdm(pmis):
        try:
            dist = ent2v.wv.distance(n,n2)
        except:
            error+=1
            dist = np.nan
        edge2dist[tuple(sorted([n,n2]))] = dist
    g = nx.Graph()
    for edge,pmi in pmis.most_common(topn):
        n,n2 = edge
        dist = edge2dist[edge]
        count = edge_c[edge]
        #t,t2 = e2typ[n],e2typ[n2]
        g.add_node(n,**{'n_docs':c[n]})
        g.add_node(n2,**{'n_docs':c[n2]})
        g.add_edge(n,n2,**{'w2vec_dist':dist,'pmi':pmi,'count':count})
    edge2jacc = {}
    for edge,count in tqdm.tqdm(pmis.most_common(topn)):
        n,n2 = edge
        d = set(g[n])
        d2 = set(g[n2])
        edge2jacc[edge] = jaccard_d(d,d2)

    knn = nx.DiGraph()
    for n in tqdm.tqdm(list(g)):
        #t = e2typ[n]
        #g.add_node(n,**{'type':t,'n_docs':c[n]})
        for k,n2 in enumerate(sorted(g[n],key=lambda x: g[n][x]['pmi'])):
            d = g[n][n2].copy()
            d['k'] = k
            knn.add_edge(n,n2,**d)
    for n,n2 in tqdm.tqdm(g.edges()):
        jacc = edge2jacc[tuple(sorted([n,n2]))]
        g[n][n2]['jacc_sim'] = jacc
        if knn.has_edge(n,n2):
            k1 = knn[n][n2]['k']
            knn[n][n2]['jacc_sim'] = jacc
        else:
            k1 = np.nan
        if knn.has_edge(n2,n):
            k2 = knn[n2][n]['k']
            knn[n2][n]['jacc_sim'] = jacc
        else:
            k2 = 999
        g[n][n2]['k1'] = k1
        g[n][n2]['k2'] = k2
        g[n][n2]['min_k'] = min(k1,k2)
        g[n][n2]['max_k'] = max(k1,k2)
    if return_knn:
        return g,knn
    if add_community:
        g = add_community_relative_degree(g)
    return g
