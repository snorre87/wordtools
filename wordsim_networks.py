import os
__copyright__ = "Copyright (C) Snorre Ralund"
__license__ = "Work in progress, please do not share or circulate"
__version__ = 0.2
__author__ = 'Snorre Ralund, PhD'
import pandas as pd
try:
    import networkx as nx
except:
    inp = input('The networkx module is not installed. Do you want to install? Press y. Can work without.')
    if inp =='y':
        os.system('pip install networkx')
        import networkx as nx
try:
    import community
except:
    inp = input('The community module is not installed. Do you want to install? Press y. Can work without.')
    if inp =='y':
        os.system(
        'pip install python-louvain')
        import community
try:
    import infomap
except:
    pass
    #inp = input('The infomap community module is not installed. Do you want to install? Press y. Can work without.')
    #if inp =='y':
    #    os.system(
    #    'pip install infomap')
    #    import infomap
try:
    import gensim
except:
    inp = input('The gensim module is not installed. Do you want to install? Press y. Can work without.')
    if inp =='y':
        os.system('pip install gensim -U --quiet')
        import gensim
try:
    import run_w2vec as W2V
except:
    print("Collecting script to estimate word2vec heuristically.")
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
def run_infomap_alt(g):
    from infomap import Infomap
    n2num = {u:num for num,u in enumerate(g)}
    num2u = sorted(n2num,key=lambda x: n2num[x])
    g_num = nx.Graph()
    for n,n1 in g.edges():
        g_num.add_edge(n2num[n],n2num[n1])

    im = Infomap("--undirected")
    for n,n1 in g_num.edges():
        im.addLink(n,n1)
    im.run()

    part = {num2u[i]:m for i,m in im.getModules().items()}
    return part
def run_infomap(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id.
    """
    import infomap
    im = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")

    im.add_networkx_graph(G)

    print("Find communities with Infomap...")
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    communities = im.get_modules()
    nx.set_node_attributes(G, communities, 'community')
def add_community_relative_degree(g,method='louvain'):
    """Partition network using either 'infomap' or 'louvain' and compute community relative degree. Degree/max_degree of community
    Infomap is unstable, and might crash the run time."""
    if method=='louvain':
        part = community_louvain.best_partition(g)
    else:
        part = run_infomap(g)
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

def add_neighbor_relative_degree(g):
    scores = []
    for n in g:
        ns = g[n]
        deg = len(ns)
        degs = [len(g[n2]) for n2 in ns]
        std = np.std(degs)
        m = np.mean(degs)
        score = (deg-m)/std
        g.nodes[n]['neighbor_relative_degree'] = score
        scores.append(score)
    #min_ = min(scores)
    #for n in g:
    #    score = g.nodes[n]['neighbor_relative_degree']
    #    score +=(min_+1)
    #    g.nodes[n]['neighbor_relative_degree'] = score
    return g
from collections import Counter
import numpy as np
from numpy import dot
from numpy.linalg import norm
def cosine_modified(c,c2,n=False,n2=False,remove_self_edges=True): # hurtigst
    "Calculates cosine with the added functionality of removing self in the cooccurence graph."
    l = set(c)|set(c2)
    if remove_self_edges:
        l = l-set([n,n2])
    a,a2 = np.array([c[i] for i in l]),np.array([c2[i] for i in l])
    return dot(a, a2)/(norm(a)*norm(a2))
def calculate_pmi_similarity(pmis,penalty_pmi = np.sqrt,max_inspected_edges = 250000,remove_self_edges=True):
  d = {}
  most = pmis.most_common(max_inspected_edges)
  for (n,n2),pmi in most:
    if type(penalty_pmi)!=type(bool):
      pmi = penalty_pmi(pmi)
    try:
      d[n][n2] = pmi
    except:
      d[n] = Counter({n2:pmi})
    try:
      d[n2][n] = pmi
    except:
      d[n2] = Counter({n:pmi})
  cos_sims = {}
  for (n,n2),_ in tqdm.tqdm(most):
    c,c2 = d[n],d[n2]
    cos_sims[(n,n2)] = cosine_modified(c,c2,n=n,n2=n2,remove_self_edges=remove_self_edges)
  return cos_sims
def build_graph_from_similarities(cos_sims,check_diff = 0.01,min_sim=False,induce_sparsity=False,manual_cut=False,log=False,large_component_size=False):
  if manual_cut:
    for edge, sim in tqdm.tqdm(Counter(cos_sims).most_common()):
      if sim<manual_cut:
        break
      g.add_edge(*edge,weight=sim)
    return g
  if not min_sim:
    min_sim = np.quantile(list(cos_sims.values()),0.9)
  g = nx.Graph()
  best_score = 0
  count = 0
  last_sim = max(cos_sims.values())
  if log:
      scores = []
  if induce_sparsity:
    print('Locating optimal cut point, maximizing sparsity, degree equally and logged component size...')
  else:
    print('Locating optimal cut point, maximizing sparsity, degree equally and logged component size, and ratio2next biggest component... ')
  for edge, sim in tqdm.tqdm(Counter(cos_sims).most_common()):
    g.add_edge(*edge,weight=sim)
    if (last_sim-sim)>check_diff:
      last_sim = sim
      comps = list(nx.connected_components(g))
      lens = sorted([len(i) for i in comps],reverse=True)
      ratio2next = lens[0]/sum(lens[0:2])
      big = nx.subgraph(g,max(comps,key=len))
      deg = [len(big[i]) for i in big]
      #ent = scipy.stats.entropy(deg)
      dens = nx.density(big)
      sparse = 1-dens
      largest = len(big)
      degs = deg
      degs = np.array(degs)
      unequal = degs.mean()**2/(degs*degs).mean()
      if log:
          d = {'equality':unequal,'largest':largest,'sparsity':sparse,'ratio2next':ratio2next,
          'sim':sim,'n_edges':len(g.edges()),'n':len(g)}
          scores.append(d)
      if induce_sparsity:
        score = np.log(largest)*sparse*unequal
      else:
        score = np.log(largest)*sparse*unequal*ratio2next
      #score = (largest/len(g))*sparse*unequal#*ent
      if large_component_size:
          if largest>large_component_size:
            best_score = score
            best_sim = sim
            break
      if score>best_score:
        #best_g = g.copy()
        best_score = score
        best_sim = sim

    count +=1
    if sim<=min_sim:
      break
  stats = {'best_score':best_score,'cut':best_sim}
  print('Cut found here:',stats)
  g = nx.Graph()
  for edge, sim in tqdm.tqdm(Counter(cos_sims).most_common()):
    if sim<best_sim:
      break
    g.add_edge(*edge,weight=sim)

  if log:
    return g,pd.DataFrame(scores)
  return g

def resolve_docs(docs,e2e,clean):
    docs2 = []
    c = Counter()
    for doc in docs:
        l = []
        for w in doc:
            w2 = clean(w.lower())
            if len(w)==0:
                continue
            if w2 in e2e:
                w = resolve_ent(w,e2e)
            l.append(w)
            c[w]+=1
        docs2.append(l)
    return docs2,c
def resolve_ent(e,e2e):
    if e in e2e:
        new = e2e[e]
        if new==e:
            return e
        # recursive part
        return resolve_ent(new,e2e)
    return e

def prepare_docs(docs,clean=lambda x:x,stem=False,resolve_entities=True,return_e2e=False):
    import random
    if type(random.choice(docs))==str:
        print('Will tokenize data...')
        import nltk
        docs = [nltk.word_tokenize(doc) for doc in docs]
        ## missing phraser.
    if stem:
        print('Not implemented yet, use custom clean function instead.')
    if resolve_entities:
        print('Resolving entities to most common shared representation (after cleaning and lowercasing)')
        c = Counter()
        c2 = Counter()
        for doc in docs:
            for w in doc:
                j = clean(w.lower())
                if len(j)==0:
                    continue
                c[w]+=1
                c2[j]+=1
        # Remove duplicates from different spellings and lowercasing
        ##missing apply optional lemmatizer or stemmer, the clean function could be a stemmer.
        g = nx.DiGraph()
        for e in c:
            j = clean(e.lower())
            #apply clean and lemma
            if len(e)==0:
                continue
            if e==j:
                continue
            if c2[j]>c[e]:
                g.add_edge(j,e)

        e2e = {}
        for e in c:
            e2 = clean(e.lower())
            if e2==e:
                continue
            if g.has_node(e2):
                if len(g[e2])==0:
                    continue
                ent = sorted(g[e2],key=lambda x: c[x])[-1]
                e2e[e]=ent
        del c2
        docs,c = resolve_docs(docs,e2e,clean)
    else:
        c = Counter()
        new_docs = []
        for doc in docs:
            new_doc = []
            for w in doc:
                w =clean(w)
                c[w] +=1
                new_doc.append(w)
            new_docs.append(new_doc)
        docs = new_docs
    if return_e2e:
        return docs,c,e2e
    return docs,c
def calculate_pmi_scores(docs,custom_filter=lambda x: not x,c=False,min_cut=10,max_frac=0.25,min_edgecount=5,maximum_nodes=10000,pmi_min=1.2,remove_self_edges=True,edge_window=64,pmi_smoothing=10):
    cut = min_cut
    max_count = int(len(docs)*max_frac)
    if not c:
        c = Counter()
        for doc in docs:
            for w in doc:
                c[w]+=1

    keep = set([i for i,count in c.most_common(maximum_nodes) if count>=cut and count<=max_count and not custom_filter(i)])
    print('%d nodes are kept using minimum cut.'%len(keep))
    n_docs = len(docs)
    print('Start characterizing edges')
    edge_c = Counter()
    for doc in docs:
        ents = [i for i in doc if i in keep]
        for i in range(len(ents)-1):
            n = ents[i]
            for j in range(i+1,min(i+edge_window,len(ents))):
                n2 = ents[j]
                if remove_self_edges:
                    if n==n2:
                        continue
                edge_c[tuple(sorted([n,n2]))] +=1
    pmis = {}
    alpha = pmi_smoothing # smoothing term
    out = []
    for edge,count in edge_c.items():
        n,n2 = edge
        if count<min_edgecount:
            out.append(edge)
            continue
        p = (c[n]+alpha)/n_docs
        p2 = (c[n2]+alpha)/n_docs
        m = count/n_docs
        pmi = m/(p*p2)
        if pmi<=pmi_min:
            continue
        pmis[edge] = pmi
    pmis = Counter(pmis)
    for edge in out:
        del edge_c[edge]
    print('PMI done.')
    return pmis,edge_c,keep
def generate_similarity_network(docs,min_cut = 10,max_frac=0.25,min_edgecount=5,maximum_nodes = 10000,stem=False,topn_edges = 100000,target_average_degree=False,edge_window=128,large_component_size=False,
custom_filter=lambda x: not x,
sorting_measure='pmi',pmi_min=1.2,build_from_pmi_weighted_sims=True,induce_sparsity=False
,penalty_pmi = np.sqrt,max_inspected_edges = 250000
,w2vec_pretrained=False,w2vec_path=False,
clean=lambda x:x, pmi_smoothing=10,return_knn=False
,add_community=True,add_w2vec_sim=True,add_knn_info=True,w2vec_docs=False
,remove_self_edges=False):
    """Function for creating a network out of documents. It calculates pmi, jaccard similarity, and word2vec similarity of entities/words, and creates a network out of the X most similar words.
    Input should be a list of tokenized docs, or list of strings. Could be lists of words or e.g. named entities. Anything goes.
    Choose min_cut and or maximum_nodes to include only tokens with a mininum count and a maximum number of nodes.
    induce_sparsity is set to false: Defines a different objective for making the optimal cut in the dendrogram of the similarity graph. If this is set to True, it will only optimize sparsity and eqaully and not size of the resulting graph. Resulting in a smaller but more clustered and sparse graph.
    custom_filter should be a function being true if you want to remove an entity. e.g. lambda x: x.isdigit() . will remove numbers
    old...
    sorting_measure: Choose which similarity measure should be used to define the network> 'pmi','w2vec'.
    topn_edges: number of most similar edges to include.
    Or use target_average_degree to choose edges by multiplying the number of nodes with an average degree.
    Add community partitions the network using the louvain modularitybased algorithm, and adds a community relative degree attribute to the nodes 'relative_degree', very useful for visualization of words.
    add_knn_info, set to true if you want edge attributes ranking neighbors.
    """
    cut = min_cut
    topn = topn_edges
    max_count = int(len(docs)*max_frac)
    docs,c = prepare_docs(docs,clean=clean,stem=stem)
    pmis,edge_c,keep = calculate_pmi_scores(docs,custom_filter=custom_filter,c=c,min_cut=min_cut,max_frac=max_frac,min_edgecount=min_edgecount,maximum_nodes=maximum_nodes
    ,pmi_min=pmi_min,remove_self_edges=remove_self_edges
    ,edge_window=edge_window,pmi_smoothing=pmi_smoothing)
    if target_average_degree!=False:
        topn = int(target_average_degree*len(keep))
    # Comuting PMI weighted cooccurrence network similarities
    cos_sims = calculate_pmi_similarity(pmis,penalty_pmi = penalty_pmi,max_inspected_edges=max_inspected_edges,remove_self_edges=remove_self_edges)

    if build_from_pmi_weighted_sims:
        g = build_graph_from_similarities(cos_sims,check_diff = 0.01,min_sim=0.3,induce_sparsity=induce_sparsity,manual_cut=False,large_component_size=large_component_size)
        return g
    ##### Alternative similarity schemes.
    # W2vec distance
    w2vec_isinstalled = 'gensim' in globals()
    if w2vec_isinstalled and add_w2vec_sim:
        if type(w2vec_pretrained) == type(False):
            if type(w2vec_pretrained)==str:
                ent2v = pickle.load(open(w2vec_pretrained,'rb'))
            else:
                if os.path.isfile(w2vec_path):
                    ent2v = pickle.load(open(w2vec_path,'rb'))
                else:
                    print('W2VEC training')
                    if type(w2vec_docs)!=type(bool):
                        w2vec_docs,c = resolve_docs(w2vec_docs,e2e,clean)
                        ent2v = W2V.run_w2vec(w2vec_docs,phrases=False)
                    else:
                        ent2v = W2V.run_w2vec(docs,phrases=False)
                    if type(w2vec_path)==str:
                        pickle.dump(ent2v,open(w2vec_path,'wb'))
        else:
            ent2v = w2vec_pretrained

    if w2vec_isinstalled:
        edge2sim = {}
        error= 0
        for n,n2 in tqdm.tqdm(pmis):
            try:
                sim = 1-ent2v.wv.distance(n,n2)
            except:
                error+=1
                # assume 1
                sim = 0
            edge2sim[tuple(sorted([n,n2]))] = sim
        edge2sim = Counter(edge2sim)
        print('W2Vec done.')
    g = nx.Graph()
    #
    if sorting_measure=='w2vec':
        assert 'edge2sim' in dir(),'Word2vec is not esimated. Install gensim or set sorting mechanism to pmi or jaccard. Will abort.'
        sort = edge2sim.most_common(topn)
    elif sorting_measure =='pmi':
        sort = pmis.most_common(topn)
    #elif sorting_measure == 'jaccard':
    #    sort = edge2jacc.most_common(topn)
    for edge,_ in sort:
        n,n2 = edge
        if w2vec_isinstalled:
            sim = edge2sim[edge]
        else:
            sim = np.nan
        pmi = pmis[edge]
        count = edge_c[edge]
        #t,t2 = e2typ[n],e2typ[n2]
        g.add_node(n,**{'n_docs':c[n]})
        g.add_node(n2,**{'n_docs':c[n2]})
        g.add_edge(n,n2,**{'w2vec_similarity':sim,'pmi':pmi,'count':count})

    edge2jacc = Counter()
    for edge,_ in tqdm.tqdm(sort):
        n,n2 = edge
        d = set(g[n])
        d2 = set(g[n2])
        edge2jacc[edge] = jaccard_d(d,d2)
    # Sorting based on pmi or w2vec or jaccard
    print('Jaccard done.')
    for edge,_ in sort:
        jacc = edge2jacc[edge]
        g[edge[0]][edge[1]]['jaccard_similarity'] = jacc
    # extract_largets component
    components = list(nx.connected_components(g))
    g = nx.subgraph(g,nbunch=max(components,key=len))
    print('%d nodes in the final connected component'%len(g))
    if add_community:
        if 'community' in globals():
            g = add_community_relative_degree(g)
        else:
            print('Community module is not installed. Will not add community information.')
    # add knn
    if add_knn_info:
        if sorting_mechanism=='pmi':
            wkey = 'pmi'
        elif sorting_mechanism=='w2vec':
            wkey = 'w2vec_similarity'
        elif sorting_mechanism=='jaccard':
            wkey = 'jaccard_similarity'
        knn = nx.DiGraph()
        for n in tqdm.tqdm(list(g)):
            for k,n2 in enumerate(sorted(g[n],key=lambda x: g[n][x][wkey])):
                d = g[n][n2].copy()
                d['k'] = k
                knn.add_edge(n,n2,**d)
        for n,n2 in tqdm.tqdm(g.edges()):
            if knn.has_edge(n,n2):
                k1 = knn[n][n2]['k']
            else:
                k1 = np.nan
            if knn.has_edge(n2,n):
                k2 = knn[n2][n]['k']
            else:
                k2 = 999
            g[n][n2]['k1'] = k1
            g[n][n2]['k2'] = k2
            g[n][n2]['min_k'] = min(k1,k2)
            g[n][n2]['max_k'] = max(k1,k2)
        if return_knn:
            return g,knn
    return g
def get_components(g):
    comps = list(nx.connected_components(g))
    return comps
def extract_largest_component(g):
    comps = list(nx.connected_components(g))
    largest = max(comps,key=len)
    big = nx.subgraph(g,largest)
    return big

def draw_network_quick(g,label_p=0.75,adjust_text=False,node_or_community_norm='neighbor',spatialization=nx.layout.kamada_kawai_layout):
    '''Function for quick visualization of networkself.
    Choose the fraction of labels to be displayed, will be ordered by relative community degree or relative neighbor degree.
    Use the adjust_text to avoid label overlap.
    node_or_community_norm: choose between 'neighbor' or 'community' or 'degree'
    Input spatialization function e.g. nx.spring_layout or nx.kamada_kawai_layout'''
    import matplotlib.pyplot as plt
    if node_or_community_norm=='neighbor':
        key = 'neighbor_relative_degree'
        try:
            sort = sorted(g,key=lambda x: g.nodes[x][key],reverse=True)
        except:
            print('will add neighbor relative degre')
            g = add_neighbor_relative_degree(g)
            sort = sorted(g,key=lambda x: g.nodes[x][key],reverse=True)
    elif node_or_community_norm=='community':
        key = 'relative_degree'
        try:
            sort = sorted(g,key=lambda x: g.nodes[x][key],reverse=True)
        except:
            print('will add community info')
            g = add_community_relative_degree(g)
            sort = sorted(g,key=lambda x: g.nodes[x][key],reverse=True)
    else:
        print('Will order by degree')
        sort = sorted(g,key=lambda x: len(g[x]),reverse=True)
    top = sort[:int(len(g)*label_p)]

    try:
        community = [int(g.nodes[i]['community']) for i in g]
    except:
        g = add_community_relative_degree(g)
        community = [int(g.nodes[i]['community']) for i in g]
    colors = [plt.cm.tab20(i/max(community)) for i in community]
    pos = spatialization(g)
    fig = plt.figure(figsize=(40,30))
    nx.draw_networkx_nodes(g,pos=pos,node_color=colors)
    nx.draw_networkx_edges(g,pos=pos)
    labels = []
    for i in top:
        labels.append(plt.text(pos[i][0],pos[i][1],i,fontweight='bold'))

    #labels = nx.draw_networkx_labels(nx.subgraph(g,top),pos=pos)
    if adjust_text:
        try:
            from adjustText import adjust_text
        except:
            print('adjustText not installed. pip install adjustText')

        try:
            print('Adjusting text. May take a while...')
            adjust_text(labels)
        except Exception as e:
            print(e)
            print('Error')

    return fig
