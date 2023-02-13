import os
__copyright__ = "Copyright (C) Snorre Ralund"
__license__ = "Work in progress, please do not share or circulate"
__version__ = 0.3
__author__ = 'Snorre Ralund, PhD'
import pandas as pd
import matplotlib.pyplot as plt
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
    import pyvis
except:
    inp = input('The pyvis module is not installed. Do you want to install? Press y. Used for interactive network visualization.')
    if inp =='y':
        os.system(
        'pip install pyvis')
        import pyvis
try:
    import infomap
except:
    pass
    inp = input('The infomap community module is not installed. Do you want to install? Press y. Can work without.')
    if inp =='y':
        os.system(
        'pip install infomap')
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
log = True
import logging

if log:
    logging.basicConfig(filename='wordsimlog.log', encoding='utf-8', level=logging.DEBUG)
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
def find_communities(G,key='community_infomap', **kwargs):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id.
    """
    from infomap import Infomap
    i2num = {i:num for num,i in enumerate(G)}
    num2i = {num:i for i,num in i2num.items()}
    g_num = nx.Graph()
    for n,n2 in G.edges():
        g_num.add_edge(i2num[n],i2num[n2])
    im = Infomap(**kwargs)

    print("Building Infomap network from a NetworkX graph...")

    im.add_networkx_graph(g_num)

    print("Find communities with Infomap...")
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    communities = im.get_modules()

    nx.set_node_attributes(g_num, communities, "community")
    for n in g_num:
        ni = num2i[n]
        com = g_num.nodes[n]['community']
        G.nodes[ni][key] = com
    return G
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
    if n==n2: # Not until now we remove self edges no matter what.
      continue
    c,c2 = d[n],d[n2]
    sim = cosine_modified(c,c2,n=n,n2=n2,remove_self_edges=remove_self_edges)
    if np.isnan(sim): # fix problem if selfedgs are not included.s
      sim = 0
    cos_sims[(n,n2)] = sim
  return cos_sims
def build_graph_from_similarities(cos_sims,check_diff = 0.01,min_sim=False,induce_sparsity=False,manual_cut=False,log=False,large_component_size=False):
  if manual_cut:
    ncount = 0
    for edge, sim in tqdm.tqdm(Counter(cos_sims).most_common()):
      if edge[0]==edge[1]:
        continue
      if sim<manual_cut:
        break
      n,n1 = edge
      if not g.has_node(n):
          g.add_node(n,**{'arrival':ncount})
          ncount+=1
      if not g.has_node(n1):
          g.add_node(n1,**{'arrival':ncount})
          ncount+=1
      g.add_edge(*edge,weight=sim)
    return g
  if not min_sim:
    min_sim = np.quantile(list(cos_sims.values()),0.5)
    print(min_sim)
    mindefined=False
  else:
    mindefined=True
  g = nx.Graph()
  best_score = 0
  count = 0
  ncount = 0
  last_sim = max(cos_sims.values())
  if log:
      scores = []
  if induce_sparsity:
    print('Locating optimal cut point, maximizing sparsity, degree equally and logged component size...')
  else:
    print('Locating optimal cut point, maximizing sparsity, degree equally and logged component size, and ratio2next biggest component... ')
  for edge, sim in tqdm.tqdm(Counter(cos_sims).most_common()):
    if edge[0]==edge[1]:
      continue
    n,n1 = edge
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
          d['clus'] = nx.average_clustering(g)
          d['trans'] = nx.transitivity(g)
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
      if large_component_size:
        mindefined = True
      break
  stats = {'best_score':best_score,'cut':best_sim}
  print('Cut found here:',stats)
  if mindefined:
    best_sim = min_sim
  g = nx.Graph()
  ncount = 0
  for edge, sim in tqdm.tqdm(Counter(cos_sims).most_common()):
    if edge[0]==edge[1]:
      continue
    if sim<best_sim:
      break
    if not g.has_node(n):
      g.add_node(n,**{'arrival':ncount})
      ncount+=1
    if not g.has_node(n1):
      g.add_node(n1,**{'arrival':ncount})
      ncount+=1
    g.add_edge(*edge,weight=sim)

  if log:
    return g,pd.DataFrame(scores)
  return g
def placeholder_func(doc):
    return doc
def preprocess_default(doc):
    if type(doc)==str:
        return nltk.word_tokenize(doc)
    return doc
import codecs
import nltk
class DocsIter():
  """Class for iterating through documents, input can be list or filename with \n\r separated documents."""
  def __init__(self,input,filter_func=lambda x: not x, preprocess=preprocess_default, params={},postprocess=placeholder_func
  ,randomize_post=False,run_in_memory=True,index_files=False,index_size=50000,index_min=5,index_folder = 'temp_indexed/'):
# index size and index min is a tradeoff between coverage (with bias for the first 50000) and efficiency of the index.
    self.filter_func = filter_func
    self.input = input
    self.params = params
    self.preprocess = preprocess
    self.postprocess = postprocess
    self.randomize_post = randomize_post
    self.run_in_memory=run_in_memory
    self.postprocess_on = True
    if type(self.input)!=str:
        if run_in_memory:
            print('Preprocessing docs...')
            proc = []

            for doc in self.input:
                doc = self.preprocess(doc,**self.params)
                doc = [i for i in doc if not self.filter_func(i)]
                if self.postprocess_on:
                    doc = self.postprocess(doc)
                proc.append(doc)
            self.input = proc
            self.n_docs = len(proc)
    self.i = -1
    if type(input)==str:
        self.filename = input
        self.folder = False
        self.current_index_list = False
        self.index_i = False
        # check if input is a folder of indexed files
        if os.path.isdir(input) and 'indexed' in input:
            self.new_index = False
            self.folder = input
            self.index_i = -1
            self.current_index_list = []
            self.index = pickle.load(open(self.input+'index.pkl','rb'))
            self.filename = False
    else:
        self.filename = False
        self.folder = False
        self.current_index_list = False
        self.index_i = False

    if index_files:
        if os.path.isdir(index_folder):
            import shutil
            shutil.rmtree(index_folder)
        os.mkdir(index_folder)
        print('Indexing files')
        logging.info('indexing files')
        #self.index_files = [self.input+i for i in os.listdir(input)]
        # initialize index
        self.index = ['__rare__']
        w2i = {}
        proc = []
        count = 0
        ifilecount = 0
        for doc in self:
            doc = self.preprocess(doc,**self.params)
            doc = [i for i in doc if not self.filter_func(i)]
            if self.postprocess_on:
                doc = self.postprocess(doc)
            proc.append(doc)
            count+=1
            if count%index_size==0:
                w_count = len(self.index)
                i_c = Counter()
                for d in proc:
                    for w in d:
                        if not w in w2i:
                            i_c[w]+=1
                for w,cw in i_c.most_common():
                    if cw<index_min:
                        break
                    w2i[w] = w_count
                    self.index.append(w)
                    w_count+=1
                newproc = []
                for d in proc:
                    newproc.append([w2i[i] if i in w2i else 0 for i in d])
                # dump indexed file
                logging.info('dumping index file %d'%ifilecount)
                pickle.dump(newproc,open(index_folder+'%d.pkl'%ifilecount,'wb'))
                ifilecount+=1
                proc = []
        w_count = len(self.index)
        i_c = Counter()
        for d in proc:
            for w in d:
                if not w in w2i:
                    i_c[w]+=1
        for w,cw in i_c.most_common():
            if cw<index_min:
                break
            w2i[w] = w_count
            self.index.append(w)
            w_count+=1
        newproc = []
        for d in proc:
            newproc.append([w2i[i] if i in w2i else 0 for i in d])
        # dump indexed file
        logging.info('%d %d %d doccount and indexfilecount and index len'%(count,ifilecount,len(self.index)))
        pickle.dump(newproc,open(index_folder+'%d.pkl'%ifilecount,'wb'))
        # dump index
        print('done indexing')
        pickle.dump(self.index,open(index_folder+'index.pkl','wb'))
        self.n_docs = count
        self.current_index_list = []
        #self.current_index_list = pickle.load(open(index_folder+'0.pkl','rb'))
        #print(len(self.current_index_list))
        self.folder = index_folder
        self.input = index_folder
        self.index_i = -1
        self.new_index = True
    #self.f = codecs.open(filename,'r','utf-8')
    self.i = -1

  def __next__(self):
    if type(self.index_i)!=bool:
        l = self.current_index_list
        if len(l)==0:
            # try to open next index files
            self.index_i+=1
            if os.path.isfile(self.folder+'%d.pkl'%self.index_i):
                self.current_index_list = pickle.load(open(self.folder+'%d.pkl'%self.index_i,'rb'))
                l = self.current_index_list
            else:
                self.n_docs = self.i+1
                raise StopIteration
        doc = l.pop(0)
        self.i+=1
        doc = [self.index[i] for i in doc]
        if self.new_index == False:
            doc = self.preprocess(doc,**self.params)
            doc = [i for i in doc if not self.filter_func(i)]
        if self.postprocess_on:
            doc = self.postprocess(doc)
        return doc

    if type(self.input)!=str:

        self.i+=1
        if self.i==len(self.input):
            self.n_docs = self.i
            raise StopIteration
        doc = self.input[self.i]
        if not self.run_in_memory:
            doc = self.preprocess(doc,**self.params)
            doc = [i for i in doc if not self.filter_func(i)]
            if self.postprocess_on:
                doc = self.postprocess(doc)
        return doc
    temp_line = []
    while True:
      line = self.f.readline()
      temp_line.append(line)
      if '\r' in line:
        doc = self.preprocess('\n'.join(temp_line),**self.params)
        doc = [i for i in doc if not self.filter_func(i)]
        self.i+=1
        #doc+=(['__out__']*5)
        if self.postprocess_on:
            doc = self.postprocess(doc)
        return doc
      if len(line)==0:
        self.f.close()
        self.n_docs = self.i+1
        raise StopIteration
  def __iter__(self):
    if self.filename:
        self.f = codecs.open(self.filename,'r','utf-8')
    if self.folder:
        self.index_i = -1
        self.current_index_list = []
    else:
        self.index_i = False
    self.i = -1
    return self
  def __repr__(self):
      return self.input
  def __len__(self):
      try:
          return getattr(self,'n_docs')
      except:

          for doc in self:
              pass
      return getattr(self,'n_docs')

class Resolver():
    def __init__(self,clean,e2e,phrases=False):
        self.clean = clean
        self.e2e = e2e
        self.phrases = phrases
    def resolve(self,w):
        w2 = self.clean(w.lower())
        if w2 in self.e2e:
            w = resolve_ent(w2,self.e2e)
        return w
    def resolve_doc(self,doc):
        l = []
        for w in doc:
            l.append(self.resolve(w))
        if type(self.phrases)!=type(False):
            l = self.phrases[l]
        return l
    def __call__(self,obj):
        if type(obj)==str:
            return self.resolve(obj)
        if type(obj)==list:
            return self.resolve_doc(obj)
        return False
def resolve_docs(docs,e2e,clean):
    docs2 = []
    c = Counter()
    for doc in docs:
        l = resolve_doc(doc,clean,e2e)
        for w in doc:
            c[w]+=1
        docs2.append(l)
    return docs2,c
def resolve_doc(doc,clean,e2e):
    for w in doc:
        w2 = clean(w.lower())
        if len(w2)==0:
            continue
        if w2 in e2e:
            w = resolve_ent(w2,e2e)
        l.append(w)
    return l
def resolve_ent(e,e2e):
    if e in e2e:
        new = e2e[e]
        return new # skip recursion for now.
        if new==e:
            return e
        # recursive part
        return resolve_ent(new,e2e)
    return e

def replace_phrases(text,phrases):
    for i in phrases:
        text = text.replace(i,'_'.join(i.split()))
    return text

def trim_counter(c,max_tokens,p=0.75):
    nnew = int(max_tokens*p)
    c = Counter(dict(c.most_common(nnew)))
    return c
import langdetect
def prepare_docs(docs,clean=lambda x:x,filter_func=lambda x: not x,stem=False,resolve_entities=True,return_e2e=False,phrases=False,run_in_memory=True,max_tokens=150000,verbose=False,index_files=True,index_folder='temp_indexed/',max_lans=3):
    """Function for preparing documents.
    Documents can be either a lists of strings, lists of tokenized docs or a path to a file for streaming data (documents should be separated by '\n\r').
    Tokenization, Cleaning, Mapping between original and cleaned version to merge entities, and Phrasing using collocation detector.
    Phrases set to True if you want to locate bigrams before creating the cooccurence network."""
    if not type(docs) == DocsIter:
        docs = DocsIter(docs,run_in_memory=run_in_memory,filter_func=filter_func,index_files=index_files,index_folder=index_folder)
    if stem:
        print('Not implemented yet, use custom clean function instead.')
    resolver = Resolver(e2e={},clean=clean)
    if resolve_entities:
        print('Resolving entities to most common shared representation (after cleaning and lowercasing)')
        c = Counter()
        c2 = Counter()
        for num,doc in enumerate(docs):
            if verbose:
                if num%20000==0:
                    print(num,len(c),len(c2),end='\r')
                    logging.info('%d,%d,%d'%(num,len(c),len(c2)))
            for w in doc:
                j = clean(w.lower())
                if len(j)==0:
                    continue
                c[w]+=1
                c2[j]+=1
            if len(c2)>max_tokens:
                #nnew = int(max_tokens*0.75)
                #c = Counter(dict(c.most_common(nnew)))
                c2 = trim_counter(c2,max_tokens)
                logging.info('trimming')
                # trim c with c2
                c = Counter({i:c[i] for i in c if clean(i.lower()) in c2})
        logging.info('Resolve')
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
                e2e[e2]=ent
        del c2
        resolver = Resolver(e2e=e2e,clean=clean)
        docs.postprocess = resolver
    logging.info('Done resolving')
    if phrases:
        print('Locating collocations...')
        from gensim.models.phrases import Phrases
        phrase_model_bi = Phrases(docs)
        #phrase_docs_bi = [phrase_model_bi[sent] for sent in docs]
        resolver.phrases = phrase_model_bi
        docs.postprocess = resolver
    if type(docs.input) == list:
        docs = DocsIter(docs.input,filter_func=filter_func,postprocess=resolver,run_in_memory=True)
    logging.info('Count tokens')
    lans = []
    lan_count = Counter()
    for doc in docs:
        lan = langdetect.detect(doc)
        lan_count[lan]+=1
        lans.append(lan)
    most_lan = lan_count.most_common(1)[0][0]
    DFREQ = {lan:Counter() for lan,count in lan_count.most_common(max_lans))}
    C = {lan:Counter() for lan in languages}
    count = 0
    for num,doc in enumerate(docs):
        lan = lans[num]
        if lan in language_in:
            dfreq = DFREQ[lan]
            c = C[lan]
        else:
            dfreq = DFREQ[most_lan]
            c = C[most_lan]
        dfreq.update(set(doc))
        c.update(Counter(doc))
        count+=1
        if len(c)>max_tokens:
            logging.info('trimming %d %d %d'%(count,len(dfreq),len(c)))
            dfreq = trim_counter(dfreq,max_tokens)
            #nnew = int(max_tokens*0.9)
            #dfreq = Counter(dict(dfreq.most_common(nnew)))
            c = Counter({i:c[i] for i in dfreq})
    w2lan = {}
    c2 = Counter()
    for lan,c in C.items():
        c2.update(c)
    # mmake language specific counts and freqs

    # assigning word to the language with highest percentage
    # might assign certain words to rare languages by coincidence
    # avoid by having a minumum count.
    c = Counter()
    dfreq = Counter()
    for w,count in c2.most_common(max_tokens):
        pm = 0
        cm = 0
        best = most_lan
        for lan in C:
            lanc = lan_count[lan]
            count = C[lan][w]
            if count<4:
                continue
            p = count/lanc
            if p>pm:
                pm = p
                best = lan
                cm = count
        if pm ==0:
            continue
        w2lan[w] = (best,pm)
        dfreq[w] = pm
        c[w] = count
    logging.info('done counting')
    if return_e2e:
        res_e2all = {e:[] for e in e2e.values()}
        g = g.to_undirected()
        for e in c:
            e2 = clean(e.lower())
            if e2 in e2e:
                res = e2e[e2]
                res_e2all[res].append(e)
        return docs,c,dfreq,w2lan,lan_count,(e2e,res_e2all)
    return docs,c,dfreq,w2lan,lan_count
def calculate_pmi_scores(docs,w2lan,lan_count,custom_filter=lambda x: not x,c=False,min_cut=10,max_frac=0.25,min_edgecount=5,max_edges=5000000,maximum_nodes=10000,pmi_min=1.2,remove_self_edges=True,edge_window=64,pmi_smoothing=10):
    logging.info('Calculate pmi')
    cut = min_cut
    if not c:
        c = Counter()
        for doc in docs:
            for w in doc:
                c[w]+=1
    n_docs = len(docs)
    max_count = int(n_docs*max_frac)
    keep = set([i for i,count in c.most_common(maximum_nodes) if count>=cut and count<=max_count and not custom_filter(i)])
    print('%d nodes are kept using minimum cut.'%len(keep))
    print('Start characterizing edges')
    logging.info('Begin counting edges')
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
        if len(edge_c)>max_edges:
            logging.info('trimming edges')
            edge_c = trim_counter(edge_c,max_edges)
            #nnew = int(max_edges*0.75)
            #edge_c = Counter(dict(edge_c.most_common(nnew)))
    logging.info('Done counting edges')
    print('Done counting edges')
    pmis = {}
    alpha = pmi_smoothing # smoothing term
    out = []
    for edge,count in edge_c.items():
        n,n2 = edge
        if count<min_edgecount:
            out.append(edge)
            continue
        l1,l2 = w2lan[n],w2lan[n2]
        nd1,nd2 = lan_count[l1],lan_count[l2]
        p = (c[n]+alpha)/nd1
        p2 = (c[n2]+alpha)/nd2
        m = count/n_docs
        pmi = m/(p*p2)
        if pmi<=pmi_min:
            continue
        pmis[edge] = pmi
    pmis = Counter(pmis)
    for edge in out:
        del edge_c[edge]
    print('PMI done.')
    logging.info('PMI done')
    return pmis,edge_c,keep
def generate_similarity_network(docs,min_cut = 10,max_frac=0.25,phrases=False,min_edgecount=5,maximum_nodes = 10000,stem=False,topn_edges = 100000,target_average_degree=False,edge_window=128,large_component_size=False,
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
    phrases set to True if you want to locate bigrams before creating the cooccurence network.
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
    docs,c,dfreq = prepare_docs(docs,clean=clean,stem=stem,phrases=phrases)
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
def add_hex_color(g,community_key='community',cm=plt.cm.tab20):
    part = {}
    for n in g:
        part[n] = g.nodes[i][community_key]
    coms = set(part.values())
    p2col = {p:num/len(coms) for num,p in enumerate(coms)}
    for n in g:
        g.nodes[n]['color'] = p2col[part[n]]
    return g

def interactive_network(g,output_name='interactive_net.html',in_notebook=True,buttons='all'):
  from pyvis.network import Network
  nt = Network('1000px', '1000px',notebook=False)
  g = add_hex_color(g)
  nt.from_nx(g)

  if buttons=='all':
    nt.show_buttons()
  else:
    nt.show_buttons(filter_=['physics']) # import or else layout cannot be stopped.
  nt.show(output_name)
  # display or download
  if in_notebook:
    import IPython
    IPython.display.HTML(filename=output_name)
def draw_network_quick(g,label_p=0.75,adjust_text=False,node_or_community_norm='neighbor',spatialization=nx.layout.kamada_kawai_layout,custom_key=False):
    '''Function for quick visualization of networkself.
    Choose the fraction of labels to be displayed, will be ordered by relative community degree or relative neighbor degree.
    Use the adjust_text to avoid label overlap.
    node_or_community_norm: choose between 'neighbor' or 'community' or 'degree'
    Input spatialization function e.g. nx.spring_layout or nx.kamada_kawai_layout'''
    import matplotlib.pyplot as plt
    if custom_key:
        sort = sorted(g,key=lambda x: g.nodes[x][key],reverse=True)
    else:
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
    nx.draw_networkx_edges(g,pos=pos,alpha=0.05)
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
