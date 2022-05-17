try:
    from fuzzywuzzy import fuzz
except:
    import os
    inp = input('The fuzzywuzzy package is not installed. Do you want to install? Press y.')
    if inp =='y':
        os.system(
        'pip install fuzzywuzzy')
    from fuzzywuzzy import fuzz
try:
    import networkx as nx
except:
    import os
    inp = input('The networkx package is not installed. Do you want to install? Press y.')
    if inp =='y':
        os.system(
        'pip install networkx')
    import networkx as nx
import pandas as pd
import tqdm
def decode(str1):
    try:
        return str1.encode('utf-8').decode('ascii','ignore')
    except:
        return str1.decode('ascii','ignore')

def get_fuzz_scores(str1,str2,lower=True):
    Str1, Str2 = str1,str2

    if lower:
        str1, str2 = str1.lower(),str2.lower()
    try:
        partial = fuzz.partial_ratio(str1,str2)
    except:
        str1,str2 = decode(str1),decode(str2)
        partial = fuzz.partial_ratio(str1,str2)
    lev = fuzz.ratio(str1,str2)
    token_sort_ratio = fuzz.token_sort_ratio(str1,str2)
    all_ = (lev+token_sort_ratio+partial)/3
    return {'partial':partial,'levensthein':lev,'token_sort':token_sort_ratio,'robust':all_,'str1':Str1,'str2':Str2}

def get_similarities(a,a2,lower=True):
    a,a2 = set(a),set(a2)
    if lower:
        new,new2 = set([i.lower() for i in a]),set([i.lower() for i in a2])
        return get_similarities(new,new2,lower=False)
    common = set()
    dat = []
    for i in tqdm.tqdm(a):
        if i in a2:
            common.add(i)
            continue
        for j in a2:
            d = get_fuzz_scores(i.replace('_',' '),j.replace('_',' '))
            dat.append(d)

    return dat,common
def string_match(a,a2,lower=True,minimum=80,one2one = False,measure='robust',return_data=False):
    dat,common = get_similarities(a,a2,lower=True)
    if lower:
        back = {}

        for i in a:
            back[i.lower()] = i
        back2 = {}
        for i in a2:
            back2[i.lower()] = i


    trans = {i:i for i in common}

    sim = pd.DataFrame(dat)
    sim = sim[sim[measure]>=minimum]
    sim = sim.sort_values(measure,ascending=False)

    if one2one:
        g = nx.Graph()
        in_ = set()
        for i,j in sim[['str1','str2']].values:
            if not g.has_node(i) and not g.has_node(j):
                g.add_edge(i,j)
                in_.add(i)
        for i in in_:
            trans[i] = list(g[i].keys())[0]
    else:
        temp = sim.groupby('str1').apply(lambda x: x.str2.values[0])
        for i,j in temp.reset_index().values:
            trans[i] = j
    if lower:
        trans = {back[i]:back2[j] for i,j in trans.items()}
    if return_data:
        return trans,pd.DataFrame(dat)
    return trans
