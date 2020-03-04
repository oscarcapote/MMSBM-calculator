# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np
from string import ascii_lowercase
from copy import deepcopy
from numba import jit,prange,int64,double,vectorize,float64
from time import time
import os,sys
import time as time_lib
import argparse
import yaml

# In[32]:


#Parametros del sistema

super_tak = time()
parser = argparse.ArgumentParser()

parser.add_argument("-K",help="Number of node groups",type=int)
parser.add_argument("-L",help="Number of items groups",type=int)

parser.add_argument("--lambda_nodes",help="Intensity of nodes priors",type=float)
parser.add_argument("--lambda_items",help="Intensity of items priors",type=float)

parser.add_argument("-s","--seed",help="Seed to generate the random matrices",type=int)
parser.add_argument("-F","--Fold",help="Number of the fold",type=int)
parser.add_argument("-N","--N_itt",help="Maximum number of iterations",type=int)
parser.add_argument("-n","--N_meas",help="Number of iterationts to check  the convergence",type=int)
parser.add_argument("--N_simu",help="Optional, simulation number to label the simulation. It will substitute the seed to name the simulation.",type=int)
parser.add_argument("-R","--Redo",help="Redo simulation if it was done before", action='store_true')
parser.add_argument("--dir_format",help="Directory format, addig information about the lambdes (lambdes) or the groups number (groups)",default="lambdes",choices=["lambdes","groups"],type=str)


args = parser.parse_args()

config_file_name = "config.yaml"

with open(config_file_name) as fp:
    data = yaml.load(fp)





def choose_params(arg,value,arg_value):
    if arg==None:
        param = value[arg_value]
    else: param = arg
    return param

def use_default(default,non_default,arg):
    try:
        if non_default[arg]!=None: return non_default[arg]
        else: return default
    except:
        return default

if args.K==None:
    K = data['nodes']['K']
else: K = args.K

if args.L==None:
    L = data['items']['L']
else: L = args.L



if args.lambda_nodes==None:
    try:
        lambda_nodes = data['nodes']['lambda_nodes']
    except: lambda_nodes = 0.0
else: lambda_nodes = args.lambda_nodes


if args.lambda_items==None:
    try:
        lambda_items = data['items']['lambda_items']
    except: lambda_items = 0.0
else: lambda_items = args.lambda_items




seed = args.seed
N_simu = args.N_simu
N_itt = choose_params(args.N_itt,data['simulation'],'N_itt')
print('N_itt',N_itt,args.N_itt)
N_measure = choose_params(args.N_meas,data['simulation'],'N_measure')
N_fold = choose_params(args.Fold,data,'N_fold')#data['N_fold']

if lambda_nodes==0.0:node_meta_data = []
else:node_meta_data =  data['nodes']['nodes_meta']

if lambda_items==0.0:
    items_meta_data = []
    Taus =  []
else:
    items_meta_data =  data['items']['items_meta']
    Taus =  data['items']['Taus']



N_meta_nodes = len(node_meta_data)
N_meta_items = len(items_meta_data)


print('file' not in data['items'])

if 'file' in data['nodes']:
    node_file_dir = data['folder']+'/'+data['nodes']['file']
else:
    node_file_dir = ''
if 'file' in data['items']:
    item_file_dir = data['folder']+'/'+data['items']['file']
else:
    item_file_dir = ''



links_base_file_dir = data['folder']+'/'+data['links']['base'].replace('{F}',str(N_fold))
links_test_file_dir = data['folder']+'/'+data['links']['test'].replace('{F}',str(N_fold))

node_header = data['nodes']['nodes_header']
item_header = data['items']['items_header']
rating_header = data['links']['rating_header']

node_separator = use_default('\t',data['nodes'],'separator')
item_separator =  use_default('\t',data['items'],'separator')
link_separator_base =  use_default('\t',data['links'],'separator_base')
link_separator_test =  use_default('\t',data['links'],'separator_test')
print('separators',link_separator_base,link_separator_test,node_separator,item_separator)
# In[7]:


if args.dir_format == 'lambdes': simu_dir_lam = 'simu_ln_{}_li_{}'.format(lambda_nodes,lambda_items)
else: simu_dir_lam = 'simu_K_{}_L_{}'.format(K,L)


if not os.path.exists(simu_dir_lam):
    try:
        os.makedirs(simu_dir_lam)
    except: pass


if N_simu==None:
    direct = simu_dir_lam+'/results_simu_s_{}_f_{}'.format(seed,N_fold)
else:
    direct = simu_dir_lam+'/results_simu_{}_f_{}'.format(N_simu,N_fold)
simu_dir = direct
print('l_nodes={}\nl_items={}\nK={}\nL={}\nfold={}\nseed={}'.format(lambda_nodes,lambda_items,K,L,N_fold,seed))


if args.Redo==False:
    if os.path.exists(direct+'/total_p.dat'):
        print('ja estava feta!!!')
        exit()
    else: print('nofeta!!!')
if not os.path.exists(simu_dir):
    os.makedirs(simu_dir)
if not os.path.exists(direct):
    os.makedirs(direct)


if sys.version_info[0] < 3:
    df_links = pd.read_csv(links_base_file_dir.format(N_fold),sep=link_separator_base.encode('utf-8'), engine='python')
    if 'file' in data['nodes']:df_nodes = pd.read_csv(node_file_dir,sep=node_separator.encode('utf-8'), engine='python')#queryodf(nodes_query, engine="IMPALA", use_cache=False, block=True)
    else:df_nodes = pd.DataFrame()#
    if 'file' in data['items']:df_items = pd.read_csv(item_file_dir,dtype={'node_id': np.int64, 'common':str},sep=item_separator.encode('utf-8'), engine='python')
    else:df_items = pd.DataFrame()#queryodf(items_query, engine="IMPALA", use_cache=False, block=True)
    links_test_df = pd.read_csv(links_test_file_dir.format(N_fold),sep=link_separator_test.encode('utf-8'), engine='python')
else:
    df_links = pd.read_csv(links_base_file_dir.format(N_fold),sep=link_separator_base, engine='python')
    if 'file' in data['nodes']:df_nodes = pd.read_csv(node_file_dir,sep=node_separator, engine='python')#queryodf(nodes_query, engine="IMPALA", use_cache=False, block=True)
    else:df_nodes = pd.DataFrame()
    print(df_nodes.head())
    print('-----',node_separator)
    if 'file' in data['items']:df_items = pd.read_csv(item_file_dir,sep=item_separator,dtype={'node_id': np.int64, 'genre_id':str}, engine='python')#queryodf(items_query, engine="IMPALA", use_cache=False, block=True)
    else:df_items = pd.DataFrame()
    links_test_df = pd.read_csv(links_test_file_dir.format(N_fold),sep=link_separator_test, engine='python')

print(links_test_df.head())

links_test = links_test_df[[node_header,item_header]].values


N_att_meta_items = []
for meta in items_meta_data:
    df_items[meta] = df_items[meta].astype('str')
    try:
        df_items[meta] = df_items[meta].str.split('|')
    except AttributeError:
        for j,l in enumerate(df_items[meta]):
            df_items[meta][j] = [df_items[meta][j]]
    #df_items[meta+"_id"] = df_items[meta+"_id"].str.split('|')
    N_att_meta_items.append(len(set(df_items[meta].values.sum())))



# In[10]:

if 'file' in data['nodes']: N_nodes = len(df_nodes)
else: N_nodes = max(df_links.max()[node_header],links_test_df.max()[node_header])+1


# In[11]:


N_links = len(df_links)


# In[12]:


if 'file' in data['items']: N_items = len(df_items)
else: N_items = max(df_links.max()[item_header],links_test_df.max()[item_header])+1


print(N_nodes,N_items)
# In[47]:


def obtain_links_arrays(node_header,item_header,lambda_nodes,lambda_items,rating_header):
    if node_meta_data != None:
        factor_meta_nodes = lambda_nodes*float(len(node_meta_data))
    else:
        factor_meta_nodes = 1.0e-16

    if items_meta_data != None:
        factor_meta_items = 1.0e-16
        for i,meta in enumerate(items_meta_data):
            factor_meta_items += lambda_items*float(N_att_meta_items[i])
    else:
        factor_meta_items = 1.0e-16

    Links_observed = len(df_links)
    veins_nodes={user:df[item_header].values for user,df in df_links.groupby(node_header)}
    print('veins nodes calculats')
    veins_items={item:df[node_header].values for item,df in df_links.groupby(item_header)}
    print('veins items calculats')
    links_by_ratings={rating:df[[node_header,item_header]].values for rating,df in df_links.groupby(rating_header)}
    print('links per rating calculats')
    links_array = df_links[[node_header,item_header]].values
    links_ratings = df_links[rating_header].values
    print('arrays calculats')

    print('pasant a arrays:')
    N_veins_nodes = []
    N_veins_items = []
    veins_items_array = []
    veins_nodes_array = []
    print('uep0',df_links.head())
    for item in range(N_items):
        if item in veins_items:
            N_veins_items.append(float(len(veins_items[item]))+1e-16+factor_meta_items)
            veins_items_array.append(veins_items[item])
        else:
            N_veins_items.append(factor_meta_items+1e-16)
            veins_items_array.append([])

    #El 2 es el número de metadatos exclusivos!! En el caso dado son edad y género. Si hago un binding de ambos es uno!!
    for node in range(N_nodes):
        if node in veins_nodes:
            N_veins_nodes.append(float(len(veins_nodes[node]))+1e-16)
            veins_nodes_array.append(veins_nodes[node])
        else:
            veins_nodes_array.append([])
            N_veins_nodes.append(1e-16)

        #PART METADADES
        if node_meta_data==None: continue
        for meta in node_meta_data:
            #print(node,meta,df_nodes[df_nodes.nodeid==node],df_nodes[df_nodes.nodeid==node][meta].values)
            if not pd.isnull(df_nodes[df_nodes[node_header]==node][meta].values[0]):
                N_veins_nodes[-1] += lambda_nodes
            #else:
            #    print('aqui!!',node)
    print('ya esta')

    #Pasem a arrays els enllaços
    N_ratings = len(links_by_ratings)
    links_by_ratings_array = [links_by_ratings[i] for i in range(N_ratings)]
    print('ya esta tot!!!!')
    return N_ratings,links_array,links_ratings,links_by_ratings_array,veins_nodes_array,veins_items_array,N_veins_nodes,N_veins_items


N_ratings,links_array,links_ratings,links_by_ratings_array,veins_nodes_array,veins_items_array,N_veins_nodes,N_veins_items = obtain_links_arrays(node_header,item_header,lambda_nodes,lambda_items,rating_header)

# In[105]:
#@vectorize([float64(float64, float64, float64)])
@jit
def any_nan(M):
    return np.any(np.isnan(M))


def obtain_meta_arrays(meta_list,id_header,observed):
    #Metadatos nodods
    Att_meta = []
    N_att_meta = []
    metas_links_arrays = []
    veins_metas = []
    veins_nodes = []
    N_veins_metas = []
    print(meta_list,id_header,observed)

    df_filtred = df_nodes[df_nodes[id_header].isin(observed)]
    #print('aqui',len(df_filtred))
    if meta_list==None:
        return Att_meta,N_att_meta,metas_links_arrays,veins_metas,N_veins_metas

    for meta in meta_list:
        Att_meta.append(list(set(df_nodes[meta][df_nodes[meta].notnull()])))
        N_att_meta.append(len(Att_meta[-1]))
        #Arrays
        metas_links_arrays.append(df_nodes[[id_header,meta]][df_nodes[meta].notnull()].astype(int).values)
        #veins_metas.append([df_filtred[id_header][df_filtred[meta]==int(att)].values for att in range(N_att_meta[-1])])
        veins_metas.append([df_nodes[id_header][df_nodes[meta]==int(att)].values for att in range(N_att_meta[-1])])
        N_veins_metas.append([float(len(arr)) for arr in veins_metas[-1]])
        veins_nodes.append(np.ones(len(df_nodes),dtype=np.int32))
        for n,att in metas_links_arrays[-1]:
            veins_nodes[-1][n] = att

    return Att_meta,N_att_meta,metas_links_arrays,veins_metas,veins_nodes,N_veins_metas

observed_nodes = np.unique(links_array[:,0])
nodes_no_observed = np.array([i for i in range(N_nodes) if i not in observed_nodes])
print('uep1 ',df_nodes.head())
#print('aqui',len(observed),len(df_nodes))
Att_meta_nodes,N_att_meta_nodes,metas_links_arrays_nodes,veins_metas_nodes,veins_nodes_metas,N_veins_metas_nodes = obtain_meta_arrays(node_meta_data,node_header,observed_nodes)


veins_items_metas = []
veins_items_metas_ones = []
veins_metas_items = []
veins_metas_items_ones = []
metas_links_arrays_items = []
metas_links_arrays_items_type = []#Label de s'enllaç 0/1
N_veins_metas_items = []
observed_items = np.unique(links_array[:,1])
items_no_observed = np.array([i for i in range(N_items) if i not in observed_items])
print('uep 2')
#category_items_inverse = {category_items[cat]:cat for cat in category_items}
for meta in range(len(items_meta_data)):
    veins_items_metas.append(np.ones((N_items,N_att_meta_items[meta]),dtype=np.int32))
    veins_items_metas[-1] *= np.arange(0,N_att_meta_items[meta])[:,np.newaxis].T
    veins_metas_items.append(np.ones((N_att_meta_items[meta],N_items),dtype=np.int32)*np.arange(0,N_items)[:,np.newaxis].T)
    veins_metas_items_ones.append([])
    veins_items_metas_ones.append([[] for j in range(N_items)])
    metas_links_arrays_items.append(np.zeros((N_att_meta_items[meta]*N_items,2),dtype=np.int32))
    N_veins_metas_items.append([N_items for i in range(N_att_meta_items[meta])])
    metas_links_arrays_items_type.append([])
    i = 0
    for g in range(N_att_meta_items[meta]):
        df_metas = df_items[df_items[items_meta_data[meta]].apply(lambda x: str(g) in x)][item_header]
        veins_metas_items_ones[-1].append(df_metas.values)
        #print(veins_metas_items_ones[-1])
        for j in range(N_items):
            metas_links_arrays_items[-1][i][0] = j
            metas_links_arrays_items[-1][i][1] = g
            if j in df_metas:
                metas_links_arrays_items_type[-1].append(1)
            else:
                metas_links_arrays_items_type[-1].append(0)
            if j in observed_items:
                veins_items_metas_ones[-1][j].append(g)
            i += 1
    metas_links_arrays_items_type[-1] = np.array(metas_links_arrays_items_type[-1])
print('ya estan los metas!!')
# In[94]:


#print(veins_metas_items)

# In[142]:


#np.seterr(all='raise')
def timer(func):
    def crono(*args):
        tic = time()
        to_return = func(*args)
        tac = time()
        print(func.__name__,'ha trigat en executarse',tac-tic)
        return to_return
    return crono


# In[143]:


def print_n_func(f):
    def func(*args):
        #f_jit = jit()(f)
        try:
            return f(*args)
        except FloatingPointError:
            print('error de FloatingPointError a',f.__name__,f.__name__=='p_kl_comp_arrays')
            if f.__name__=='p_kl_comp_arrays':
                for i,varName in zip(range(len(args)),['omega','p_kl','eta','theta','K','L']):
                    if type(args[i])==type(np.array([])):
                        print('before',varName,args[i])
                        args[i][args[i][:,:]<1e-16] =0
                        print('after',varName,args[i])
            else:
                for i in range(len(args)):
                    if type(args[i])==type(np.array([])):
                        if len(args[i].shape)==2:
                            args[i][args[i][:,:]<1e-16] =0
                        elif len(args[i].shape)==4:
                            args[i][args[i][:,:,:,:]<1e-16] =0
                        else:
                            args[i][args[i][:]<1e-16] =0
            return f(*args)
        except TypeError:
            print('error de TypeError a',f.__name__)

    return func


# In[144]:


@print_n_func
#@vectorize([float64(float64, float64, float64)])
def sum_matrix_lambda(m1,m2,l):
    return m1+l*m2


# In[145]:


#@timer
@jit(cache=True,nopython=True)
def finished(theta,theta_old,N_elements,tol):
    finished = False
    if(np.sum(np.abs(theta-theta_old))/(N_elements)<tol):
        finished = True
    return finished



# In[147]:


@print_n_func
#@timer
@jit(locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double,new_theta=double[:,:]),parallel=True)
def theta_comp_arrays(omega,theta,K,veins_nodes_array,N_veins_nodes):
    new_theta = np.array(theta)
    for i,veins in enumerate(veins_nodes_array):
        for k in prange(K):
            #theta_ik = theta[i,k]
            new_theta[i,k] = np.sum(omega[i,veins,k,:])
            new_theta[i,k] /= N_veins_nodes[i]
    return new_theta

#@print_n_func
#@timer
@jit(cache=True,locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double,new_theta=double[:,:]),parallel=True)
def theta_comp_arrays_multilayer_2(omega_metas,omega,theta,K,veins_nodes_array,N_veins_nodes,veins_metas_nodes,N_att_meta_nodes):
    new_theta = np.zeros((N_nodes,K))
    N_metas = len(N_att_meta_nodes)
    if lambda_nodes==0:
        means = []
        for meta,N_att in enumerate(N_att_meta_nodes):
            means.append(np.zeros((K,N_att)))
            for att in range(N_att):
                c = 0.0
                for k in range(K):
                    means[-1][k,att] = np.sum(theta[veins_metas_nodes[meta][att],k])/len(veins_metas_nodes[meta][att])
                    c += means[-1][k,att]
                means[-1][:,att] /= c

    for i in prange(N_nodes):
        veins_node = veins_nodes_array[i]
        if veins_node==[]:
            if lambda_nodes==0:
                for meta in range(N_metas):
                    a = veins_nodes_metas[meta][i]
                    new_theta[i,:] = means[meta][:,a]
                new_theta[i,:] = new_theta[i,:]/np.sum(new_theta[i,:])
                #print('aillat',i,new_theta[i,:])
                continue
            for meta in range(N_metas):
                new_theta[i,:] += omega_metas[meta][i,:]
            new_theta[i,:] *= lambda_nodes/N_veins_nodes[i]
        else:
            for meta in range(N_metas):
                new_theta[i,:] += lambda_nodes*omega_metas[meta][i,:]
            for k in prange(K):
                #theta_ik = theta[i,k]
                new_theta[i,k] += np.sum(omega[i,veins_node,k,:])
            new_theta[i,:] /= N_veins_nodes[i]
    return new_theta

#@print_n_func
#@timer
@jit()
def theta_comp_arrays_multilayer(omega_metas,omega,theta,K,observed_nodes,nodes_no_observed,N_veins_nodes):
    #new_theta = np.zeros((N_nodes,K))
    N_metas = len(omega_metas)
    means = np.sum(theta[observed_nodes,:],axis=0)/float(len(observed_nodes))
    #means /= means#.sum()
    new_theta = omega.sum(axis=1).sum(axis=2)
    for meta in range(N_metas):
        new_theta += omega_nodes[meta].sum(axis=1)*lambda_nodes
    new_theta /= N_veins_nodes
    if lambda_nodes == 0:new_theta[nodes_no_observed] = means
    return new_theta


#@print_n_func
#@timer
#@jit(locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double,new_theta=double[:,:]),parallel=True)
def theta_comp_arrays_exclusive(omega,theta,K,links_array,veins_nodes_array,N_veins_metas_nodes,veins_metas_nodes,veins_nodes_metas,N_att,N_veins_nodes):
    new_theta = np.zeros((N_nodes,K))
    if lambda_nodes==0:
        means = np.zeros((K,N_att))
        for att in range(N_att):
            c = 0.0
            for k in range(K):
                means[k,att] = np.sum(theta[veins_metas_nodes[att],k])/len(veins_metas_nodes[att])
                c += means[k,att]
            means[:,att] /= c
    for link  in prange(len(links_array)):
        i = links_array[link][0]
        a = links_array[link][1]
        if lambda_nodes==0 and veins_nodes_array[i]==[]:
            new_theta[i,:] = means[:,a]
            #print('aillat',i,new_theta[i,:])
            continue
        for k in prange(K):
            new_theta[i,k] = omega[i,k]
        new_theta[i,:] /= N_veins_nodes[i]
        #if any_nan(new_theta[i,:]):
        #    print('no aillat',i,omega[i,k])
    #if any_nan(new_theta):exit()
    return new_theta


#@print_n_func
#@timer
#@jit(cache=True,parallel=True)
def eta_multilayer(eta,omega,omega_flat,N_veins_items,lambda_items,L,Taus,items_no_observed,N_att_meta_items):
    new_eta = np.zeros((N_items,L))
    N_metas = len(N_att_meta_items)
    means = np.sum(eta[observed_items,:],axis=0)/len(observed_items)

    new_eta += omega.sum(axis=0).sum(axis=1)

    start = 0
    for meta,n in enumerate(N_att_meta_items):
        omega_meta = omega_flat[start:start+n*L*N_items*Taus[meta]].reshape(N_items,n,L,Taus[meta])
        new_eta += omega_meta.sum(axis=1).sum(axis=2)*lambda_items
        start += n*L*N_items*Taus[meta]
    for l in range(L):
        new_eta[:,l] /= N_veins_items
    if lambda_items==0:
        new_eta[items_no_observed] = means
    return new_eta

# In[6]:


#@print_n_func
#@timer
@jit(cache=True,parallel=True)
def eta_multilayer_2(eta,omega,omega_items,veins_items_array,L,N_veins_items,lambda_items,veins_metas_items,veins_items_metas,veins_items_metas_ones,N_att_meta_items):
    new_eta = np.zeros((N_items,L))
    N_metas = len(N_att_meta_items)
    means = []
    if lambda_items==0:
        for meta,N_att in enumerate(N_att_meta_items):
            means.append(np.zeros((L,N_att)))
            for att in range(N_att):
                c = 0.0
                for l in range(L):
                    means[-1][l,att] = np.sum(eta[veins_metas_items[meta][att],l])/len(veins_metas_items[meta][att])
                    c += means[-1][l,att]
                means[-1][:,att] /= c



    for j in prange(N_items):
        veins = veins_items_array[j]
        if veins==[]:
            if lambda_items==0:
                for meta in range(N_metas):
                    a = veins_items_metas[meta][j]
                    new_eta[j,:] = means[meta][:,a]
                new_eta[j,:] = new_eta[j,:]/np.sum(new_eta[j,:])
                #print('aillat',i,new_theta[i,:])
                continue
            #for k in prange(K):
            for meta,omega_meta in enumerate(omega_items):
                meta_veins = veins_items_metas[meta][j]
                #print('------------>',j,omega_meta[j,meta_veins,l,:],np.sum(omega_meta[j,meta_veins,l,:]))
                #raw_input()
                new_eta[j,:] += np.sum(omega_meta[j,meta_veins,:,:],axis=(0,2))#*lambda_items
            new_eta[j,:] *= lambda_items/N_veins_items[j]
            #print('------------>2',new_eta[j,:],lambda_items,N_veins_items[j],lambda_items/N_veins_items[j])
        else:
            for l in prange(L):
                for meta,omega_meta in enumerate(omega_items):
                    meta_veins = veins_items_metas[meta][j]
                    new_eta[j,l] += np.sum(omega_meta[j,meta_veins,l,:])*lambda_items
                    #print(j,np.sum(omega_meta[j,meta_veins,l,:])*lambda_items/N_veins_items[j])
                        #raw_input()
                new_eta[j,l] += np.sum(omega[veins,j,:,l])
            new_eta[j,:] /= N_veins_items[j]
            #print(j,new_eta[j,:])
            #raw_input()

    return new_eta

# In[6]:


@print_n_func
#@timer
@jit(cache=True,locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double,new_eta=double[:,:]))
def eta_comp_arrays(omega,eta,L,veins_items_array,N_veins_items):
    new_eta = np.array(eta)
    for j,veins in enumerate(veins_items_array):
        for l in range(L):
            #eta_jl = eta[j,l]
            new_eta[j,l] = np.sum(omega[veins,j,:,l])
            new_eta[j,l] /= N_veins_items[j]
    return new_eta


# In[150]:


# In[7]:


@print_n_func
#@timer
@jit(cache=True,nopython=True,parallel=True)
def p_kl_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings):
    p_kl[:,:,:] = 0
    for k in range(K):
        for l in prange(L):
            for link  in prange(len(links_ratings)):
                i = links_array[link][0]
                j = links_array[link][1]
                rating = links_ratings[link]
                p_kl[k,l,rating] += omega[i,j,k,l]
            suma = np.sum(p_kl[k,l,:])
            p_kl[k,l,:] /= (suma+1e-16)
    return p_kl


# In[151]:


# In[20]:


@print_n_func
#@timer
@jit(cache=True,nopython=True,parallel=True)
def q_ka_comp_arrays(omega,q_ka,K,links_array,att_elements):
    q_ka2 = np.zeros((K,len(att_elements)))
    s = np.zeros(K)
    for link  in range(len(links_array)):
        i = links_array[link][0]
        a = links_array[link][1]
        for k in range(K):
            q_ka2[k,a] += omega[i,a,k]#/(att_elements[a]+1.0e-16)
            s[k] += omega[i,a,k]
    q_ka2 /= np.expand_dims(s,axis=1)+1.0e-16
    #q_ka2 /= s[:,np.newaxis]
    return q_ka2


# In[152]:


# In[9]:


#@print_n_func
#@timer
#@jit(cache=True,nopython=True,locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double),parallel=True)
@jit(nopython=True,locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double))
def omega_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings):
    #new_omega = np.array(omega)
    for link  in range(len(links_ratings)):
        i = links_array[link][0]
        j = links_array[link][1]
        rating = links_ratings[link]
        omega[i,j,:,:] = p_kl[:,:,rating]*(np.expand_dims(theta[i,:], axis=1)@np.expand_dims(eta[j,:],axis=0))
        suma = omega[i,j,:,:].sum()
        omega[i,j,:,:] /= suma+1e-16
    return omega




# In[117]:


# In[10]:


@print_n_func
#@timer
#@jit(nopython=True,locals=dict(i=int64,a=int64,k=int64,link=int64,suma=double),parallel=True)
@jit(nopython=True,locals=dict(i=int64,a=int64,k=int64,link=int64,suma=double))
def omega_comp_arrays_exclusive(omega,q_ka,theta,N_nodes,N_att_meta):
    o = np.zeros(omega.shape)
    for i in range(N_nodes):
        for a in range(int(N_att_meta)):
            o[i,a,:] = theta[i,:]*q_ka[:,a]
    s = o.sum(axis=2)+1e-16
    o /= np.expand_dims(s, axis=2)
    return o

@jit(nopython=True,locals=dict(i=int64,a=int64,k=int64,link=int64,suma=double),parallel=True)
def omega_comp_arrays_exclusive(omega,q_ka,theta,N_nodes,metas_links_arrays_nodes):
    for j in prange(len(metas_links_arrays_nodes)):
        i = metas_links_arrays_nodes[j,0]
        a = metas_links_arrays_nodes[j,1]
        s = 0
        for k in range(K):
            omega[i,a,k] = theta[i,k]*q_ka[k,a]
            s +=omega[i,a,k]
        omega[i,a,:] /= s
    return omega

@print_n_func
#@timer
@jit(nopython=True,parallel=True)
def total_p_comp_test(N_nodes,N_items,N_ratings,K,L,theta,eta,p_kl,test):
    total_p = np.zeros((len(test),N_ratings))
    for n in prange(len(test)):
        i = test[n,0]
        j = test[n,1]
        for r in range(N_ratings):
            suma = 0
            for k in range(K):
                for l in range(L):
                    suma += theta[i,k]*eta[j,l]*p_kl[k,l,r]
            total_p[n,r] += suma
    return total_p



# In[85]:


# return Att_meta_nodes,N_att_meta_nodes,metas_links_arrays,veins_metas,N_veins_metas
def inicialitzacio(K,L,Taus,N_nodes,N_items,N_ratings,N_att_meta_nodes,N_att_meta_items,links_array,links_ratings,metas_links_arrays_nodes,metas_links_arrays_items):
    #Inicialitzacio

    ##Definim matrius
    theta  = np.random.rand(N_nodes,K)
    eta = np.random.rand(N_items,L)
    p_kl = np.random.rand(K,L,N_ratings)


    suma = np.sum(theta,axis =1)
    theta /= suma[:,np.newaxis]
    suma = np.sum(eta,axis=1)
    eta /= suma[:,np.newaxis]
    suma = np.sum(p_kl,axis =2)
    p_kl /=suma[:,:,np.newaxis]

    omega = np.zeros((N_nodes,N_items,K,L),dtype=np.double)
    omega = omega_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings)

    q_l_taus = []
    zetes = []
    omega_items = []
    for i in range(len(items_meta_data)):
        Tau = Taus[i]
        zetes.append(np.random.rand(N_att_meta_items[i],Tau))
        suma = np.sum(zetes[-1],axis=1)
        zetes[-1] /= suma[:,np.newaxis]

        q_l_taus.append(np.random.rand(L,Tau,N_att_meta_items[i]))
        suma = np.sum(q_l_taus[-1],axis =2)
        q_l_taus[-1] /= suma[:,:,np.newaxis]

        omega_items.append(np.zeros((N_items,N_att_meta_items[i],L,Tau),dtype=np.double))
        omega_items[-1] = omega_comp_arrays(omega_items[-1],q_l_taus[-1],zetes[-1],eta,L,Tau,metas_links_arrays_items[i],metas_links_arrays_items_type[i])

    q_kas = []
    omega_nodes = []
    for meta,N in enumerate(N_att_meta_nodes):
        q_kas.append(np.random.rand(K,N))

        ##Normalitzem
        suma = np.sum(q_kas[-1],axis =1)
        q_kas[-1] /=suma[:,np.newaxis]

        omega_nodes.append(np.zeros((N_nodes,N_att_meta_nodes[-1],K),dtype=np.double))
        omega_nodes[-1] = omega_comp_arrays_exclusive(omega_nodes[-1],q_kas[-1],theta,N_nodes,metas_links_arrays_nodes[-1])

    #omega_comp_arrays.inspect_types()
    '''simu_dir = "input_matrix"
    np.savetxt(simu_dir+'/theta.dat'.format(N_run),theta)
    np.savetxt(simu_dir+'/eta.dat'.format(N_run),eta)

    #np.savetxt(simu_dir+'/eta_{}.dat'.format(N_run),eta)
    for r in range(N_ratings):
        np.savetxt(simu_dir+'/pkl_{}.dat'.format(r),p_kl[:,:,r])


    for meta in range(len(items_meta_data)):
        for r in range(2):
            np.savetxt(simu_dir+'/qlT_{}_{}.dat'.format(r,items_meta_data[meta]),q_l_taus[meta][:,:,r])

        np.savetxt(simu_dir+'/zeta_{}.dat'.format(items_meta_data[meta]),zetes[meta])

    for meta in range(len(N_att_meta_items)):
        np.savetxt(simu_dir+'/q_ka_{}.dat'.format(node_meta_data[meta]),q_kas[meta])'''

    #print(theta,eta,p_kl,omega,q_kas,omega_nodes,zetes,q_l_taus,omega_items)
    return theta,eta,p_kl,omega,q_kas,omega_nodes,zetes,q_l_taus,omega_items


# In[118]:


@print_n_func
#@timer
@jit(nopython=True,parallel=True,locals=dict(i=int64,j=int64,k=int64,l=int64,rating=int64,link=int64,suma=double))
def log_like_comp_arrays(p_kl,eta,theta,K,L,links_array,links_ratings):
    log_like = 0
    for link  in prange(len(links_ratings)):
        i = links_array[link][0]
        j = links_array[link][1]
        rating = links_ratings[link]
        suma = 0
        for k in range(K):
            for l in range(L):
                suma += theta[i,k]*eta[j,l]*p_kl[k,l,rating]
        log_like += np.log(suma)
    return log_like


# In[119]:



@print_n_func
#@timer
@jit(nopython=True,parallel=True)
def log_like_comp_arrays_exclusive(theta,q_ka,K,links_array):
    log_like = 0
    for link  in range(len(links_array)):
        i = links_array[link][0]
        a = links_array[link][1]
        suma = 0
        for k in range(K):
            suma += theta[i,k]*q_ka[k,a]
        #if suma<1.0e-16:
        #    print(i,veins_nodes_array[i],theta[i,:],q_ka[:,a])
        log_like += np.log(suma)
    return log_like



def load_matrix_simu(dir_matrix,K,L,N_ratings,items_meta_data,Taus,node_meta_data,N_att_meta_items):
    eta = np.loadtxt('{}/eta.dat'.format(dir_matrix))
    theta = np.loadtxt('{}/theta.dat'.format(dir_matrix))

    p_kl = np.zeros((K,L,N_ratings))
    for r in range(N_ratings):
        p_kl[:,:,r] = np.loadtxt('{}/pkl_{}.dat'.format(dir_matrix,r))


    omega = np.zeros((N_nodes,N_items,K,L),dtype=np.double)
    omega = omega_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings)

    q_l_taus = []
    zetes = []
    q_kas = []
    omega_items = []
    for meta in range(len(items_meta_data)):
        q_l_taus.append(np.zeros((L,Taus[meta],N_att_meta_items[meta])))
        for r in range(2):
            q_l_taus[-1][:,:,r] = np.loadtxt('{}/qlT_{}_{}.dat'.format(dir_matrix,r,items_meta_data[meta]))

        zetes.append(np.loadtxt('{}/zeta_{}.dat'.format(dir_matrix,items_meta_data[meta])))
        omega_items.append(np.zeros((N_items,N_att_meta_items[meta],L,Taus[meta]),dtype=np.double))
        omega_items[-1] = omega_comp_arrays(omega_items[-1],q_l_taus[-1],zetes[-1],eta,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])


    omega_nodes = []
    for meta in range(len(node_meta_data)):
        q_kas.append(np.loadtxt('{}/q_ka_{}.dat'.format(dir_matrix,node_meta_data[meta])))

        omega_nodes.append(np.zeros((N_nodes,K),dtype=np.double))
        omega_nodes[-1] = omega_comp_arrays_exclusive(omega_nodes[-1],q_kas[-1],theta,K,metas_links_arrays_nodes[meta])

    return eta,theta,p_kl,q_l_taus,zetes,q_kas,omega,omega_items,omega_nodes

# In[132]:
print('ini')


N_ratings,links_array,links_ratings,links_by_ratings_array,veins_nodes_array,veins_items_array,N_veins_nodes,N_veins_items = obtain_links_arrays(node_header,item_header,lambda_nodes,lambda_items,rating_header)
# In[ ]:

N_run = 'prueba/'
direct = '.'


#BUCLE AQUI

if seed!=None:
    np.random.seed(int(seed))
theta,eta,p_kl,omega,q_kas,omega_nodes,zetes,q_l_taus,omega_items = inicialitzacio(K,L,Taus,N_nodes,N_items,N_ratings,N_att_meta_nodes,N_att_meta_items,links_array,links_ratings,metas_links_arrays_nodes,metas_links_arrays_items)

print('UEP!!!')
# In[44]:

## date and time representation
file_info = open(simu_dir+'/info_simus.info','w')
file_info.write("Simulation started at:" + time_lib.strftime("%c")+'\n\n')
file_info.write('With parameters:\nK={}\nL={}\nN_nodes={}\nN_items={}\nN_ratings={}\nLinks_observed={}\nSeed={}\n\n############################\n'.format(K,L,N_nodes,N_items,N_ratings,N_links,seed))
file_info.write('Prior metadatas of nodes:\n')
if node_meta_data != None:
    for meta in node_meta_data:
        file_info.write('\t{}\n'.format(meta))
file_info.write('Prior metadatas of items:\n')
if items_meta_data != None:
    for meta in items_meta_data:
        file_info.write('\t{}\n'.format(meta))
file_info.write('Prior coupling constants:\n')
file_info.write('\tNodes: {}\n'.format(lambda_nodes))


file_info.write('-Items categories:\n')

file_info.write('\tItems: {}\n'.format(lambda_items))
file_info.close()

tik_simu = time()

#theta,eta,p_kl,omega,q_ka_ages,omega_ages,q_ka_genders,omega_genders,zeta,q_l_tau,omega_genres = inicialitzacio(K,L,Tau,N_nodes,N_items,N_ratings,N_ages,N_genres,N_genders,links_array,links_ratings,genre_link_array,age_link_array,gender_link_array,link_genre)
#eta,theta,p_kl,q_l_taus,zetes,q_kas,omega,omega_items,omega_nodes = load_matrix_simu('input_matrix',K,L,N_ratings,items_meta_data,Taus,node_meta_data,N_att_meta_items)
file_logLike = open(simu_dir+'/log_evolution.dat'.format(N_run),'w')
old_log_like = 0.0
old_log_like += log_like_comp_arrays(p_kl,eta,theta,K,L,links_array,links_ratings)

for meta in range(len(node_meta_data)):
    old_log_like += lambda_nodes*log_like_comp_arrays_exclusive(theta,q_kas[meta],K,metas_links_arrays_nodes[meta])

for meta in range(len(items_meta_data)):
    old_log_like += lambda_items*log_like_comp_arrays(q_l_taus[meta],zetes[meta],eta,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])
# In[109]:

print('copy')
theta_old = theta.copy()
eta_old = eta.copy()
theta_temp = theta.copy()
eta_temp = eta.copy()
p_kl_old = p_kl.copy()
zetes_old = deepcopy(zetes)
q_kas_old = deepcopy(q_kas)
q_l_taus_old = deepcopy(q_l_taus)


# In[ ]:

print('simu',N_itt,N_measure)
for itt in range(N_itt):
    theta = theta_comp_arrays_multilayer(omega_nodes,omega,theta,K,observed_nodes,nodes_no_observed,np.array(N_veins_nodes)[:,np.newaxis])

    for meta in range(len(node_meta_data)):
        q_kas[meta] = q_ka_comp_arrays(omega_nodes[meta],q_kas[meta],K,metas_links_arrays_nodes[meta],N_veins_metas_nodes[meta])

        #print(itt,q_kas[meta].sum())
        omega_nodes[meta] = omega_comp_arrays_exclusive(omega_nodes[meta],q_kas[meta],theta,N_nodes,metas_links_arrays_nodes[meta])
    '''if itt>-1:
        print(itt)
        #print(theta)
        tmp = theta_comp_arrays_exclusive(omega_ages,theta,K,age_link_array,N_veins_nodes)
        for i in tmp:
            print(i)'''
    N_om = 0
    for meta,n in enumerate(N_att_meta_items):
        N_om += n*L*N_items*Taus[meta]
    omega_items_flated = np.zeros(N_om)

    start = 0
    for meta,o in enumerate(omega_items):
        omega_items_flated[start:N_att_meta_items[meta]*L*N_items*Taus[meta]+start] = o.flatten()
        start += N_att_meta_items[meta]*L*N_items
    eta = eta_multilayer(eta,omega,omega_items_flated,N_veins_items,lambda_items,L,Taus,items_no_observed,N_att_meta_items)

    for meta in range(len(items_meta_data)):
        #eta2 = lambda_items*theta_comp_arrays(omega_items[meta],eta,Taus[meta],veins_items_metas[meta],N_veins_items)
        #eta = sum_matrix_lambda(eta,theta_comp_arrays(omega_items[meta],eta,Taus[meta],veins_items_metas[meta],N_veins_items),lambda_items)
        zetes[meta] = eta_comp_arrays(omega_items[meta],zetes[meta],Taus[meta],veins_metas_items[meta],N_veins_metas_items[meta])
        q_l_taus[meta] = p_kl_comp_arrays(omega_items[meta],q_l_taus[meta],zetes[meta],eta_temp,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])
        omega_items[meta] = omega_comp_arrays(omega_items[meta],q_l_taus[meta],zetes[meta],eta,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])

    p_kl = p_kl_comp_arrays(omega,p_kl,eta_temp,theta_temp,K,L,links_array,links_ratings)


    omega = omega_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings)
    #print(itt)

    #theta_temp = theta.copy()
    eta_temp = eta.copy()
    if itt%N_measure==0:
        log_like = 0.0
        log_like += log_like_comp_arrays(p_kl,eta,theta,K,L,links_array,links_ratings)
        for meta in range(len(node_meta_data)):
            log_like += lambda_nodes*log_like_comp_arrays_exclusive(theta,q_kas[meta],K,metas_links_arrays_nodes[meta])
        for meta in range(len(items_meta_data)):
            log_like += lambda_items*log_like_comp_arrays(q_l_taus[meta],zetes[meta],eta,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])
        variation = old_log_like-log_like

        file_logLike.write('{}\t{}\t{}\n'.format(itt,log_like,variation))

        #else:
        if finished(theta,theta_old,K*N_nodes,0.0001):
            if finished(eta,eta_old,L*N_items,0.0001):
                out = False
                for meta in range(len(node_meta_data)):
                    if finished(q_kas[meta],q_kas_old[meta],K*N_att_meta_nodes[meta],0.0001):
                        out = False
                    else:
                        out = True
                        break
                if not out:
                    out2 = False
                    for meta in range(len(items_meta_data)):
                        if finished(zetes[meta],zetes_old[meta],L*N_att_meta_items[meta],0.0001):
                            out2 = False
                        else:
                            out2 = True
                            break
                    if not out2:
                        if finished(p_kl,p_kl_old,L*K*N_ratings,0.0001):
                            out3 = False
                            for meta in range(len(items_meta_data)):
                                if finished(q_l_taus[meta],q_l_taus_old[meta],L*2*Taus[meta],0.0001):
                                    out3 = False
                                else:
                                    out3 = True
                                    break
                            if not out3:
                                print('------------------HA acabat!!!',itt,log_like)
                                break

        theta_old = theta.copy()
        eta_old = eta.copy()
        p_kl_old = p_kl.copy()
        zetes_old = deepcopy(zetes)
        q_kas_old = deepcopy(q_kas)
        q_l_taus_old = deepcopy(q_l_taus)
        #print(itt,time()-tik)
file_logLike.close()
np.savetxt(simu_dir+'/theta.dat'.format(N_run),theta)
np.savetxt(simu_dir+'/eta.dat'.format(N_run),eta)

#np.savetxt(simu_dir+'/eta_{}.dat'.format(N_run),eta)
total_p = total_p_comp_test(N_nodes,N_items,N_ratings,K,L,theta,eta,p_kl,links_test)
for r in range(N_ratings):
    np.savetxt(simu_dir+'/pkl_{}.dat'.format(r),p_kl[:,:,r])


for meta in range(len(items_meta_data)):
    for r in range(2):
        np.savetxt(simu_dir+'/qlT_{}_{}.dat'.format(r,items_meta_data[meta]),q_l_taus[meta][:,:,r])

    np.savetxt(simu_dir+'/zeta_{}.dat'.format(items_meta_data[meta]),zetes[meta])

for meta in range(len(node_meta_data)):
    np.savetxt(simu_dir+'/q_ka_{}.dat'.format(node_meta_data[meta]),q_kas[meta])

total = open(simu_dir+'/total_p.dat'.format(N_run),'w')
header ='i\tj'
string_format = '{}\t{}\t'
for r in range(N_ratings):
    header+='\tp_rij={}'.format(r)
    string_format += '{}\t'
string_format=string_format[:-1]+'\n'
total.write(header+'\n')
items_index = np.array(range(N_items))
n = 0
for i,j in links_test:
    t = tuple([i,j]+list(total_p[n,:]))
    n+=1
    total.write(string_format.format(*t))
total.close()

#exit()
myfile = open(simu_dir+'/info_simus_end.info','w')# as myfile:
myfile.write("Simulation finished at:" + time_lib.strftime("%c")+'\n')
myfile.write("Simulation took:" + str(time()-tik_simu)+'s\n')
myfile.write("Simulation took:" + str(itt+1)+' iterations\n')
myfile.close()
print("Simulation took:" + str(time()-super_tak)+'s')
