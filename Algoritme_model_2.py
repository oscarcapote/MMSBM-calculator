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
parser.add_argument("-F","--Fold",help="Number of the fold",type=int, default=3000)
parser.add_argument("-N","--N_itt",help="Maximum number of iterations",type=int)
parser.add_argument("-n","--N_meas",help="Number of iterationts to check  the convergence",type=int)
parser.add_argument("-R","--Redo",help="Redo simulation if it was done before", action='store_true')


args = parser.parse_args()

config_file_name = "config.yaml"

with open(config_file_name) as fp:
    data = yaml.load(fp)





def choose_params(arg,value,arg_value):
    if arg==None:
        param = value[arg_value]
    else: param = arg
    return param


if args.K==None:
    K = data['nodes']['K']
else: K = args.K

if args.L==None:
    L = data['items']['L']
else: L = args.L



if args.lambda_nodes==None:
    lambda_nodes = data['nodes']['lambda_nodes']
else: lambda_nodes = args.lambda_nodes
if args.lambda_items==None:
    lambda_items = data['items']['lambda_items']
else: lambda_items = args.lambda_items


seed = args.seed
N_itt = choose_params(args.N_itt,data['simulation'],'N_itt')
print('N_itt',N_itt,args.N_itt)
N_measure = choose_params(args.N_meas,data['simulation'],'N_measure')
N_fold = choose_params(args.Fold,data,'N_fold')#data['N_fold']

node_meta_data =  data['nodes']['nodes_meta']
items_meta_data =  data['items']['items_meta']
Taus =  data['items']['Taus']
N_meta_nodes = len(node_meta_data)
N_meta_items = len(items_meta_data)


node_file_dir = data['folder']+'/'+data['nodes']['file']
item_file_dir = data['folder']+'/'+data['items']['file']
links_base_file_dir = data['folder']+'/'+data['links']['base'].replace('{F}',str(N_fold))
links_test_file_dir = data['folder']+'/'+data['links']['test'].replace('{F}',str(N_fold))

node_header = data['nodes']['nodes_header']
item_header = data['items']['items_header']
rating_header = data['links']['rating_header']

# In[7]:



df_links = pd.read_csv(links_base_file_dir.format(N_fold),sep='\t', engine='python')
# In[10]:

# In[ ]:


df_nodes = pd.read_csv(node_file_dir,sep='\t', engine='python')#queryodf(nodes_query, engine="IMPALA", use_cache=False, block=True)



# In[9]:


df_items = pd.read_csv(item_file_dir,sep='\t', dtype={'node_id': np.int64, 'genre_id':str}, engine='python')#queryodf(items_query, engine="IMPALA", use_cache=False, block=True)


links_test_df = pd.read_csv(links_base_file_dir.format(N_fold),sep='\t', engine='python')
links_test = links_test_df[[node_header,item_header]].values


N_att_meta_items = []
for meta in items_meta_data:
    try:
        df_items[meta] = df_items[meta].str.split('|')
    except AttributeError:
        for j,l in enumerate(df_items[meta]):
            df_items[meta][j] = [df_items[meta][j]]
    #df_items[meta+"_id"] = df_items[meta+"_id"].str.split('|')
    N_att_meta_items.append(len(set(df_items[meta].values.sum())))




# In[10]:


N_nodes = len(df_nodes)


# In[11]:


N_links = len(df_links)


# In[12]:


N_items = len(df_items)


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



def obtain_meta_arrays(meta_list,id_header):
    #Metadatos nodods
    Att_meta = []
    N_att_meta = []
    metas_links_arrays = []
    veins_metas = []
    veins_nodes = []
    N_veins_metas = []

    if meta_list==None:
        return Att_meta,N_att_meta,metas_links_arrays,veins_metas,N_veins_metas

    for meta in meta_list:
        Att_meta.append(list(set(df_nodes[meta][df_nodes[meta].notnull()])))
        N_att_meta.append(len(Att_meta[-1]))

        #Arrays
        metas_links_arrays.append(df_nodes[[id_header,meta]][df_nodes[meta].notnull()].astype(int).values)
        veins_metas.append([df_nodes[id_header][df_nodes[meta]==int(att)].values for att in range(N_att_meta[-1])])
        N_veins_metas.append([float(len(arr)) for arr in veins_metas[-1]])
        veins_nodes.append(np.ones(len(df_nodes),dtype=np.int32))
        for n,att in metas_links_arrays[-1]:
        #    for i in:
            veins_nodes[-1][n] = att

    return Att_meta,N_att_meta,metas_links_arrays,veins_metas,veins_nodes,N_veins_metas

Att_meta_nodes,N_att_meta_nodes,metas_links_arrays_nodes,veins_metas_nodes,veins_nodes_metas,N_veins_metas_nodes = obtain_meta_arrays(node_meta_data,node_header)


veins_items_metas = []
veins_metas_items = []
metas_links_arrays_items = []
metas_links_arrays_items_type = []#Label de s'enllaç 0/1
N_veins_metas_items = []
#category_items_inverse = {category_items[cat]:cat for cat in category_items}
for meta in range(len(items_meta_data)):
    veins_items_metas.append(np.ones((N_items,N_att_meta_items[meta]),dtype=np.int32))
    veins_items_metas[-1] *= np.arange(0,N_att_meta_items[meta])[:,np.newaxis].T
    veins_metas_items.append(np.ones((N_att_meta_items[meta],N_items),dtype=np.int32)*np.arange(0,N_items)[:,np.newaxis].T)
    metas_links_arrays_items.append(np.zeros((N_att_meta_items[meta]*N_items,2),dtype=np.int32))
    N_veins_metas_items.append([N_items for i in range(N_att_meta_items[meta])])
    metas_links_arrays_items_type.append([])
    i = 0
    for g in range(N_att_meta_items[meta]):
        df_metas = df_items[df_items[items_meta_data[meta]].apply(lambda x: str(g) in x)][item_header]
        for j in range(N_items):
            metas_links_arrays_items[-1][i][0] = j
            metas_links_arrays_items[-1][i][1] = g
            if j in df_metas:
                metas_links_arrays_items_type[-1].append(1)
            else:
                metas_links_arrays_items_type[-1].append(0)
            i += 1
    metas_links_arrays_items_type[-1] = np.array(metas_links_arrays_items_type[-1])
print('ya estan los metas!!')
# In[94]:




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
@jit(nopython=True)
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


# In[148]:


# In[5]:


#@print_n_func
#@timer
#@jit(locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double,new_theta=double[:,:]),parallel=True)
def theta_comp_arrays_exclusive(omega,theta,K,links_array,N_veins_metas_nodes,veins_metas_nodes,N_att,N_veins_nodes):
    new_theta = np.zeros((N_nodes,K))
    if lambda_nodes==0:
        means = np.zeros((K,N_att))
        for att in range(N_att):
            c = 0.0
            for k in range(K):
                means[k,att] = np.sum(theta[veins_metas_nodes[att],k])/len(veins_metas_nodes[att])
                c += means[k,att]
            means[:,att] /= c
        print(means.sum(axis=0))
    for link  in prange(len(links_array)):
        i = links_array[link][0]
        a = links_array[link][1]
        for k in prange(K):
            new_theta[i,k] = omega[i,k]
        new_theta[i,:] /= N_veins_nodes[i]
            #print(omega[i,k],N_veins_nodes[i],omega[i,k]/N_veins_nodes[i],new_theta[i,k])
    return new_theta



@print_n_func
#@timer
@jit(parallel=True)
def eta_multilayer(eta,omega,omega_items,veins_items_array,L,N_veins_items,lambda_items):
    new_eta = np.array(eta)
    for j,veins in enumerate(veins_items_array):
        for l in prange(L):
            new_eta[j,l] = 0.0e0
            for meta,omega_meta in enumerate(omega_items):
                meta_veins = veins_items_metas[meta][j]
                new_eta[j,l] += np.sum(omega_meta[j,meta_veins,l,:])*lambda_items
                    #raw_input()
            new_eta[j,l] += np.sum(omega[veins,j,:,l])
        new_eta[j,:] /= N_veins_items[j]
    return new_eta

# In[6]:


@print_n_func
#@timer
@jit(locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double,new_eta=double[:,:]))
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
@jit(nopython=True,parallel=True)
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
@jit(nopython=True,parallel=True)
def q_ka_comp_arrays(omega,q_ka,K,links_array,att_elements):
    q_ka[:,:] = 0
    for k in prange(K):
        for link  in range(len(links_array)):
            i = links_array[link][0]
            a = links_array[link][1]
            q_ka[k,a] += omega[i,k]

        #suma = np.sum(q_ka[k,a,:])
        q_ka[k,:] /= att_elements[a]
    return q_ka


# In[152]:


# In[9]:


@print_n_func
#@timer
@jit(nopython=True,locals=dict(i=int64,j=int64,k=int64,l=int64,suma=double),parallel=True)
def omega_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings):
    #new_omega = np.array(omega)
    for link  in range(len(links_ratings)):
        i = links_array[link][0]
        j = links_array[link][1]
        rating = links_ratings[link]
        suma = 0
        for k in range(K):
            for l in range(L):
                omega[i,j,k,l] = p_kl[k,l,rating]*theta[i,k]*eta[j,l]
                #else:print(p_kl[k,l,rating],theta[i,k],eta[j,l])
                suma += omega[i,j,k,l]
        omega[i,j,:,:] /= suma+1e-16
    return omega




# In[117]:


# In[10]:


@print_n_func
#@timer
@jit(nopython=True,locals=dict(i=int64,a=int64,k=int64,link=int64,suma=double),parallel=True)
def omega_comp_arrays_exclusive(omega,q_ka,theta,K,links_array):
    for link  in prange(len(links_array)):
        i = links_array[link][0]
        a = links_array[link][1]
        suma = 0.0
        for k in range(K):
            omega[i,k] = q_ka[k,a]*theta[i,k]
            suma += omega[i,k]
        omega[i,:] /= suma
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
    for i,Tau in enumerate(Taus):
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

        omega_nodes.append(np.zeros((N_nodes,K),dtype=np.double))
        omega_nodes[-1] = omega_comp_arrays_exclusive(omega_nodes[-1],q_kas[-1],theta,K,metas_links_arrays_nodes[meta])

    #omega_comp_arrays.inspect_types()
    '''simu_dir = "input_matrix"
    np.savetxt(simu_dir+'/theta.dat'.format(N_run),theta)
    np.savetxt(simu_dir+'/eta.dat'.format(N_run),eta)

    #np.savetxt(simu_dir+'/eta_{}.dat'.format(N_run),eta)
    for r in range(N_ratings):
        np.savetxt(simu_dir+'/pkl_{}.dat'.format(r),p_kl[:,:,r])


    for meta in range(len(Taus)):
        for r in range(2):
            np.savetxt(simu_dir+'/qlT_{}_{}.dat'.format(r,items_meta_data[meta]),q_l_taus[meta][:,:,r])

        np.savetxt(simu_dir+'/zeta_{}.dat'.format(items_meta_data[meta]),zetes[meta])

    for meta in range(len(N_att_meta_items)):
        np.savetxt(simu_dir+'/q_ka_{}.dat'.format(node_meta_data[meta]),q_kas[meta])'''


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
    for meta in range(len(Taus)):
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
simu_dir_lam = 'simu_ln_{}_li_{}'.format(lambda_nodes,lambda_items)

if not os.path.exists(simu_dir_lam):
    os.makedirs(simu_dir_lam)

# In[ ]:


N_run = 'prueba/'
direct = '.'

direct = simu_dir_lam+'/results_simu_s_{}_f_{}'.format(seed,N_fold)
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



if seed!=None:
    np.random.seed(int(seed))
theta,eta,p_kl,omega,q_kas,omega_nodes,zetes,q_l_taus,omega_items = inicialitzacio(K,L,Taus,N_nodes,N_items,N_ratings,N_att_meta_nodes,N_att_meta_items,links_array,links_ratings,metas_links_arrays_nodes,metas_links_arrays_items)


# In[44]:

## date and time representation
file_info = open(simu_dir+'/info_simus.info','w')
file_info.write("Simulation started at:" + time_lib.strftime("%c")+'\n\n')
file_info.write('With parameters:\nK={}\nL={}\nN_nodes={}\nN_items={}\nN_ratings={}\nLinks_observed={}\n\n############################\n'.format(K,L,N_nodes,N_items,N_ratings,N_links))
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
file_logLike = open(direct+'/log_evolution.dat'.format(N_run),'w')
old_log_like = 0.0
old_log_like += log_like_comp_arrays(p_kl,eta,theta,K,L,links_array,links_ratings)

for meta in range(len(N_att_meta_items)):
    old_log_like += lambda_nodes*log_like_comp_arrays_exclusive(theta,q_kas[meta],K,metas_links_arrays_nodes[meta])

for meta in range(len(Taus)):
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
    #tik = time()
    #print('omega',any_nan(omega))
    #print('theta_0',any_nan(theta))
    theta = theta_comp_arrays(omega,theta,K,veins_nodes_array,N_veins_nodes)
    for meta in range(len(N_att_meta_items)):
        theta = sum_matrix_lambda(theta,theta_comp_arrays_exclusive(omega_nodes[meta],theta,K,metas_links_arrays_nodes[meta],N_veins_metas_nodes[meta],veins_metas_nodes[meta],N_att_meta_nodes[meta],N_veins_nodes),lambda_nodes)
        q_kas[meta] = q_ka_comp_arrays(omega_nodes[meta],q_kas[meta],K,metas_links_arrays_nodes[meta],N_veins_metas_nodes[meta])
        omega_nodes[meta] = omega_comp_arrays_exclusive(omega_nodes[meta],q_kas[meta],theta,K,metas_links_arrays_nodes[meta])
    '''if itt>-1:
        print(itt)
        #print(theta)
        tmp = theta_comp_arrays_exclusive(omega_ages,theta,K,age_link_array,N_veins_nodes)
        for i in tmp:
            print(i)'''
    eta = eta_multilayer(eta,omega,omega_items,veins_items_array,L,N_veins_items,lambda_items)

    for meta in range(len(Taus)):
        #eta2 = lambda_items*theta_comp_arrays(omega_items[meta],eta,Taus[meta],veins_items_metas[meta],N_veins_items)
        #eta = sum_matrix_lambda(eta,theta_comp_arrays(omega_items[meta],eta,Taus[meta],veins_items_metas[meta],N_veins_items),lambda_items)
        zetes[meta] = eta_comp_arrays(omega_items[meta],zetes[meta],Taus[meta],veins_metas_items[meta],N_veins_metas_items[meta])
        q_l_taus[meta] = p_kl_comp_arrays(omega_items[meta],q_l_taus[meta],zetes[meta],eta_temp,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])
        omega_items[meta] = omega_comp_arrays(omega_items[meta],q_l_taus[meta],zetes[meta],eta,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])

    p_kl = p_kl_comp_arrays(omega,p_kl,eta_temp,theta_temp,K,L,links_array,links_ratings)


    omega = omega_comp_arrays(omega,p_kl,eta,theta,K,L,links_array,links_ratings)
    #print(itt,'theta_1',omega[0,0])

    #theta_temp = theta.copy()
    eta_temp = eta.copy()
    if itt%N_measure==0:
        log_like = 0.0
        log_like += log_like_comp_arrays(p_kl,eta,theta,K,L,links_array,links_ratings)

        for meta in range(len(N_att_meta_items)):
            log_like += lambda_nodes*log_like_comp_arrays_exclusive(theta,q_kas[meta],K,metas_links_arrays_nodes[meta])

        for meta in range(len(Taus)):
            log_like += lambda_items*log_like_comp_arrays(q_l_taus[meta],zetes[meta],eta,L,Taus[meta],metas_links_arrays_items[meta],metas_links_arrays_items_type[meta])

        variation = old_log_like-log_like

        file_logLike.write('{}\t{}\t{}\n'.format(itt,log_like,variation))

        '''if finished(theta,eta,p_kl,theta_old,eta_old,p_kl_old,N_nodes,N_items,N_ratings,K,L,0.0001):
            if finished(zeta,eta,q_l_tau,zeta_old,eta_old,q_l_tau_old,N_genres,N_items,2,K,Tau,0.0001):
                if finished_2(theta,q_ka_ages,theta_old,q_ka_ages_old,N_nodes,K,0.0001):
                    if finished_2(theta,q_ka_genders,theta_old,q_ka_genders_old,N_nodes,K,0.0001):
                        print('------------------HA acabat!!!',itt,log_like)
                        break'''
        #else:
        if finished(theta,theta_old,K*N_nodes,0.0001):
            if finished(eta,eta_old,L*N_items,0.0001):
                out = False
                for meta in range(len(N_att_meta_items)):
                    if finished(q_kas[meta],q_kas_old[meta],K*N_att_meta_nodes[meta],0.0001):
                        out = False
                    else:
                        out = True
                        break
                if not out:
                    out2 = False
                    for meta in range(len(Taus)):
                        if finished(zetes[meta],zetes_old[meta],L*N_att_meta_items[meta],0.0001):
                            out2 = False
                        else:
                            out2 = True
                            break
                    if not out2:
                        if finished(p_kl,p_kl_old,L*K*N_ratings,0.0001):
                            out3 = False
                            for meta in range(len(Taus)):
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


for meta in range(len(Taus)):
    for r in range(2):
        np.savetxt(simu_dir+'/qlT_{}_{}.dat'.format(r,items_meta_data[meta]),q_l_taus[meta][:,:,r])

    np.savetxt(simu_dir+'/zeta_{}.dat'.format(items_meta_data[meta]),zetes[meta])

for meta in range(len(N_att_meta_items)):
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
