import os
import gc
import h5py
import torch
import psutil
import pickle
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial import distance
from rdkit.Chem import AllChem as Chem
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


def print_ram():
  # Get the current process ID
  pid = psutil.Process()
  # Get the memory information
  memory_info = pid.memory_info()
  # Print the RAM usage in bytes, kilobytes, and megabytes
  print(f"Memory used: {memory_info.rss} bytes, {memory_info.rss / 1024:.2f} KB, {memory_info.rss / (1024 * 1024):.2f} MB. {memory_info.rss / (1024 **3):.2f} GB")

def read_h5(ifile):
  f = h5py.File(ifile, 'r')
  return f

def toArray(iobj):
  return np.array(iobj)
def datasetToArray(iobj):
  return np.array(iobj[:])

import time

def get_common_ick_df(fs):
  o = []
  for f in fs:
    name = f.split("/")[-2]
    print(name)
    h = read_h5(f)
    inchikeys = np.expand_dims(np.array([ _.decode("utf-8") for _ in toArray(h['keys']) ]),-1)
    ii = np.expand_dims(np.arange(len(inchikeys)).flatten(),-1)
    d = np.hstack([inchikeys,ii])
    _df = pd.DataFrame(d,columns=[name,"{}_i".format(name)])
    o.append(_df)
  ref_df = o[0]
  c1 = ref_df.columns.to_list()[0]
  for _ in range(1,len(o)):
    new_df = o[_]
    c2 = new_df.columns.to_list()[0]
    ref_df = pd.merge(ref_df, new_df, left_on=c1, right_on=c2, how='inner')
  return ref_df

def load_matrix_by_signature(idf_ick_index,signature_numbers=[0],space="B4",signatures_dir="/home"):
  _o = {}
  for sn in signature_numbers:
    print("Signature {}".format(sn))
    c = "sign{}_i".format(sn)
    ick_indexes = list(idf_ick_index[c].values.astype(int).flatten())
    f = "{}/{}/{}/{}.001/sign{}/sign{}.h5".format(signatures_dir,space[0],space,space,sn,sn)
    print("Reading file")
    time.sleep(1)
    with h5py.File(f, 'r') as h:
      print("Getting matrix")
      dataset = h['V'][ick_indexes, :]
      time.sleep(1)
      print("To numpy")
      _o["s{}".format(sn)] = np.array(dataset)
    time.sleep(1)
    print("Done\n")
  return _o

def load_signature_by_icks(icks,signature_number=0,space="B4",signatures_dir="/home"):
  sn = signature_number
  print("Signature {}".format(sn))
  c = "sign{}_i".format(sn)
  f = "{}/{}/{}/{}.001/sign{}/sign{}.h5".format(signatures_dir,space[0],space,space,sn,sn)
  print("Reading file")
  name = f.split("/")[-2]
  index_col = "{}_i".format(name)
  time.sleep(1)
  with h5py.File(f, 'r') as h:
    inchikeys = np.expand_dims(np.array([ _.decode("utf-8") for _ in toArray(h['keys']) ]),-1)
    ii = np.expand_dims(np.arange(len(inchikeys)).flatten(),-1)
    d = np.hstack([inchikeys,ii])
    _df = pd.DataFrame(d,columns=[name,index_col])
    _idf = pd.DataFrame(icks,columns=['i_icks'])
    _df = pd.merge(_idf,_df,left_on='i_icks',right_on=name,how='inner')
    ick_indexes = _df[index_col].values.astype(int)
    print("Getting matrix")
    dataset = h['V'][ick_indexes, :]
    time.sleep(1)
    print("To numpy")
    _o = np.array(dataset)
    print("Done\n")
  return _o, _df.copy()


def get_mfp(s1):
    c1 = Chem.MolFromSmiles(s1)
    return np.array(Chem.GetMorganFingerprintAsBitVect(c1, radius=2, nBits=2048))

def get_tanimoto_similarity(s1,s2):
    c1,c2 = Chem.MolFromSmiles(s1),Chem.MolFromSmiles(s2)
    fp1 = np.array(Chem.GetMorganFingerprintAsBitVect(c1, radius=2, nBits=2048))
    fp2 = np.array(Chem.GetMorganFingerprintAsBitVect(c2, radius=2, nBits=2048))
    return 1 - distance.jaccard(fp1, fp2)




def read_h5(ifile):
  f = h5py.File(ifile, 'r')
  return f

def toArray(iobj):
  return np.array(iobj)
def datasetToArray(iobj):
  return np.array(iobj[:])

class SignatureZero:
  def __init__(self,signature,inchikey,smile="",decoded_smile="",hlatent=None,label=""):
    self.signature = signature
    self.inchikey = inchikey
    self.smile = smile
    self.decoded_smile = decoded_smile
    self.hlatent = hlatent
    self.label  = label
  def __str__(self):
    m = """
    self.signature = signature #Signature Zero dims=[1,4635]
    self.inchikey = inchikey
    self.smile = smile #Cannonical smile
    self.decoded_smile = decoded_smile
    self.hlatent = hlatent #dims=[1,300]
    self.label  = label #train/val/test
    """
    return m

class mfpDataset(Dataset):
  def __init__(self, imfp,s0,device="cuda:0",max_items=-1):
      self.mfp = imfp
      self.s0 = s0
      if max_items>0:
        self.mfp = imfp[:max_items,:]
        self.s0 = s0[:max_items,:]
      self.device = device

  def __len__(self):
      return self.mfp.shape[0]
  def to_tensor(self,ix):
    return torch.tensor(ix.flatten(),device=self.device,dtype=torch.float32)

  def __getitem__(self, idx):
    x = self.mfp[idx]
    y = self.s0[idx]
    return self.to_tensor(x),self.to_tensor(y)

class xDataset(Dataset):
  def __init__(self,s0,device="cuda:0",max_items=-1):
      self.s0 = s0
      if max_items>0:
        self.s0 = s0[:max_items,:]
      self.device = device

  def __len__(self):
      return self.s0.shape[0]
  def to_tensor(self,ix):
    return torch.tensor(ix.flatten(),device=self.device,dtype=torch.float32)

  def __getitem__(self, idx):
    _ = self.s0[idx]
    return self.to_tensor(_)

def clean_cache(print_info=False):
  if print_info:
    print(get_cuda_info())
  gc.collect()
  torch.cuda.empty_cache()
  if print_info:
    print(get_cuda_info())

def get_cuda_info():
  mem_alloc = "%fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)
  mem_reserved = "%fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024)
  max_memory_reserved = "%fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024)
  return "GPU alloc: {}. Reserved: {}. MaxReserved: {}".format(mem_alloc,mem_reserved,max_memory_reserved)

class CosineDistanceLoss(nn.Module):
    def __init__(self, margin=0.0,target=1):
        super(CosineDistanceLoss, self).__init__()
        self.margin = margin
        self.target = target

    def forward(self, input1, input2):
        # Compute the cosine similarity
        cs = F.cosine_similarity(input1, input2, dim=1)
        return (-1*(cs-1)/2).mean()

class kl_divergence_loss(nn.Module):
  def __init__(self):
    super(kl_divergence_loss, self).__init__()
  def forward(self,target, predicted):
    # Ensure both target and predicted are probability distributions (e.g., apply softmax)
    target = F.softmax(target, dim=-1)
    predicted = F.softmax(predicted, dim=-1)

    # Calculate the KL Divergence
    kl_loss = torch.sum(target * (torch.log(target) - torch.log(predicted)), dim=-1)

    # Compute the mean loss over your batch
    mean_kl_loss = torch.mean(kl_loss)

    return mean_kl_loss

def pairwise_cosine_similarity(ix,iy):
  _x = ix if torch.is_tensor(ix) else torch.tensor(ix)
  _y = iy if torch.is_tensor(iy) else torch.tensor(iy)
  return F.cosine_similarity(_x[None,:,:].cuda(), _y[:,None,:].cuda(), dim=-1)

def pairwise_mse(vector_set1, vector_set2):
    """
    Compute pairwise mean squared error (MSE) between two sets of vectors.

    Args:
    - vector_set1 (torch.Tensor): Set of vectors, shape (batch_size, vector_dim).
    - vector_set2 (torch.Tensor): Another set of vectors, shape (batch_size, vector_dim).

    Returns:
    - torch.Tensor: Pairwise MSE matrix, shape (batch_size, batch_size).
    """
    mse_matrix = torch.cdist(vector_set1, vector_set2, p=2.0).pow(2)
    return mse_matrix

class contrastive_mse_cosinedistance(nn.Module):
  def __init__(self,base_loss=kl_divergence_loss):
    super(contrastive_mse_cosinedistance, self).__init__()
    self.base_loss = base_loss()
  def forward(self,cosine_distance_latent,mse_latent):
    cd = ((pairwise_cosine_similarity(cosine_distance_latent,cosine_distance_latent)-1)*-1) #pairwise cosine distance
    ed = ((pairwise_cosine_similarity(mse_latent,mse_latent)-1)*-1) #pairwise cosine distance
    #_ed = pairwise_mse(mse_latent,mse_latent) #euclidian distance
    return self.base_loss(cd,ed) #+ _ed.mean()

class contrastive_kld_cosine_similarity_loss(nn.Module):
  def __init__(self,base_loss=kl_divergence_loss):
    super(contrastive_kld_cosine_similarity_loss, self).__init__()
    self.base_loss = base_loss()
  def forward(self,target, predicted):
    x1 = pairwise_cosine_similarity(target,target)
    x2 = pairwise_cosine_similarity(predicted,predicted)
    return self.base_loss(x1,x2) + self.base_loss(target,predicted)

def calculate_correlations(x, y):
  vx = x - torch.mean(x,dim=-1,keepdim=True)
  vy = y - torch.mean(y,dim=-1,keepdim=True)
  return torch.sum(vx * vy,dim=-1) / (torch.sqrt(torch.sum(vx ** 2,dim=-1)) * torch.sqrt(torch.sum(vy ** 2,dim=-1)))

class sAE(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=32, layers = [1024,1024]):
        super(sAE, self).__init__()
        enc_layers_size = [input_dim] + layers + [latent_dim]
        dec_layers_size = enc_layers_size[::-1]
        self.encoder = nn.Sequential(*self.get_sequential(enc_layers_size),nn.Sigmoid())
        self.decoder = nn.Sequential(*self.get_sequential(dec_layers_size),nn.Sigmoid())

    def forward(self, x,add_random=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    def get_sequential(self,nlayers):
        olayers = []
        for i in range(len(nlayers)-1):
          sSize = nlayers[i]
          eSize = nlayers[i+1]
          olayers.append(nn.Linear(sSize, eSize))
          olayers.append(nn.BatchNorm1d(eSize))
          olayers.append(nn.ReLU())
        return olayers[:-2]

class sE(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=32, layers = [1024,1024]):
        super(sE, self).__init__()
        enc_layers_size = [input_dim] + layers + [latent_dim]
        dec_layers_size = enc_layers_size[::-1]
        self.encoder = nn.Sequential(*self.get_sequential(enc_layers_size),nn.Sigmoid())

    def forward(self, x,add_random=False):
        return self.encoder(x)
    def get_sequential(self,nlayers):
        olayers = []
        for i in range(len(nlayers)-1):
          sSize = nlayers[i]
          eSize = nlayers[i+1]
          olayers.append(nn.Linear(sSize, eSize))
          olayers.append(nn.BatchNorm1d(eSize))
          olayers.append(nn.ReLU())
        return olayers[:-2]

def calculate_correlations(x, y):
  vx = x - torch.mean(x,dim=-1,keepdim=True)
  vy = y - torch.mean(y,dim=-1,keepdim=True)
  return torch.sum(vx * vy,dim=-1) / (torch.sqrt(torch.sum(vx ** 2,dim=-1)) * torch.sqrt(torch.sum(vy ** 2,dim=-1)))

def plot_densities(vals,labels,bins=128,range=(-1,1),figsize=(12,8),dpi=80,alpha=0.4,
                   t="",xl="",yl="Density"):
  plt.figure(figsize=figsize,dpi=dpi)
  ys = []
  for v in vals:
    tr_cc_y, tr_cc_b = np.histogram(v,bins=bins,range=range)
    tr_cc_y = tr_cc_y/tr_cc_y.sum()
    ys.append(tr_cc_y.copy())
  dx = abs(tr_cc_b[0]-tr_cc_b[1])
  tr_cc_x = tr_cc_b[:-1] + dx
  for y,label in zip(ys,labels):
    plt.fill_between(tr_cc_x,y,label=label,alpha=alpha)
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.title(t)
  plt.legend()
  plt.show()

#mseLoss = nn.MSELoss()
CDL = CosineDistanceLoss()
CMSE_CS = contrastive_mse_cosinedistance()


def s0_to_l0_e_loss_function(s0x, encoded):
  latent_kld = CMSE_CS(s0x,encoded)
  #latent_cd = CDL(s0x,encoded)
  #cs_rec = F.cosine_similarity(s0x,decoded, dim=1).mean()
  return latent_kld , 0


def s0_to_l0_ae_loss_function(s0x, encoded, decoded,compute_reconstruction=True):
  latent_kld = CMSE_CS(s0x,encoded)
  rec_loss = CDL(s0x,decoded)
  if compute_reconstruction:
    rec_loss=rec_loss*0.01
  cs_rec = F.cosine_similarity(s0x,decoded, dim=1).mean()
  return latent_kld  +  rec_loss, cs_rec

def zero_one_to_127(ix):
  return (ix*127*2 - 127).round().astype(np.int8)

def from_127_to_0_1(ix):
  _ = ix.astype(np.float16)
  return (_ + 127)/(127*2)

def min_max_to_127(ix,minv=-1,maxv=1):
  _x = (ix - minv)/(maxv-minv)
  return zero_one_to_127(_x)

def from_127_to_min_max(ix,minv=-1,maxv=1):
  _x = from_127_to_0_1(ix)
  return _x*(maxv-minv) + minv


def get_paiwise_iou_two_inputs(imatrix,tmatrix,device="cuda:0",output_format=np.int8,
                               batch_size=1024):
  """
  Returns element wise matrix of IntersectionOverUnion, Tanimoto Similarity or Jaccard Distance.
  Input: torch tensor, numpy array or list of numpy vectors. All types should be integer already.
  device="cuda:0". Define if you use cuda (GPU) or cpu device="cpu"
  return_complete_matrix=False. If True, the diagonal and complementary section of the matrix is also filled
  is_distance=False. By default computes Tanimoto Similarity, Intersection Over Union. Set to True to get Jaccard distance.
  """
  with torch.no_grad():
    n_elements,n_feats = imatrix.shape
    m_elements,n_feats = tmatrix.shape
    _i = imatrix
    print("generating output empty")
    _o = np.zeros((n_elements,m_elements)).astype(output_format)
    print(_o.shape,"Output shape")
    print("Start")
    for _row in tqdm(range(n_elements)):
      _v = torch.tensor(imatrix[_row,:],dtype=torch.int).to(device)
      _row_iou = []
      for _start in range(0,m_elements+batch_size,batch_size):
        _m = torch.tensor(tmatrix[_start:_start+batch_size,:],dtype=torch.int).to(device)
        i = (_v&_m).sum(1)
        u = (_v|_m).sum(1)
        iou = (i/u).cpu().numpy()
        del _m
        if output_format==np.int8:
          iou = zero_one_to_127(iou)
        else:
          iou = iou.astype(output_format)
        _row_iou.append(iou)
      if len(_row_iou)>1:
        iou = np.hstack(_row_iou)
      else:
        iou = _row_iou[0]
      _o[_row,:]=iou
      clean_cache(print_info=False)
  return _o

def get_paiwise_iou(imatrix,device="cuda:0",return_complete_matrix=False,is_distance=False
                    ,output_format=np.int8,
                    batch_size=1024,min_value=-1,max_value=1):
  """
  Returns element wise matrix of IntersectionOverUnion, Tanimoto Similarity or Jaccard Distance.
  Input: torch tensor, numpy array or list of numpy vectors. All types should be integer already.
  device="cuda:0". Define if you use cuda (GPU) or cpu device="cpu"
  return_complete_matrix=False. If True, the diagonal and complementary section of the matrix is also filled
  is_distance=False. By default computes Tanimoto Similarity, Intersection Over Union. Set to True to get Jaccard distance.
  """
  with torch.no_grad():
    n_elements = imatrix.shape[0]
    _i = imatrix
    print("generating output empty")
    #_o = np.zeros((n_elements,n_elements)).astype(np.int8)
    _o = np.zeros((n_elements,n_elements)).astype(output_format)
    print("Start")
    for _row in tqdm(range(n_elements-1)):
      second_vector_start = _row+1
      if return_complete_matrix:
        second_vector_start = _row
      _v = torch.tensor(imatrix[_row,:],dtype=torch.int).to(device)
      for _start in range(second_vector_start,n_elements+batch_size,batch_size):
        _m = torch.tensor(imatrix[_start:_start+batch_size,:],dtype=torch.int).to(device)
        i = (_v&_m).sum(1)
        u = (_v|_m).sum(1)
        iou = (i/u).cpu().numpy()
        if output_format==np.int8:
          #iou = zero_one_to_127(iou)
          iou = min_max_to_127(iou,minv=min_value,maxv=max_value)
        else:
          iou = iou.astype(output_format)
        if is_distance:
          iou = 1 - iou
        _o[_row,_start:iou.shape[0]+_start]=iou
        if return_complete_matrix:
          _o[second_vector_start:,_row]=iou
        #clean_cache(print_info=False)
  return _o

def get_paiwise_cosine_similarity(imatrix,device="cuda:0",return_complete_matrix=False,is_distance=False
                    ,output_format=np.int8,
                    batch_size=1024,min_value=-1,max_value=1):
  """
  Returns element wise matrix of IntersectionOverUnion, Tanimoto Similarity or Jaccard Distance.
  Input: torch tensor, numpy array or list of numpy vectors. All types should be integer already.
  device="cuda:0". Define if you use cuda (GPU) or cpu device="cpu"
  return_complete_matrix=False. If True, the diagonal and complementary section of the matrix is also filled
  is_distance=False. By default computes Tanimoto Similarity, Intersection Over Union. Set to True to get Jaccard distance.
  """
  with torch.no_grad():
    n_elements = imatrix.shape[0]
    _i = imatrix
    print("generating output empty")
    #_o = np.zeros((n_elements,n_elements)).astype(np.int8)
    _o = np.zeros((n_elements,n_elements)).astype(output_format)
    print("Start")
    for _row in tqdm(range(n_elements-1)):
      second_vector_start = _row+1
      if return_complete_matrix:
        second_vector_start = _row
      _v = torch.tensor(imatrix[_row,:],dtype=torch.float16).to(device)
      for _start in range(second_vector_start,n_elements+batch_size,batch_size):
        _m = torch.tensor(imatrix[_start:_start+batch_size,:],dtype=torch.float16).to(device)
        iou = F.cosine_similarity(_v,_m,dim=-1).cpu().numpy()
        if output_format==np.int8:
          #iou = zero_one_to_127(iou)
          iou = min_max_to_127(iou,minv=min_value,maxv=max_value)
        else:
          iou = iou.astype(output_format)
        if is_distance:
          iou = 1 - iou
        _o[_row,_start:iou.shape[0]+_start]=iou
        if return_complete_matrix:
          _o[second_vector_start:,_row]=iou
        #clean_cache(print_info=False)
  return _o



def plot_precision_recall(idic,title="",figsize=(16,16),dpi=80):
  plt.figure(figsize=figsize,dpi=dpi)
  plt.suptitle(title)
  #for i,k in enumerate(sorted(idic.keys())):
  for i,k in enumerate(idic.keys()):
    plt.subplot(3, 3, i+1)
    pr, rc = idic[k]['precision'],idic[k]['recall']
    #auc = np.round(metrics.auc(fpr, tpr),3)
    #plt.title("MMP{}. AUC: {}".format(k,auc))
    plt.title("{}".format(k))
    plt.plot(rc, pr)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    #plt.plot([0,1],[0,1],"r--")
    plt.xlim(0,1)
    plt.ylim(0,1)
  plt.show()

def plot_rocs(idic,title="",figsize=(16,16),dpi=80):
  from sklearn import metrics
  plt.figure(figsize=figsize,dpi=dpi)
  plt.suptitle(title)
  #for i,k in enumerate(sorted(idic.keys())):
  for i,k in enumerate(idic.keys()):
    plt.subplot(3, 3, i+1)
    fpr, tpr = idic[k]['fpr'],idic[k]['tpr']
    auc = np.round(metrics.auc(fpr, tpr),3)
    plt.title("{}. AUC: {}".format(k,auc))
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0,1],[0,1],"r--")
    plt.xlim(0,1)
    plt.ylim(0,1)
  plt.show()

def plot_densities(vals,labels,bins=128,range=(-1,1),figsize=(12,8),dpi=80,alpha=0.4,
                   t="",xl="",yl="Density"):
  plt.figure(figsize=figsize,dpi=dpi)
  ys = []
  for v in vals:
    tr_cc_y, tr_cc_b = np.histogram(v,bins=bins,range=range)
    tr_cc_y = tr_cc_y/tr_cc_y.sum()
    ys.append(tr_cc_y.copy())
  dx = abs(tr_cc_b[0]-tr_cc_b[1])
  tr_cc_x = tr_cc_b[:-1] + dx
  for y,label in zip(ys,labels):
    plt.fill_between(tr_cc_x,y,label=label,alpha=alpha)
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.title(t)
  plt.legend()
  plt.show()

def get_neighbor_predictions(defined_neighbors_df,pred_matrix,th=85):
  _N = defined_neighbors_df.shape[0]
  _opositives = []
  _onegatives = []
  for df_row_i in tqdm(range(_N)):
    _row = defined_neighbors_df.iloc[df_row_i]
    mol_ix = _row['mol']
    _ps = _row['positive']
    _ns = _row['negative']
    _values = pred_matrix[mol_ix,:].flatten()+pred_matrix[:,mol_ix].flatten()
    #_values = (_values>=th).astype(int)
    _opositives+=list(_values[_ps])+list(_values[_ns])
    _onegatives+=list(np.ones((len(_ps)))) + list(np.zeros((len(_ns))))
  return np.array(_opositives),np.array(_onegatives)

def get_neighbors_by_threshold(imatrix,th=87):
  _o = {}
  _all_ixs = set(list(range(imatrix.shape[0])))
  for ix in tqdm(_all_ixs):
    row = imatrix[ix,:]
    col = imatrix[:,ix]
    rowis = np.where(row>=th)
    colis = np.where(col>=th)
    _pos = set(sorted(set(list(rowis[0])+list(colis[0]))))
    if len(_pos)>0:
      _all_else = _all_ixs - _pos
      _ = np.random.choice(list(_all_else),len(_pos),replace=False)
      _neg = list(_)
      _o[ix]={'positive':list(_pos)[:],'negative':list(_neg)[:]}
  return _o.copy()
