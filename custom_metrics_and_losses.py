import torch

def differentiable_tanimoto_distance(u,v,input_sharpenss=1000000,sharpness=0.0005,e=1e-8,dim=-1):
  """"
  Computes a differentiable approximation of the tanimoto distance that could be used with pytorch and gpus: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html#scipy.spatial.distance.jaccard
    https://github.com/scipy/scipy/blob/v1.12.0/scipy/spatial/distance.py#L756-L827
  the number of decimals in sharpeness will aproximate it closer to 1. 0.0005 makes 0.999, 0.005 makes 0.99 (Not complete 1s)
  e: to avoid division by zero.
  It sums on the last dimension so keeps the batch dimension for parallel processing.
  """
  t1 = torch.sigmoid((u-0.5)*input_sharpenss)
  t2 = torch.sigmoid((v-0.5)*input_sharpenss)
  te = torch.tensor(e,dtype=torch.float32, requires_grad=False)
  nonzero = torch.pow(torch.sigmoid(t1+t2)-0.5,sharpness)
  different = torch.pow(t1-t2 ,2)
  unequal_nonzero = different * nonzero
  a = torch.sum(unequal_nonzero,dim)
  b = torch.sum(nonzero,dim)
  return a/(b+te)
