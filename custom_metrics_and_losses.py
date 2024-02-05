import torch
from scipy.spatial import distance
from rdkit.Chem import AllChem as Chem

def differentiable_tanimoto_distance(target,pred):
  """"
  Computes a differentiable approximation of the tanimoto distance / Intersection Over Union, that could be used with pytorch and gpus: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html#scipy.spatial.distance.jaccard
    https://github.com/scipy/scipy/blob/v1.12.0/scipy/spatial/distance.py#L756-L827
  It sums on the last dimension so keeps the batch dimension for parallel processing.
  """
  pred = torch.relu(pred-0.5)*2
  math_and = pred * target
  intersection = torch.sum(math_and,dim=-1)
  union = torch.sum(pred + target - math_and,dim=-1)
  iou = intersection / union
  return iou

def compare_custom_tanimoto(mfp1,mfp2,new_func,new_func_kargs={}):
  with torch.no_grad():
    #baseline
    mfp1_cpu = mfp1
    mfp1_gpu = mfp1
    if torch.is_tensor(mfp1):
      mfp1_cpu = mfp1.cpu().numpy()
    else:
      mfp1_gpu = torch.tensor(mfp1_gpu).cuda()
    mfp2_cpu = mfp2
    mfp2_gpu = mfp2
    if torch.is_tensor(mfp2):
      mfp2_cpu = mfp2.cpu().numpy()
    else:
      mfp2_gpu = torch.tensor(mfp2_gpu).cuda()
    base_line = [get_tanimoto_similarity_from_mfp(mfp1_cpu[_,:].round(),mfp2_cpu[_,:].round()) for _ in range(mfp1.shape[0])]
    #new_func
    new_result = new_func(mfp1_gpu,mfp2_gpu,**new_func_kargs)
    print("Baseline\tNew")
    for _ in range(mfp1.shape[0]):
      print("{:.6f}\t{:.6f}".format(base_line[_],new_result[_]))

def check_validity(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False
    else:
        try:
            Chem.SanitizeMol(m)
            return True
        except:
            return False

def get_tanimoto_similarity(s1,s2):
    c1,c2 = Chem.MolFromSmiles(s1),Chem.MolFromSmiles(s2)
    fp1 = np.array(Chem.GetMorganFingerprintAsBitVect(c1, radius=2, nBits=2048))
    fp2 = np.array(Chem.GetMorganFingerprintAsBitVect(c2, radius=2, nBits=2048))
    return 1 - distance.jaccard(fp1, fp2)

def get_tanimoto_similarity_from_mfp(fp1,fp2):
    return 1 - distance.jaccard(fp1, fp2)
