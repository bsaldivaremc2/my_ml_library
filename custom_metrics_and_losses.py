import torch

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
