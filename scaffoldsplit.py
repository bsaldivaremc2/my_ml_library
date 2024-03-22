
import os
import json
import copy
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

# Scaffold split
def generate_scaffolds(mols):
    import collections
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = collections.defaultdict(list)
    for idx, mol in enumerate(mols):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaffold].append(idx)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

# def generate_scaffold_groups(mols):
#     scaffolds = generate_scaffolds(mols)
#     scaffold_matrix = np.zeros((len(mols), len(scaffolds)), dtype=int)
#     for s_id, m_id in enumerate(scaffolds):
#         scaffold_matrix[m_id, s_id] = 1
#     groups = [''.join(yy) for yy in scaffold_matrix.astype(int).astype(str)]
#     return np.array(groups)

def generate_scaffold_groups_from_smiles(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    scaffolds = generate_scaffolds(mols)
    scaffold_matrix = np.zeros((len(mols), len(scaffolds)), dtype=int)
    for s_id, m_id in enumerate(scaffolds):
        scaffold_matrix[m_id, s_id] = 1
    groups = [''.join(yy) for yy in scaffold_matrix.astype(int).astype(str)]
    return np.array(groups)

def generate_list_scaffold_groups_from_smiles(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    scaffolds = generate_scaffolds(mols)
    output = [None]*len(smiles_list)
    for s_id, m_id in enumerate(scaffolds):
      for m in m_id:
        output[m]=s_id
    return np.array(output)

def generate_list_scaffold_groups_from_smiles2(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    scaffolds = generate_scaffolds(mols)
    o = {}
    for s_id, m_id in enumerate(scaffolds):
      for m in m_id:
        _ = o.get(m,[])
        if s_id not in _:
          _.append(str(s_id))
        o[m]=_[:]
    #output = [ "".join(_)  for _ in output]
    output = [ ",".join(sorted(o[k]))  for k in sorted(o.keys()) ]
    return np.array(output)


def read_file(ifile):
  with open(ifile,"r") as f:
    _ = f.read().split("\n")
  _o =  []
  for __ in _:
    __ = __.strip()
    if len(__)>0:
      _o.append(__)
  return _o[:]

def get_splits_from_groups(ismiles,igroups,test_size=0.2):

  X = ismiles
  y = np.zeros_like(X)
  #groups = generate_scaffold_groups_from_smiles(smiles_list)
  gss = GroupShuffleSplit(
      n_splits = 1,
      train_size = 1- test_size,
      random_state = 0)
  n_split=0
  for train_idx, test_idx in gss.split(X, y, igroups):
      n_split += 1
      if n_split == 1:
          smiles_train = [ismiles[i] for i in train_idx]
          smiles_test = [ismiles[i] for i in test_idx]
  return smiles_train[:],smiles_test[:]

def get_splits_ids_from_groups(ismiles,igroups,test_size=0.2,splits=1):
  from sklearn.model_selection import GroupShuffleSplit
  X = ismiles
  y = np.zeros_like(X)
  #groups = generate_scaffold_groups_from_smiles(smiles_list)
  gss = GroupShuffleSplit(
      n_splits = splits,
      train_size = 1- test_size,
      random_state = 0)
  n_split=0
  o={}
  for train_idx, test_idx in gss.split(X, y, igroups):
    o[n_split]={'train_ids':train_idx.tolist(),'test_ids':test_idx.tolist()}
    n_split += 1
  return copy.deepcopy(o)

def save_info(info,ofile="info.json"):
  if "/" in ofile:
    _ = ofile.split("/")
    ff = _[-1]
    path = "/".join([__ for __ in _[:-1]])
    os.makedirs(path,exist_ok=True)
  with open(ofile,"w") as w:
    w.write(info)
    print("Saved on",ofile)

def save_splits(ismiles,splits_dic,odir):
  os.makedirs(odir,exist_ok=True)
  for k,v in splits_dic.items():
    for kk, vv in v.items():
      ofile = os.path.join(odir,"{}_split_{}_{}.txt".format(args.prefix,k,kk))
      with open(ofile,"w") as w:
        for _,idx in enumerate(vv):
          if _!=len(vv)-1:
            _c = ismiles[idx]+"\n"
          else:
            _c = ismiles[idx]
          w.write(_c)
      print("Saved:",ofile)

def main():
  ismiles = read_file(args.smiles_file)
  if args.sample_n>0:
    ismiles = list(np.random.choice(ismiles,args.sample_n,replace=False))
  #igroups = generate_scaffold_groups_from_smiles(ismiles)
  print("Generating groups")
  igroups = generate_list_scaffold_groups_from_smiles2(ismiles)
  print("splitting groups. Hard task.")
  split_ids = get_splits_ids_from_groups(ismiles,igroups,test_size=args.test_size,splits=args.n_splits)
  print("Saving result.")
  split_info = json.dumps(split_ids)
  save_info(split_info,args.split_info_file)
  m = """
  bk = read_file("data/info.json")[0]
  bk_split = eval(bk)
  """
  print("For reading the save file:",m)
  if args.save_proc_splits:
    save_splits(ismiles,split_ids,args.save_proc_splits_dir)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--smiles_file', type=str, default="valid_cc_molecules.txt")
  parser.add_argument('--split_info_file', type=str, default="data/splits.json")
  parser.add_argument('--save_proc_splits', type=str, default="false")
  parser.add_argument('--save_proc_splits_dir', type=str, default="./")
  parser.add_argument('--prefix', type=str, default="mols")
  parser.add_argument('--sample_n', type=int, default=100)
  parser.add_argument('--test_size', type=float, default=0.2)
  parser.add_argument('--n_splits', type=int, default=1)
  args = parser.parse_args()
  args.save_proc_splits = eval(args.save_proc_splits.strip("\t\n ").capitalize())
  main()
  print("Done")
