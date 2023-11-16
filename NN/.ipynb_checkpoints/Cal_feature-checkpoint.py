from rdkit import Chem
import torch
from torch_geometric.data import Data, DataLoader
from featurizer import *

sdf_path = "../data/PubChem_65sdf.sdf" #64 compounds indeed.
suppl = Chem.SDMolSupplier(sdf_path)
cid_path = "../data/cids.csv"
cid2name = pd.read_csv(cid_path, header = 0)
cid2name = {key:cid2name["compound"][i] for i, key in enumerate(cid2name["CID"])}

#max_len = max([mol.GetNumAtoms() for mol in suppl])
#mean_len = int(np.mean([mol.GetNumAtoms() for mol in suppl]))
    
for mol in suppl:
atom_f = torch.as_tensor(atom_f,dtype = torch.float32)
bond_f = torch.as_tensor(bond_f,dtype = torch.float32)
edge_index = torch.as_tensor(edge_index)
edge_index = torch.tensor([edge_index[:,0].tolist(), edge_index[:,1].tolist()])

batch = torch.zeros(61, dtype=torch.long)
p = model(atom_f,edge_index,  bond_f, batch = batch)