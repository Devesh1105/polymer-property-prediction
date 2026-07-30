import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

def _stars(m): return [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]

def cyclize(smi):
    """Bond the two *-neighbours to each other and drop the dummies -> periodic (macrocyclic) unit."""
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    st=_stars(m)
    if len(st)!=2: return None
    rw=Chem.RWMol(m)
    nb=[]
    for s in st:
        n=[x.GetIdx() for x in rw.GetAtomWithIdx(s).GetNeighbors()]
        if len(n)!=1: return None
        nb.append(n[0])
    if rw.GetBondBetweenAtoms(nb[0],nb[1]) is None and nb[0]!=nb[1]:
        rw.AddBond(nb[0],nb[1],Chem.BondType.SINGLE)
    for s in sorted(st,reverse=True): rw.RemoveAtom(s)
    try:
        out=rw.GetMol(); Chem.SanitizeMol(out); return Chem.MolToSmiles(out)
    except Exception: return None

def oligomer(smi,n=3,cap="[H]"):
    """Head-to-tail n-mer from a 2-star repeat unit, ends capped."""
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    if len(_stars(m))!=2: return None
    combo=m
    for _ in range(n-1):
        combo=Chem.CombineMols(combo,m)
    rw=Chem.RWMol(combo); na=m.GetNumAtoms()
    st_local=_stars(m)
    # per-copy dummy indices
    copies=[[s+k*na for s in st_local] for k in range(n)]
    def nbr(rw,s):
        x=[a.GetIdx() for a in rw.GetAtomWithIdx(s).GetNeighbors()]
        return x[0] if len(x)==1 else None
    # link copy k tail -> copy k+1 head
    for k in range(n-1):
        a,b=copies[k][1],copies[k+1][0]
        na_,nb_=nbr(rw,a),nbr(rw,b)
        if na_ is None or nb_ is None: return None
        if rw.GetBondBetweenAtoms(na_,nb_) is None:
            rw.AddBond(na_,nb_,Chem.BondType.SINGLE)
    keep_dummies={copies[0][0],copies[n-1][1]}
    drop=sorted([s for c in copies for s in c if s not in keep_dummies],reverse=True)
    for s in drop: rw.RemoveAtom(s)
    try:
        out=rw.GetMol(); Chem.SanitizeMol(out)
        smi_out=Chem.MolToSmiles(out)
        return smi_out.replace("*","[H]") if cap=="[H]" else smi_out.replace("*",cap)
    except Exception: return None

tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv')
ex=tr.smiles.drop_duplicates().head(6).tolist()
for s in ex:
    print(f"repeat : {s}")
    print(f"  cyclic: {cyclize(s)}")
    print(f"  trimer: {oligomer(s,3)}")
uu=tr.smiles.drop_duplicates()
print(f"\ncyclize success : {sum(cyclize(s) is not None for s in uu)}/{len(uu)}")
print(f"trimer  success : {sum(oligomer(s,3) is not None for s in uu)}/{len(uu)}")
