import pandas as pd, numpy as np, pickle
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv'); te=pd.read_csv('/home/user/AISEHACK-2.0/test.csv')
T=sorted(tr.target_type.unique())
DF=[(n,f) for n,f in Descriptors.descList]
g2=AllChem.GetMorganGenerator(radius=2,fpSize=1024)
g3=AllChem.GetMorganGenerator(radius=3,fpSize=1024)
def feat(smis):
    D,M2,M3,MA=[],[],[],[]
    for s in smis:
        m=Chem.MolFromSmiles(str(s))
        d=[]
        for _,f in DF:
            try: d.append(f(m))
            except Exception: d.append(np.nan)
        D.append(d)
        a2=np.zeros(1024,np.int16)
        for i,c in g2.GetCountFingerprint(m).GetNonzeroElements().items(): a2[i]=c
        a3=np.zeros(1024,np.int16)
        for i,c in g3.GetCountFingerprint(m).GetNonzeroElements().items(): a3[i]=c
        ma=np.zeros(167,np.int8); DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m),ma)
        M2.append(a2); M3.append(a3); MA.append(ma)
    return np.hstack([np.array(D,np.float32),np.array(M2,np.float32),
                      np.array(M3,np.float32),np.array(MA,np.float32)])
print('featurizing train...'); Xtr=feat(tr.smiles)
print('featurizing test...');  Xte=feat(te.smiles)
Xtr=np.nan_to_num(np.where(np.isinf(Xtr),np.nan,Xtr),nan=0.0).clip(-1e6,1e6)
Xte=np.nan_to_num(np.where(np.isinf(Xte),np.nan,Xte),nan=0.0).clip(-1e6,1e6)
# cross-target sibling block (their scheme, self zeroed)
LUT={t:dict(zip(tr.smiles[tr.target_type==t],tr.target[tr.target_type==t])) for t in T}
MED={t:float(np.median(list(LUT[t].values()))) for t in T}
def cross(smis,tts):
    n=len(smis); V=np.zeros((n,len(T)),np.float32); K=np.zeros((n,len(T)),np.float32)
    for j,t in enumerate(T):
        for i,s in enumerate(smis):
            if s in LUT[t]: V[i,j]=LUT[t][s]; K[i,j]=1.0
            else: V[i,j]=MED[t]
    for i,tt in enumerate(tts):
        j=T.index(tt); V[i,j]=MED[tt]; K[i,j]=0.0
    return np.hstack([V,K])
Xtr=np.hstack([Xtr,cross(tr.smiles.values,tr.target_type.values)]).astype(np.float32)
Xte=np.hstack([Xte,cross(te.smiles.values,te.target_type.values)]).astype(np.float32)
v=Xtr.var(0); keep=v>1e-8
Xtr=Xtr[:,keep]; Xte=Xte[:,keep]
print('feature matrix',Xtr.shape,Xte.shape)
pickle.dump({'Xtr':Xtr,'Xte':Xte,'T':T},open('feats.pkl','wb'))
