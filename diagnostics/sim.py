import pandas as pd, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv'); te=pd.read_csv('/home/user/AISEHACK-2.0/test.csv')
T=sorted(tr.target_type.unique())
gen=AllChem.GetMorganGenerator(radius=2,fpSize=2048)
def fps(smis):
    out=[]
    for s in smis:
        m=Chem.MolFromSmiles(str(s))
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out
print(f"{'target':6s} {'trainNN_med':>11s} {'testNN_med':>11s} {'trNN>0.8':>9s} {'teNN>0.8':>9s} {'trNN>0.95':>9s} {'teNN>0.95':>9s}")
for t in T:
    a=tr[tr.target_type==t]; b=te[te.target_type==t]
    fa=fps(a.smiles); fb=fps(b.smiles)
    fa=[f for f in fa if f is not None]; fb=[f for f in fb if f is not None]
    # train mol -> nearest OTHER train mol (same target)  == what group-CV validation sees
    trnn=[]
    for i,f in enumerate(fa):
        s=DataStructs.BulkTanimotoSimilarity(f,fa); s[i]=-1; trnn.append(max(s))
    # test mol -> nearest train mol (same target) == what LB sees
    tenn=[max(DataStructs.BulkTanimotoSimilarity(f,fa)) for f in fb]
    trnn=np.array(trnn); tenn=np.array(tenn)
    print(f"{t:6s} {np.median(trnn):11.3f} {np.median(tenn):11.3f} "
          f"{100*np.mean(trnn>0.8):8.1f}% {100*np.mean(tenn>0.8):8.1f}% "
          f"{100*np.mean(trnn>0.95):8.1f}% {100*np.mean(tenn>0.95):8.1f}%")
