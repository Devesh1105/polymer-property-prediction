import pandas as pd, numpy as np, pickle, sys, warnings
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
d=pickle.load(open('feats.pkl','rb')); Xtr=d['Xtr']; Xte=d['Xte']
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv')
te=pd.read_csv('/home/user/AISEHACK-2.0/test.csv')
p1=pd.read_csv('/home/user/AISEHACK-2.0/archive/train.csv')
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m is not None else str(s)
for df in (tr,te,p1): df['c']=df.smiles.map(canon)
p1['target_type']=p1.target_type.str.lower()
LUT={(c,t):v for c,t,v in zip(p1.c,p1.target_type,p1.target)}
SEED=42; NF=8
def mk(seed,n=None):
    return lgb.LGBMRegressor(n_estimators=n or 3000,learning_rate=0.02,num_leaves=63,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.2,reg_lambda=1.5,min_child_samples=20,
        random_state=seed,verbosity=-1,n_jobs=4)
def honest(Xa,ya,Xb,seed=SEED):
    ia,ib=next(iter(KFold(5,shuffle=True,random_state=seed).split(Xa)))
    p=mk(seed); p.fit(Xa[ia],ya[ia],eval_set=[(Xa[ib],ya[ib])],
                      callbacks=[lgb.early_stopping(120,verbose=False)])
    f=mk(seed,n=max(50,int((p.best_iteration_ or 500)/0.8))); f.fit(Xa,ya); return f.predict(Xb)
print(f"{'target':6s} {'n_base':>7s} {'n_extra':>8s} {'baseline':>9s} {'+phase1':>9s} {'gain':>8s}")
res={}
for t in ['tg','egc']:
    m=(tr.target_type.values==t); Xb=Xtr[m]; y=tr.target.values[m].astype(np.float64)
    mt=(te.target_type.values==t)
    base_c=set(tr.c.values[m])
    keep=[i for i,c in enumerate(te.c.values[mt]) if (c,t) in LUT and c not in base_c]
    Xe=Xte[mt][keep]; ye=np.array([LUT[(te.c.values[mt][i],t)] for i in keep],np.float64)
    assert len(base_c & {te.c.values[mt][i] for i in keep})==0
    o0=np.zeros(len(y)); o1=np.zeros(len(y))
    for a,b in KFold(NF,shuffle=True,random_state=SEED).split(np.arange(len(y))):
        o0[b]=honest(Xb[a],y[a],Xb[b])
        o1[b]=honest(np.vstack([Xb[a],Xe]),np.concatenate([y[a],ye]),Xb[b])
    r0,r1=r2_score(y,o0),r2_score(y,o1)
    print(f"{t:6s} {len(y):7d} {len(ye):8d} {r0:9.4f} {r1:9.4f} {r1-r0:+8.4f}")
    res[t]=(r0,r1); sys.stdout.flush()
pickle.dump(res,open('exp4.pkl','wb'))
