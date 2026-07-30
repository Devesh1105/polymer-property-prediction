import json, numpy as np, pandas as pd, warnings, types
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

nb=json.load(open('/home/user/AISEHACK-2.0/polymer_r2_v8.ipynb'))
src={i:''.join(c['source']) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
G=dict(globals()); G.update(np=np,pd=pd,Chem=Chem,Descriptors=Descriptors,
    rdMolDescriptors=rdMolDescriptors,lgb=lgb,xgb=xgb,CatBoostRegressor=CatBoostRegressor,
    KFold=KFold,r2_score=r2_score,StandardScaler=StandardScaler,SEED=42)

tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv'); te=pd.read_csv('/home/user/AISEHACK-2.0/test.csv')

# ---- 1. polymer block (cell 14) ----
poly=src[14].split("print(\"Building polymer blocks")[0]
exec(poly, G)
sub=tr.smiles.drop_duplicates().head(400).tolist()
rows=[G['poly_block'](s) for s in sub]
ok=[r for r in rows if r]
A=np.array(ok,np.float32)
print(f"[1] poly_block: {len(ok)}/{len(sub)} parsed, width={A.shape[1]}, "
      f"nonfinite={int((~np.isfinite(A)).sum())}")
assert len(ok)==len(sub) and A.shape[1]==77

# ---- 2. honest early stopping (cell 18) ----
head=src[18].split("# v8: small targets get 10 seeds")[0]
exec(head, G)
m=(tr.target_type.values=='eea'); y=tr.target.values[m].astype(np.float32)
Xs=np.array([[Descriptors.MolWt(Chem.MolFromSmiles(s)),Descriptors.TPSA(Chem.MolFromSmiles(s)),
              Descriptors.MolLogP(Chem.MolFromSmiles(s)),Chem.MolFromSmiles(s).GetNumHeavyAtoms()]
             for s in tr.smiles[m]],np.float32)
Xs=np.hstack([Xs, A[:len(y)] if len(A)>=len(y) else np.zeros((len(y),77),np.float32)])
for name,mkfn in (("lgb",G['get_models_small']),("cat",G['get_models_small']),("xgb",G['get_models'])):
    probe=mkfn(42)[name] if name in mkfn(42) else G['get_models'](42)[name]
    mk = mkfn if name in mkfn(42) else G['get_models']
    b=G['_budget_on_inner'](name, probe, Xs, y, 42)
    mdl=G['_refit_full'](name, mk, 42, b, Xs, y)
    p=mdl.predict(Xs[:20])
    print(f"[2] {name:4s} budget={b} refit OK, preds finite={np.isfinite(p).all()}")

# ---- 3. GP kernels + solver (cell 27) ----
gpsrc=src[27].split("_M2_IDX")[0]+"\n"+src[27].split("def _minmax_K",1)[1]
gpsrc="from scipy.linalg import cho_factor, cho_solve\ndef _minmax_K"+src[27].split("def _minmax_K",1)[1]
gpsrc=gpsrc.split("def train_gp")[0]
exec(gpsrc, G)
from rdkit.Chem import AllChem
gen=AllChem.GetMorganGenerator(radius=2,fpSize=1024)
FP=np.zeros((len(y),1024),np.float32)
for i,s in enumerate(tr.smiles[m]):
    for k,v in gen.GetCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements().items(): FP[i,k]=v
oof=np.zeros(len(y))
for a,b in KFold(8,shuffle=True,random_state=42).split(FP):
    sc=StandardScaler().fit(Xs[a]); Za=np.clip(sc.transform(Xs[a]),-8,8); Zb=np.clip(sc.transform(Xs[b]),-8,8)
    g=1.0/Za.shape[1]; sa=(Za**2).sum(1)
    Ra=np.exp(-g*(sa[:,None]+sa[None,:]-2*Za@Za.T))
    Rb=np.exp(-g*((Zb**2).sum(1)[:,None]+sa[None,:]-2*Zb@Za.T))
    Ka=0.5*G['_minmax_K'](FP[a],FP[a])+0.5*Ra; Kb=0.5*G['_minmax_K'](FP[b],FP[a])+0.5*Rb
    oof[b]=G['_gp_fit_predict'](Ka,y[a],Kb)
print(f"[3] GP (MinMax) eea OOF R2 = {r2_score(y,oof):.4f}")
Kt=G['_tanimoto_K'](FP[:50],FP[:50])
print(f"[3] tanimoto_K: shape={Kt.shape} diag~1={np.allclose(np.diag(Kt),1)} "
      f"symmetric={np.allclose(Kt,Kt.T)} range=[{Kt.min():.2f},{Kt.max():.2f}]")

# ---- 4. shrunk stacking weights (cell 29) ----
from scipy.optimize import nnls
G['nnls']=nnls
exec(src[29].split("LAM_GRID")[0], G)
M=np.column_stack([oof, oof*0.9+y.mean()*0.1])
for lam in (0.0,0.5,1.0):
    w=G['_shrunk_weights'](M,y,lam)
    print(f"[4] lam={lam}: w={np.round(w,3)} sum={w.sum():.3f}")
print("\nALL SMOKE TESTS PASSED")
