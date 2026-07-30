import pandas as pd, numpy as np, pickle, sys
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
d=pickle.load(open('feats.pkl','rb')); Xtr=d['Xtr']; T=d['T']
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv')
SEED=42; NF=8
def folds(n,seed=SEED):
    return list(KFold(NF,shuffle=True,random_state=seed).split(np.arange(n)))
def sel(X,y,k,nl):
    p=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=nl,subsample=0.8,
        colsample_bytree=0.6,random_state=SEED,verbosity=-1,n_jobs=4).fit(X,y)
    return np.sort(np.argsort(-p.feature_importances_)[:k])
def mdl(seed,small):
    if small: return lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.03,num_leaves=15,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.5,reg_lambda=3.0,
        min_child_samples=8,random_state=seed,verbosity=-1,n_jobs=4)
    return lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.02,num_leaves=63,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.2,reg_lambda=1.5,min_child_samples=20,
        random_state=seed,verbosity=-1,n_jobs=4)
TARGETS=['eea','ei','eps','nc','egb','egc']
print(f"{'target':6s} {'n':>5s} {'leakFS+leakES':>14s} {'honFS+leakES':>13s} {'leakFS+honES':>13s} {'honFS+honES':>12s} {'INFLATION':>10s}")
rows=[]
for t in TARGETS:
    m=(tr.target_type.values==t); X=Xtr[m]; y=tr.target.values[m].astype(np.float32)
    small=m.sum()<600; k=300 if small else 900; nl=15 if small else 63
    gsel=sel(X,y,k,nl)                       # leaky: uses ALL labels
    res={}
    for fs in ('leak','hon'):
        for es in ('leak','hon'):
            oof=np.zeros(len(y))
            for a,b in folds(len(y)):
                idx = gsel if fs=='leak' else sel(X[a],y[a],k,nl)
                Xa,Xb=X[a][:,idx],X[b][:,idx]
                mm=mdl(SEED,small)
                if es=='leak':
                    mm.fit(Xa,y[a],eval_set=[(Xb,y[b])],callbacks=[lgb.early_stopping(120,verbose=False)])
                else:
                    ia,ib=next(iter(KFold(5,shuffle=True,random_state=SEED).split(a)))
                    mm.fit(Xa[ia],y[a][ia],eval_set=[(Xa[ib],y[a][ib])],
                           callbacks=[lgb.early_stopping(120,verbose=False)])
                oof[b]=mm.predict(Xb)
            res[(fs,es)]=r2_score(y,oof)
    infl=res[('leak','leak')]-res[('hon','hon')]
    print(f"{t:6s} {m.sum():5d} {res[('leak','leak')]:14.4f} {res[('hon','leak')]:13.4f} "
          f"{res[('leak','hon')]:13.4f} {res[('hon','hon')]:12.4f} {infl:+10.4f}")
    rows.append((t,res,infl)); sys.stdout.flush()
pickle.dump(rows,open('exp.pkl','wb'))
print()
print(f"mean inflation over these 6 targets: {np.mean([r[2] for r in rows]):+.4f}")
