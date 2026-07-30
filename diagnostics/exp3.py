import pandas as pd, numpy as np, pickle, sys, warnings
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
warnings.filterwarnings("ignore")
d=pickle.load(open('feats2.pkl','rb')); b3=pickle.load(open('feats3.pkl','rb'))
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv')
SEED=42; NF=8; A2=slice(217,217+1024)
def mk(seed,small,n=None):
    p=dict(random_state=seed,verbosity=-1,n_jobs=4)
    if small: p.update(n_estimators=n or 1500,learning_rate=0.03,num_leaves=15,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.5,reg_lambda=3.0,min_child_samples=8)
    else: p.update(n_estimators=n or 3000,learning_rate=0.02,num_leaves=63,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.2,reg_lambda=1.5,min_child_samples=20)
    return lgb.LGBMRegressor(**p)
def lgb_honest(Xa,ya,Xb,small,seed=SEED):
    ia,ib=next(iter(KFold(5,shuffle=True,random_state=seed).split(Xa)))
    pr=mk(seed,small); pr.fit(Xa[ia],ya[ia],eval_set=[(Xa[ib],ya[ib])],
                              callbacks=[lgb.early_stopping(120,verbose=False)])
    f=mk(seed,small,n=max(50,int((pr.best_iteration_ or 500)/0.8))); f.fit(Xa,ya); return f.predict(Xb)
def minmax_K(A,B):
    K=np.empty((A.shape[0],B.shape[0]))
    for i in range(A.shape[0]): K[i]=np.minimum(A[i],B).sum(1)/np.maximum(np.maximum(A[i],B).sum(1),1e-9)
    return K
def gp_fold(Ka,ya,Kb):
    mu,sd=ya.mean(),ya.std()+1e-9; yn=(ya-mu)/sd; n=len(ya); best=(-1e18,None)
    for amp in (0.5,1.0,2.0):
        for nz in (3e-3,1e-2,3e-2,1e-1,3e-1):
            try: c=cho_factor(amp*Ka+nz*np.eye(n),lower=True)
            except Exception: continue
            al=cho_solve(c,yn); lml=-0.5*yn@al-np.log(np.diag(c[0])).sum()
            if lml>best[0]: best=(lml,(amp,al))
    amp,al=best[1]; return (amp*Kb)@al*sd+mu
COMPACT=['PHYS','STRUCT','CONJ','TRI']
VARIANTS={'base':[], '+compact':COMPACT, '+cmp+smarts':COMPACT+['SMARTS'], '+all':COMPACT+['SMARTS','AUTO']}
TARGETS=['eea','ei','eps','nc','egb','egc','tg']
folds=lambda n: list(KFold(NF,shuffle=True,random_state=SEED).split(np.arange(n)))
res={}
print(f"{'target':6s} {'n':>5s} "+" ".join(f"{k:>12s}" for k in VARIANTS)+f" {'gp':>8s} {'STACK':>8s} {'gain':>7s}")
for t in TARGETS:
    m=(tr.target_type.values==t); y=tr.target.values[m].astype(np.float64); small=m.sum()<600
    Xb=d['Xb_tr'][m]; add={k:b3['tr'][k][m] for k in b3['dims']}
    FP=Xb[:,A2]; dense=np.hstack([Xb[:,:217],Xb[:,2432:2446]]+[add[k] for k in COMPACT])
    oofs={v:np.zeros(len(y)) for v in VARIANTS}; og=np.zeros(len(y))
    for a,bidx in folds(len(y)):
        for v,blocks in VARIANTS.items():
            X=np.hstack([Xb]+[add[k] for k in blocks])
            if not small:
                p=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=63,subsample=0.8,
                    colsample_bytree=0.6,random_state=SEED,verbosity=-1,n_jobs=4).fit(X[a],y[a])
                ix=np.sort(np.argsort(-p.feature_importances_)[:900])
            else: ix=np.arange(X.shape[1])
            oofs[v][bidx]=lgb_honest(X[a][:,ix],y[a],X[bidx][:,ix],small)
        sc=StandardScaler().fit(dense[a]); Za,Zb=sc.transform(dense[a]),sc.transform(dense[bidx])
        Za=np.clip(Za,-8,8); Zb=np.clip(Zb,-8,8); g=1.0/Za.shape[1]
        Ka=minmax_K(FP[a],FP[a]); Kb=minmax_K(FP[bidx],FP[a])
        Ra=np.exp(-g*((Za**2).sum(1)[:,None]+(Za**2).sum(1)[None,:]-2*Za@Za.T))
        Rb=np.exp(-g*((Zb**2).sum(1)[:,None]+(Za**2).sum(1)[None,:]-2*Zb@Za.T))
        og[bidx]=gp_fold(0.5*Ka+0.5*Ra,y[a],0.5*Kb+0.5*Rb)
    scores={v:r2_score(y,oofs[v]) for v in VARIANTS}
    bestv=max(scores,key=scores.get); rg=r2_score(y,og)
    # shrunk NNLS over [best lgb variant, gp], lambda by inner CV
    M=np.column_stack([oofs[bestv],og]); best_st=(-9,None)
    for lam in (0.0,0.25,0.5,0.75,1.0):
        cv=np.zeros(len(y))
        for a2_,b2_ in KFold(5,shuffle=True,random_state=SEED).split(M):
            w,_=nnls(M[a2_],y[a2_]); w=w/w.sum() if w.sum()>0 else np.ones(2)/2
            w=(1-lam)*w+lam*np.ones(2)/2; cv[b2_]=M[b2_]@w
        s=r2_score(y,cv)
        if s>best_st[0]: best_st=(s,lam)
    rs,lam=best_st
    res[t]=dict(scores=scores,gp=rg,stack=rs,lam=lam,base=scores['base'],bestv=bestv)
    print(f"{t:6s} {m.sum():5d} "+" ".join(f"{scores[v]:12.4f}" for v in VARIANTS)
          +f" {rg:8.4f} {rs:8.4f} {rs-scores['base']:+7.4f}   [{bestv}, lam={lam}]")
    sys.stdout.flush()
pickle.dump(res,open('exp3.pkl','wb'))
print(f"\nmean honest OOF: base={np.mean([v['base'] for v in res.values()]):.4f}"
      f"  best_feats={np.mean([max(v['scores'].values()) for v in res.values()]):.4f}"
      f"  +GP stack={np.mean([v['stack'] for v in res.values()]):.4f}")
