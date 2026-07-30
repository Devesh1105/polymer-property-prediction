import pandas as pd, numpy as np, pickle, sys, warnings
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.linalg import cho_factor, cho_solve
warnings.filterwarnings("ignore")
d=pickle.load(open('feats2.pkl','rb'))
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv')
SEED=42; NF=8
A2=slice(217,217+1024)   # Morgan r2 counts inside the baseline block

def mk(seed,small,n=None):
    p=dict(random_state=seed,verbosity=-1,n_jobs=4)
    if small: p.update(n_estimators=n or 1500,learning_rate=0.03,num_leaves=15,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.5,reg_lambda=3.0,min_child_samples=8)
    else: p.update(n_estimators=n or 3000,learning_rate=0.02,num_leaves=63,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.3,reg_alpha=0.2,reg_lambda=1.5,min_child_samples=20)
    return lgb.LGBMRegressor(**p)

def lgb_honest(Xa,ya,Xb,small,seed=SEED):
    """early stopping on an inner split of the TRAIN fold, then refit on the whole fold"""
    ia,ib=next(iter(KFold(5,shuffle=True,random_state=seed).split(Xa)))
    pr=mk(seed,small); pr.fit(Xa[ia],ya[ia],eval_set=[(Xa[ib],ya[ib])],
                              callbacks=[lgb.early_stopping(120,verbose=False)])
    n=max(50,int((pr.best_iteration_ or 500)/0.8))
    f=mk(seed,small,n=n); f.fit(Xa,ya); return f.predict(Xb)

def sel_infold(Xa,ya,k,nl,seed=SEED):
    p=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=nl,subsample=0.8,
        colsample_bytree=0.6,random_state=seed,verbosity=-1,n_jobs=4).fit(Xa,ya)
    return np.sort(np.argsort(-p.feature_importances_)[:k])

def minmax_K(A,B):
    K=np.empty((A.shape[0],B.shape[0]),np.float64)
    for i in range(A.shape[0]):
        K[i]=np.minimum(A[i],B).sum(1)/np.maximum(np.maximum(A[i],B).sum(1),1e-9)
    return K

def gp_fold(Ka,ya,Kb,amp_grid=(0.5,1.0,2.0),noise_grid=(3e-3,1e-2,3e-2,1e-1,3e-1)):
    mu,sd=ya.mean(),ya.std()+1e-9; yn=(ya-mu)/sd; n=len(ya)
    best=(-1e18,None)
    for amp in amp_grid:
        for nz in noise_grid:
            M=amp*Ka+nz*np.eye(n)
            try: c=cho_factor(M,lower=True)
            except Exception: continue
            al=cho_solve(c,yn)
            lml=-0.5*yn@al-np.log(np.diag(c[0])).sum()-0.5*n*np.log(2*np.pi)
            if lml>best[0]: best=(lml,(amp,al))
    amp,al=best[1]
    return (amp*Kb)@al*sd+mu

TARGETS=['eea','ei','eps','nc','egb','egc','tg']
folds=lambda n,s=SEED: list(KFold(NF,shuffle=True,random_state=s).split(np.arange(n)))
out={}
print(f"{'target':6s} {'n':>5s} {'lgb_base':>9s} {'lgb_aug':>9s} {'d_feat':>7s} "
      f"{'gp_aug':>8s} {'pls_aug':>8s} {'BLEND':>8s} {'d_total':>8s}")
for t in TARGETS:
    m=(tr.target_type.values==t); y=tr.target.values[m].astype(np.float64)
    small=m.sum()<600; k=300 if small else 900; nl=15 if small else 63
    Xb=d['Xb_tr'][m]; Xa=d['Xa_tr'][m]
    ob=np.zeros(len(y)); oa=np.zeros(len(y)); og=np.full(len(y),np.nan); op=np.zeros(len(y))
    FP=d['Xb_tr'][m][:,A2]
    # baseline block layout: [desc 217 | morgan2 1024 | morgan3 1024 | maccs 167 | cross 14]
    dense_src=np.hstack([d['Xb_tr'][m][:,:217], d['Xb_tr'][m][:,2432:2446]])
    for a,b in folds(len(y)):
        for X,o in ((Xb,ob),(Xa,oa)):
            idx=sel_infold(X[a],y[a],k,nl) if not small else np.arange(X.shape[1])
            o[b]=lgb_honest(X[a][:,idx],y[a],X[b][:,idx],small)
        # PLS on standardised dense+physics block
        sc=StandardScaler().fit(dense_src[a])
        Za,Zb=sc.transform(dense_src[a]),sc.transform(dense_src[b])
        ncomp=min(20,Za.shape[0]-1,Za.shape[1])
        op[b]=PLSRegression(n_components=ncomp).fit(Za,y[a]).predict(Zb).ravel()
        if small:   # GP: MinMax(Morgan counts) + RBF(dense) sum kernel
            Ka=minmax_K(FP[a],FP[a]); Kb=minmax_K(FP[b],FP[a])
            g=1.0/max(Za.shape[1],1)
            Ra=np.exp(-g*((Za**2).sum(1)[:,None]+(Za**2).sum(1)[None,:]-2*Za@Za.T))
            Rb=np.exp(-g*((Zb**2).sum(1)[:,None]+(Za**2).sum(1)[None,:]-2*Zb@Za.T))
            og[b]=gp_fold(0.5*Ka+0.5*Ra,y[a],0.5*Kb+0.5*Rb)
    rb,ra,rp=r2_score(y,ob),r2_score(y,oa),r2_score(y,op)
    cols=[oa,op]+([og] if small else [])
    bl=np.mean(cols,axis=0)
    rg=r2_score(y,og) if small else np.nan
    rbl=r2_score(y,bl)
    print(f"{t:6s} {m.sum():5d} {rb:9.4f} {ra:9.4f} {ra-rb:+7.4f} "
          f"{rg:8.4f} {rp:8.4f} {rbl:8.4f} {rbl-rb:+8.4f}")
    out[t]=dict(base=rb,aug=ra,gp=rg,pls=rp,blend=rbl); sys.stdout.flush()
pickle.dump(out,open('exp2.pkl','wb'))
print(f"\nmean honest OOF  lgb_base={np.mean([v['base'] for v in out.values()]):.4f}"
      f"  lgb_aug={np.mean([v['aug'] for v in out.values()]):.4f}"
      f"  BLEND={np.mean([v['blend'] for v in out.values()]):.4f}")
