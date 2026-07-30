import pandas as pd, numpy as np, pickle, sys
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, AllChem, MACCSkeys, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv'); te=pd.read_csv('/home/user/AISEHACK-2.0/test.csv')
T=sorted(tr.target_type.unique())
DF=[(n,f) for n,f in Descriptors.descList]
g2=AllChem.GetMorganGenerator(radius=2,fpSize=1024); g3=AllChem.GetMorganGenerator(radius=3,fpSize=1024)
g2t=AllChem.GetMorganGenerator(radius=2,fpSize=1024)
VDW={'C':20.58,'N':15.60,'O':14.71,'S':24.43,'F':13.31,'Cl':22.45,'Br':26.52,'I':32.52,
     'Si':38.79,'P':24.43,'B':21.0,'*':0.0,'H':7.24}
def vol(m):
    return sum(VDW.get(a.GetSymbol(),18.0) for a in Chem.AddHs(m).GetAtoms())
def _stars(m): return [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]
def oligomer(smi,n=3):
    m=Chem.MolFromSmiles(str(smi))
    if m is None or len(_stars(m))!=2: return None
    combo=m
    for _ in range(n-1): combo=Chem.CombineMols(combo,m)
    rw=Chem.RWMol(combo); na=m.GetNumAtoms(); sl=_stars(m)
    cps=[[s+k*na for s in sl] for k in range(n)]
    def nb(s):
        x=[a.GetIdx() for a in rw.GetAtomWithIdx(s).GetNeighbors()]
        return x[0] if len(x)==1 else None
    for k in range(n-1):
        a,b=nb(cps[k][1]),nb(cps[k+1][0])
        if a is None or b is None: return None
        if rw.GetBondBetweenAtoms(a,b) is None: rw.AddBond(a,b,Chem.BondType.SINGLE)
    keep={cps[0][0],cps[n-1][1]}
    for s in sorted([x for c in cps for x in c if x not in keep],reverse=True): rw.RemoveAtom(s)
    try:
        o=rw.GetMol(); Chem.SanitizeMol(o)
        return Chem.MolFromSmiles(Chem.MolToSmiles(o).replace("*","[H]"))
    except Exception: return None
def conj_max(m):
    if m is None: return 0
    par={}; 
    def find(x):
        while par.get(x,x)!=x: par[x]=par.get(par[x],par[x]); x=par[x]
        return x
    ats=set()
    for b in m.GetBonds():
        if b.GetIsConjugated():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ats.add(i); ats.add(j)
            par.setdefault(i,i); par.setdefault(j,j); ri,rj=find(i),find(j)
            if ri!=rj: par[ri]=rj
    if not ats: return 0
    from collections import Counter
    return max(Counter(find(a) for a in ats).values())
DONOR=["[NX3;H2,H1,H0;!$(N[C,S]=[O,S,N])]","[OX2;!$(O=*);!$(O[C]=O)][CX4]","c1ccsc1","[SX2][CX4]",
       "[cH0]([CX4])","[OX2]c","[NX3]c","c1cc[nH]c1","[Se]","[SiX4]"]
ACCEPT=["[NX3](=O)=O","[CX2]#[NX1]","[CX3]=[OX1]","[CX3](=O)[OX2]","[CX3](=O)[NX3]","[SX4](=O)(=O)",
        "[F][CX4]","c1nsnc1","[CX3]=[NX2]","[n+]","[CX4]([F])([F])[F]","[PX4]=O"]
SMARTS=[Chem.MolFromSmarts(s) for s in DONOR+ACCEPT]
SMARTS=[s for s in SMARTS if s is not None]
GROUPS=["[CX3](=O)[OX2H0][#6]","[CX3](=O)[NX3]","[NX3][CX3](=O)[OX2]","[#6][OX2][#6]","[OX2][CX3](=O)[OX2]",
        "[SX4](=O)(=O)[#6]","O=C1[NX3][CX3](=O)c2ccccc12","[SiX4][OX2]","c1ccccc1","c1ccc2ccccc2c1",
        "[CX4]([CH3])([CH3])","[CX4]([F])([F])[F]","[CX2]#[NX1]","[NX3H2]","[OX2H]","[CX4H2][CX4H2][CX4H2]"]
GSM=[Chem.MolFromSmarts(s) for s in GROUPS]; GSM=[s for s in GSM if s is not None]
def block(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    hv=m.GetNumHeavyAtoms() or 1
    d=[]
    for _,f in DF:
        try: d.append(f(m))
        except Exception: d.append(np.nan)
    d=np.array(d,np.float64)
    a2=np.zeros(1024,np.float32)
    for i,c in g2.GetCountFingerprint(m).GetNonzeroElements().items(): a2[i]=c
    a3=np.zeros(1024,np.float32)
    for i,c in g3.GetCountFingerprint(m).GetNonzeroElements().items(): a3[i]=c
    ma=np.zeros(167,np.int8); DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m),ma)
    try: auto=np.array(rdMolDescriptors.CalcAUTOCORR2D(m),np.float64)
    except Exception: auto=np.zeros(192)
    # ---- physics: Lorentz-Lorenz / Clausius-Mossotti ----
    mr=Crippen.MolMR(m); v=vol(m); ll=mr/v if v>0 else 0.0
    npred=np.sqrt((1+2*ll)/(1-ll)) if 0<ll<1 else 0.0
    tpsa=Descriptors.TPSA(m)
    phys=[mr,v,ll,npred,mr/hv,tpsa/v if v else 0,Crippen.MolLogP(m)/hv,Descriptors.MolWt(m)/v if v else 0]
    # ---- backbone vs side chain ----
    st=_stars(m); bbl=0; bb=set()
    if len(st)==2:
        try:
            p=Chem.GetShortestPath(m,st[0],st[1]); bb=set(p)-set(st); bbl=max(len(p)-2,0)
        except Exception: pass
    sc=[a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in bb and a.GetAtomicNum()>0]
    def sub(idxs):
        if not idxs: return [0.0]*6
        As=[m.GetAtomWithIdx(i) for i in idxs]
        return [len(idxs), sum(a.GetIsAromatic() for a in As)/len(idxs),
                sum(a.IsInRing() for a in As)/len(idxs),
                sum(a.GetAtomicNum() not in (6,1) for a in As)/len(idxs),
                sum(a.GetMass() for a in As),
                sum(a.GetHybridization()==Chem.rdchem.HybridizationType.SP3 for a in As)/len(idxs)]
    bbf=sub(sorted(bb)); scf=sub(sc)
    nrot_bb=sum(1 for b in m.GetBonds() if b.GetBeginAtomIdx() in bb and b.GetEndAtomIdx() in bb
                and not b.IsInRing() and b.GetBondType()==Chem.BondType.SINGLE)
    struct=bbf+scf+[bbl, nrot_bb, nrot_bb/max(bbl,1), len(sc)/hv, bbf[4]/max(scf[4],1e-6)]
    # ---- conjugation ----
    cm=conj_max(m); tri=oligomer(smi,3)
    ct=conj_max(tri) if tri is not None else cm
    conj=[cm, cm/hv, 1.0/max(cm,1), ct, ct/max(cm,1), 1.0/max(ct,1)]
    # ---- SMARTS donor/acceptor + group contribution ----
    sm=[len(m.GetSubstructMatches(s)) for s in SMARTS]
    smn=[x/hv for x in sm]
    gp=[len(m.GetSubstructMatches(s)) for s in GSM]
    gpn=[x/hv for x in gp]
    # ---- trimer view ----
    if tri is not None:
        thv=tri.GetNumHeavyAtoms() or 1
        t2=np.zeros(1024,np.float32)
        for i,c in g2t.GetCountFingerprint(tri).GetNonzeroElements().items(): t2[i]=c
        t2=t2/3.0
        tmr=Crippen.MolMR(tri); tv=vol(tri); tll=tmr/tv if tv>0 else 0.0
        tf=[thv, Descriptors.TPSA(tri)/thv, rdMolDescriptors.CalcFractionCSP3(tri),
            rdMolDescriptors.CalcNumRotatableBonds(tri)/thv, tll,
            np.sqrt((1+2*tll)/(1-tll)) if 0<tll<1 else 0.0,
            sum(a.GetIsAromatic() for a in tri.GetAtoms())/thv,
            rdMolDescriptors.CalcNumAromaticRings(tri)]
    else:
        t2=np.zeros(1024,np.float32); tf=[0.0]*8
    # per-heavy-atom normalisation of the extensive monomer descriptors
    dn=d/hv
    return (np.concatenate([d,a2,a3,ma]).astype(np.float32),
            np.concatenate([auto,phys,struct,conj,sm,smn,gp,gpn,tf,t2,dn]).astype(np.float32))
uniq=pd.unique(np.concatenate([tr.smiles.values,te.smiles.values]))
print(f"unique molecules: {len(uniq)}"); sys.stdout.flush()
BASE={}; NEW={}
for k,s in enumerate(uniq):
    r=block(s)
    if r is None: r=(np.zeros(217+1024+1024+167,np.float32),np.zeros(1,np.float32))
    BASE[s],NEW[s]=r
    if (k+1)%1000==0: print(f"  {k+1}/{len(uniq)}"); sys.stdout.flush()
nd=max(len(v) for v in NEW.values())
def stack(df):
    B=np.array([BASE[s] for s in df.smiles],np.float32)
    N=np.array([NEW[s] if len(NEW[s])==nd else np.zeros(nd,np.float32) for s in df.smiles],np.float32)
    return B,N
Btr,Ntr=stack(tr); Bte,Nte=stack(te)
def clean(X):
    X=np.where(np.isfinite(X),X,0.0); return np.clip(X,-1e6,1e6).astype(np.float32)
Btr,Ntr,Bte,Nte=map(clean,(Btr,Ntr,Bte,Nte))
LUT={t:dict(zip(tr.smiles[tr.target_type==t],tr.target[tr.target_type==t])) for t in T}
MED={t:float(np.median(list(LUT[t].values()))) for t in T}
def cross(sm,tt):
    n=len(sm); V=np.zeros((n,len(T)),np.float32); K=np.zeros((n,len(T)),np.float32)
    for j,t in enumerate(T):
        for i,s in enumerate(sm):
            if s in LUT[t]: V[i,j]=LUT[t][s]; K[i,j]=1.0
            else: V[i,j]=MED[t]
    for i,x in enumerate(tt): j=T.index(x); V[i,j]=MED[x]; K[i,j]=0.0
    return np.hstack([V,K])
Ctr=cross(tr.smiles.values,tr.target_type.values); Cte=cross(te.smiles.values,te.target_type.values)
Xb_tr=np.hstack([Btr,Ctr]); Xb_te=np.hstack([Bte,Cte])
Xa_tr=np.hstack([Btr,Ctr,Ntr]); Xa_te=np.hstack([Bte,Cte,Nte])
v=Xa_tr.var(0); keep=v>1e-10
print("baseline",Xb_tr.shape,"augmented",Xa_tr.shape,"-> kept",keep.sum())
pickle.dump({'Xb_tr':Xb_tr,'Xb_te':Xb_te,'Xa_tr':Xa_tr[:,keep],'Xa_te':Xa_te[:,keep],
             'nbase':Xb_tr.shape[1],'T':T},open('feats2.pkl','wb'))
print("saved")
