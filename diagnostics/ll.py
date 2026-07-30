import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr
RDLogger.DisableLog("rdApp.*")
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv')
# Lorentz-Lorenz:  (n^2-1)/(n^2+2) = MolarRefraction / MolarVolume
# Clausius-Mossotti (same form) for eps. Volume proxy from Crippen MR's partner: LabuteASA / vdW.
VDW={'C':20.58,'N':15.60,'O':14.71,'S':24.43,'F':13.31,'Cl':22.45,'Br':26.52,'I':32.52,
     'Si':38.79,'P':24.43,'B':21.0,'*':0.0,'H':7.24}
def vol(m):
    m=Chem.AddHs(m)
    return sum(VDW.get(a.GetSymbol(),18.0) for a in m.GetAtoms())
def feats(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    mr=Crippen.MolMR(m); v=vol(m)
    ll=mr/v if v>0 else np.nan
    n_pred=np.sqrt((1+2*ll)/(1-ll)) if 0<ll<1 else np.nan
    tpsa=Descriptors.TPSA(m); hv=m.GetNumHeavyAtoms() or 1
    return dict(mr=mr, vol=v, ll=ll, n_pred=n_pred, mr_per_hv=mr/hv,
                polar_density=tpsa/v, logp_density=Crippen.MolLogP(m)/hv)
for t in ('nc','eps'):
    sub=tr[tr.target_type==t]
    rows=[feats(s) for s in sub.smiles]
    ok=[i for i,r in enumerate(rows) if r is not None and np.isfinite(r['n_pred'])]
    y=sub.target.values[ok]
    print(f"--- {t}  (n={len(ok)}) ---")
    for k in ('ll','n_pred','mr_per_hv','polar_density','logp_density','mr'):
        x=np.array([rows[i][k] for i in ok])
        g=np.isfinite(x)
        print(f"   {k:15s} pearson={pearsonr(x[g],y[g])[0]:+.3f}  spearman={spearmanr(x[g],y[g])[0]:+.3f}"
              f"  R2_univariate={pearsonr(x[g],y[g])[0]**2:.3f}")
