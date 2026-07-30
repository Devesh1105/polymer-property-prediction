# Recompute ONLY the compact interpretable blocks, saved separately so each can be tested alone.
import pandas as pd, numpy as np, pickle, sys
exec(open('feats2.py').read().split('def block(')[0].split('uniq=')[0])
def blocks(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    hv=m.GetNumHeavyAtoms() or 1
    mr=Crippen.MolMR(m); v=vol(m); ll=mr/v if v>0 else 0.0
    npred=np.sqrt((1+2*ll)/(1-ll)) if 0<ll<1 else 0.0
    tpsa=Descriptors.TPSA(m)
    PHYS=[mr,v,ll,npred,mr/hv,tpsa/v if v else 0,Crippen.MolLogP(m)/hv,Descriptors.MolWt(m)/v if v else 0]
    st=_stars(m); bbl=0; bb=set()
    if len(st)==2:
        try: p=Chem.GetShortestPath(m,st[0],st[1]); bb=set(p)-set(st); bbl=max(len(p)-2,0)
        except Exception: pass
    sc=[a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in bb and a.GetAtomicNum()>0]
    def sub(ix):
        if not ix: return [0.0]*6
        As=[m.GetAtomWithIdx(i) for i in ix]
        return [len(ix),sum(a.GetIsAromatic() for a in As)/len(ix),sum(a.IsInRing() for a in As)/len(ix),
                sum(a.GetAtomicNum() not in (6,1) for a in As)/len(ix),sum(a.GetMass() for a in As),
                sum(a.GetHybridization()==Chem.rdchem.HybridizationType.SP3 for a in As)/len(ix)]
    bbf,scf=sub(sorted(bb)),sub(sc)
    nrb=sum(1 for b in m.GetBonds() if b.GetBeginAtomIdx() in bb and b.GetEndAtomIdx() in bb
            and not b.IsInRing() and b.GetBondType()==Chem.BondType.SINGLE)
    STRUCT=bbf+scf+[bbl,nrb,nrb/max(bbl,1),len(sc)/hv,bbf[4]/max(scf[4],1e-6)]
    cm=conj_max(m); tri=oligomer(smi,3); ct=conj_max(tri) if tri is not None else cm
    CONJ=[cm,cm/hv,1.0/max(cm,1),ct,ct/max(cm,1),1.0/max(ct,1)]
    if tri is not None:
        thv=tri.GetNumHeavyAtoms() or 1; tmr=Crippen.MolMR(tri); tv=vol(tri); tll=tmr/tv if tv>0 else 0.0
        TRI=[thv,Descriptors.TPSA(tri)/thv,rdMolDescriptors.CalcFractionCSP3(tri),
             rdMolDescriptors.CalcNumRotatableBonds(tri)/thv,tll,
             np.sqrt((1+2*tll)/(1-tll)) if 0<tll<1 else 0.0,
             sum(a.GetIsAromatic() for a in tri.GetAtoms())/thv,rdMolDescriptors.CalcNumAromaticRings(tri)]
    else: TRI=[0.0]*8
    SM=[len(m.GetSubstructMatches(s))/hv for s in SMARTS]
    GPc=[len(m.GetSubstructMatches(s))/hv for s in GSM]
    try: AUTO=list(rdMolDescriptors.CalcAUTOCORR2D(m))
    except Exception: AUTO=[0.0]*192
    return dict(PHYS=PHYS,STRUCT=STRUCT,CONJ=CONJ,TRI=TRI,SMARTS=SM+GPc,AUTO=AUTO)
tr=pd.read_csv('/home/user/AISEHACK-2.0/train.csv'); te=pd.read_csv('/home/user/AISEHACK-2.0/test.csv')
uniq=pd.unique(np.concatenate([tr.smiles.values,te.smiles.values]))
R={}
for k,s in enumerate(uniq):
    R[s]=blocks(s)
    if (k+1)%2000==0: print(f"  {k+1}/{len(uniq)}"); sys.stdout.flush()
KEYS=['PHYS','STRUCT','CONJ','TRI','SMARTS','AUTO']
dims={k:len(next(v[k] for v in R.values() if v)) for k in KEYS}
print("block dims:",dims)
def mat(df,k):
    z=[0.0]*dims[k]
    return np.nan_to_num(np.array([R[s][k] if R[s] else z for s in df.smiles],np.float32),
                         posinf=0,neginf=0).clip(-1e6,1e6)
pickle.dump({'tr':{k:mat(tr,k) for k in KEYS},'te':{k:mat(te,k) for k in KEYS},'dims':dims},
            open('feats3.pkl','wb'))
print("saved")
