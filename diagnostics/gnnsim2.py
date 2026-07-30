import numpy as np
rng=np.random.default_rng(0)
# Residual = shared irreducible error (same every epoch) + epoch jitter (only this is selectable).
# phi = fraction of residual variance that actually moves epoch-to-epoch near the plateau.
def sim(n,E,rho,phi,folds=8,reps=250,ar=0.85):
    tot_g=[]
    for _ in range(reps):
        y=rng.standard_normal(n); sel=np.zeros(n); fix=np.zeros(n)
        for c in np.array_split(rng.permutation(n),folds):
            m=len(c); resid_sd=np.sqrt(1-rho**2)
            shared=resid_sd*np.sqrt(1-phi)*rng.standard_normal(m)
            base=rho*y[c]+shared
            prev=np.zeros(m); traj=np.empty((E,m))
            for i in range(E):
                prev=ar*prev+np.sqrt(1-ar**2)*rng.standard_normal(m)
                traj[i]=base+resid_sd*np.sqrt(phi)*prev
            ss=((traj-y[c])**2).sum(1)
            sel[c]=traj[int(np.argmin(ss))]          # their code: best val R2 checkpoint
            fix[c]=traj[rng.integers(E//2,E)]        # honest: budget/inner-split -> a typical plateau epoch
        d=((y-y.mean())**2).sum()
        tot_g.append(((fix-y)**2).sum()/d-((sel-y)**2).sum()/d)
    return np.mean(tot_g)
print("OOF R2 inflation from keeping the best-validation-R2 epoch (E=180 epochs, rho=0.92)")
print("phi = share of residual variance that jitters between epochs\n")
print(f"{'target':22s} {'fold n':>7s} "+" ".join(f"{'phi='+str(p):>9s}" for p in (0.10,0.20,0.35)))
for name,n in [('eea/ei/eps/nc (~222)',222),('egb (337)',337),('egc (2028)',2028),('tg (4143)',4143)]:
    r=[sim(n,180,0.92,p,reps=200 if n<1000 else 40) for p in (0.10,0.20,0.35)]
    print(f"{name:22s} {n//8:7d} "+" ".join(f"{x:+9.4f}" for x in r))
