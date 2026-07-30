import numpy as np
rng=np.random.default_rng(1)
TEST={'eea':147,'egb':224,'egc':1352,'ei':148,'eps':153,'nc':153,'tg':2763}
R2={'eea':0.929,'egb':0.950,'egc':0.933,'ei':0.857,'eps':0.860,'nc':0.909,'tg':0.924}
def sd_r2(n,r2,reps=20000):
    rho=np.sqrt(r2); out=np.empty(reps)
    for i in range(reps):
        y=rng.standard_normal(n); p=rho*y+np.sqrt(1-r2)*rng.standard_normal(n)
        out[i]=1-((p-y)**2).sum()/((y-y.mean())**2).sum()
    return out.std()
print("Sampling sd of per-target R2 on the FULL test set (and on a 50% public split)\n")
print(f"{'target':6s} {'n_test':>7s} {'R2':>6s} {'sd(full)':>9s} {'sd(50%)':>8s}")
sds=[];sds50=[]
for t in sorted(TEST):
    s=sd_r2(TEST[t],R2[t]); s50=sd_r2(max(5,TEST[t]//2),R2[t])
    sds.append(s); sds50.append(s50)
    print(f"{t:6s} {TEST[t]:7d} {R2[t]:6.3f} {s:9.4f} {s50:8.4f}")
print(f"\nsd of the 7-target MEAN: full test = {np.sqrt(sum(np.array(sds)**2))/7:.4f}"
      f" | 50% public = {np.sqrt(sum(np.array(sds50)**2))/7:.4f}")
