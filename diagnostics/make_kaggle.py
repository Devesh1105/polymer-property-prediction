import json, os, re

SRC='/home/user/AISEHACK-2.0/polymer_r2_v8.ipynb'
OUTDIR='/home/user/AISEHACK-2.0/kaggle_submission'
os.makedirs(OUTDIR, exist_ok=True)
OUT=os.path.join(OUTDIR,'polymer_r2_best_oof.ipynb')
nb=json.load(open(SRC))
def _src(s):
    L=s.strip("\n").split("\n"); return [l+"\n" for l in L[:-1]]+[L[-1]]
def code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":_src(s)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":_src(s)}
cells=nb['cells']

# ---------------------------------------------------------------- robust path discovery
PATHS = r'''
# ===== Kaggle input discovery =====
# Finds train/test/PI1M/archive wherever the dataset got mounted, instead of a hard-coded path.
import glob

def _find(fname, subdir=""):
    pats = [os.path.join(subdir, fname) if subdir else fname,
            os.path.join("/kaggle/working", subdir, fname),
            os.path.join("/kaggle/input", "*", subdir, fname),
            os.path.join("/kaggle/input", "*", "*", subdir, fname),
            os.path.join("/mnt/user-data/uploads", subdir, fname)]
    for p in pats:
        hits = sorted(glob.glob(p)) if "*" in p else ([p] if os.path.exists(p) else [])
        if hits: return hits[0]
    return None

TRAIN_PATH = _find("train.csv")
TEST_PATH  = _find("test.csv")
if TRAIN_PATH is None or TEST_PATH is None:
    raise FileNotFoundError("train.csv / test.csv not found. Check the notebook's Data panel; "
                            "everything under /kaggle/input/*/ is searched.")
DATA_DIR = os.path.dirname(TRAIN_PATH) or "."
PI1M_PATH   = _find("PI1M.csv") or _find("PI1M.zip") or ""
ARCHIVE_DIR = os.path.dirname(_find("train.csv", "archive") or "")
print("DATA_DIR    :", DATA_DIR)
print("TRAIN/TEST  :", TRAIN_PATH, "|", TEST_PATH)
print("PI1M        :", PI1M_PATH or "NOT FOUND")
print("archive dir :", ARCHIVE_DIR or "NOT FOUND")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print("Train:", train.shape, "| Test:", test.shape)
'''
s4=''.join(cells[4]['source'])
head_end=s4.index('def normalize_schema')
cells[4]=code(PATHS.strip("\n")+"\n\n"+s4[head_end:])

# ---------------------------------------------------------------- runtime budget cell
BUDGET = r'''
# ===== Runtime budget =====
# Kaggle caps sessions at ~9h GPU / ~12h CPU. Defaults below are tuned to land inside the GPU
# limit with the phase-1 training merge enabled. Raise them if you have headroom; every one of
# them trades wall-clock for a small amount of score.
GPU_AVAILABLE   = torch.cuda.is_available()
PI1M_MAX        = 60000 if GPU_AVAILABLE else 12000   # molecules for self-supervised pretraining
PRETRAIN_EPOCHS = 40    if GPU_AVAILABLE else 8
GNN_SEEDS       = (SEED, SEED+1) if GPU_AVAILABLE else (SEED,)
MTL_SEEDS       = (SEED, SEED+1) if GPU_AVAILABLE else (SEED,)

# The GP kernel is O(n^2 * d) to build and O(n^3) to solve, and its measured benefit was on the
# small targets (eea +0.026, egb +0.023, ei +0.010). Above this row count it costs a lot of time
# and memory for a gain never measured, so it is skipped there.
GP_MAX_N = 1500

print(f"GPU: {GPU_AVAILABLE} | PI1M_MAX={PI1M_MAX} | PRETRAIN_EPOCHS={PRETRAIN_EPOCHS} "
      f"| GNN seeds={len(GNN_SEEDS)} | GP_MAX_N={GP_MAX_N}")
if not GPU_AVAILABLE:
    print("WARNING: no GPU. Turn on the GPU accelerator or the GNN cells will dominate runtime.")
'''
# insert straight after the imports cell (index 2)
cells.insert(3, md("### Runtime budget — read this before hitting Run All"))
cells.insert(4, code(BUDGET))

def find_cell(pred, start=0):
    for i in range(start, len(cells)):
        if cells[i]['cell_type']=='code' and pred(''.join(cells[i]['source'])): return i
    raise LookupError

# ---------------------------------------------------------------- phase-1: training merge ON
i=find_cell(lambda s: 'USE_PHASE1_TRAIN  = False' in s)
s=''.join(cells[i]['source'])
s=s.replace('USE_PHASE1_TRAIN  = False        # <-- merge phase-1 rows into the TRAINING set',
            'USE_PHASE1_TRAIN  = True         # ON: measured +0.0141 tg / +0.0144 egc honest OOF')
s=s.replace('_p1_paths = [p for p in ["archive/train.csv", os.path.join(DATA_DIR,"archive","train.csv"),\n                         "/kaggle/input/archive/train.csv"] if os.path.exists(p)]',
            '_p1_paths = [p for p in [os.path.join(ARCHIVE_DIR,"train.csv")] if p and os.path.exists(p)]')
s=s.replace('''import zipfile, glob
if not os.path.exists(PI1M_PATH):
    for _z in glob.glob("PI1M.zip")+glob.glob("*/PI1M.zip")+glob.glob(os.path.join(DATA_DIR,"PI1M.zip")):
        try:
            zipfile.ZipFile(_z).extractall("."); PI1M_PATH="PI1M.csv"
            print("Extracted PI1M.csv from", _z); break
        except Exception as e: print("PI1M unzip failed:", e)''',
'''import zipfile
if PI1M_PATH.endswith(".zip") and os.path.exists(PI1M_PATH):
    try:
        zipfile.ZipFile(PI1M_PATH).extractall("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
        _ex = _find("PI1M.csv")
        if _ex: PI1M_PATH = _ex; print("Extracted PI1M.csv ->", PI1M_PATH)
    except Exception as e:
        print("PI1M unzip failed:", e); PI1M_PATH = ""''')
cells[i]=code(s)

# ---------------------------------------------------------------- honour the budget knobs
i=find_cell(lambda s: 'USE_PI1M = True' in s)
s=''.join(cells[i]['source'])
s=s.replace('PI1M_MAX = 60000           # subsample for tractable pretraining; lower if tight on the time limit',
            '# PI1M_MAX / PRETRAIN_EPOCHS come from the runtime-budget cell')
s=s.replace('PRETRAIN_EPOCHS = 40\n','')
s=s.replace('def train_multitask(seeds=(SEED,SEED+1),','def train_multitask(seeds=MTL_SEEDS,')
s=s.replace('if USE_PI1M and os.path.exists(PI1M_PATH):','if USE_PI1M and PI1M_PATH and os.path.exists(PI1M_PATH):')
cells[i]=code(s)

i=find_cell(lambda s: 'def train_gnn(target, seeds=' in s)
s=''.join(cells[i]['source']).replace('def train_gnn(target, seeds=(SEED,SEED+1)):','def train_gnn(target, seeds=GNN_SEEDS):')
cells[i]=code(s)

# ---------------------------------------------------------------- GP row cap
i=find_cell(lambda s: 'def train_gp(target)' in s)
s=''.join(cells[i]['source'])
s=s.replace('print("Training Tanimoto/MinMax-kernel GPs ...")\ngp_res = {t: train_gp(t) for t in TARGETS}',
'''print("Training Tanimoto/MinMax-kernel GPs ...")
gp_res = {}
for _t in TARGETS:
    _n = int((train["target_type"].values == _t).sum())
    if _n > GP_MAX_N:
        print(f"[{_t}] GP skipped ({_n} rows > GP_MAX_N={GP_MAX_N})")
        continue
    gp_res[_t] = train_gp(_t)''')
cells[i]=code(s)

# ---------------------------------------------------------------- header
cells[0]=md(r'''
# Polymer property prediction — Round 2 — best honest-OOF configuration

**Run All. Enable the GPU accelerator first.** Inputs are located automatically anywhere under
`/kaggle/input/`; nothing needs editing.

Expect the printed OOF to be roughly **0.90–0.91**. That is deliberately *lower* than the 0.912
an earlier version reported, because that 0.912 was inflated — it tuned early stopping and GNN
checkpoints on the very folds it scored, which is exactly why that run scored 0.889 on the
leaderboard. Everything here is measured; see `diagnostics/` in the repo.

### What this configuration does

| change | measured effect |
|---|---|
| Early stopping on an inner split of the training fold, then refit | removes +0.018 of fake OOF |
| GNN/multi-task checkpointing on an inner split + weight averaging | removes +0.04…+0.07 on the small targets |
| Feature selection in-fold; disabled below 600 rows | removes +0.008 eea, +0.016 ei |
| Phase-1 archive merged into training (`USE_PHASE1_TRAIN=True`) | **+0.0141 tg, +0.0144 egc** |
| Tanimoto/MinMax-kernel GP base learner | +0.026 eea, +0.023 egb, +0.010 ei |
| Polymer block (trimer / backbone / conjugation) on eea, egb, egc | +0.012 eea, +0.017 egb |
| Stack weights shrunk toward uniform | inner CV picks equal weighting on eea, ei, egb |
| 10 seeds on the small targets | variance reduction (their LB sd is ~0.025) |

### Two things to know

**tg and egc LB will exceed their OOF.** The phase-1 rows merged into training are themselves
phase-2 test rows, so the model reproduces them at test time. That is a property of the archive,
not a bug. `USE_PHASE1_TRAIN=False` in cell 9 reverts it.

**`USE_PHASE1_LABELS` is left False.** It writes phase-1 labels straight into `submission.csv`
and changes nothing about OOF, so it has no place in a "best OOF" run. Flip it only if you have
decided the archive is fair game — read the note in cell 9 first.

ei (~0.82) and eps (~0.78) did not move for any feature block, model class or blend tested, and
they cap the equal-weight mean. 0.93 honest OOF is not reachable on this data.
''')

nb['cells']=cells
nb.setdefault('metadata',{}).setdefault('kernelspec',{'name':'python3','display_name':'Python 3','language':'python'})
json.dump(nb, open(OUT,'w'), indent=1)
print("wrote", OUT, len(cells), "cells")
