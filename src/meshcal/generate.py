"""Generate independent training, calibration and test finite-element data."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse,json,platform,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .fem import solve

LEVELS=np.array([8,16,32,64,128])
CASES=['square','notch','hole']
FIELDS=['lower','upper','seconds','primal_dofs','flux_dofs','triangles','residual']
def task(t):
    case,n,theta=t
    r=solve(case,n,theta)
    return [r[k] for k in FIELDS]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='data');ap.add_argument('--workers',type=int,default=12);ap.add_argument('--replicates',type=int,default=30);ap.add_argument('--ncal',type=int,default=199);ap.add_argument('--ntest',type=int,default=2000);ap.add_argument('--ntrain',type=int,default=512);args=ap.parse_args()
    dest=Path(args.out);dest.mkdir(exist_ok=True,parents=True)
    wall=time.perf_counter()
    for ci,case in enumerate(CASES):
        rng=np.random.default_rng(720026+ci)
        # All sampled rows are independent uniform draws, with disjoint partitions.
        Xtr=rng.uniform(-1.3,1.3,(args.ntrain,6));Xcal=rng.uniform(-1.3,1.3,(args.replicates,args.ncal,6));Xte=rng.uniform(-1.3,1.3,(args.ntest,6))
        arrays={'train_x':Xtr,'cal_x':Xcal,'test_x':Xte,'levels':LEVELS}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for split,X,levs in [('train',Xtr,[8,32]),('cal',Xcal.reshape(-1,6),LEVELS),('test',Xte,[8,32,128])]:
                for n in levs:
                    path=dest/f'{case}_{split}_{n}.npz'
                    if path.exists():
                        data=np.load(path);R=data['results'];assert np.array_equal(X,data['x'])
                    else:
                        t=time.perf_counter()
                        R=np.asarray(list(pool.map(task,((case,int(n),x) for x in X),chunksize=4)))
                        np.savez_compressed(path,x=X,results=R,fields=FIELDS)
                        print(f'{case} {split} n={n}: {len(X)} solves, {time.perf_counter()-t:.2f}s wall, max residual={R[:,-1].max():.2e}',flush=True)
                    arrays[f'{split}_{n}']=R
        np.savez_compressed(dest/f'{case}.npz',**arrays)
    meta=dict(seed_base=720026,workers=args.workers,replicates=args.replicates,ncal=args.ncal,ntest=args.ntest,ntrain=args.ntrain,levels=LEVELS.tolist(),fields=FIELDS,wall_seconds=time.perf_counter()-wall,platform=platform.platform(),python=platform.python_version(),machine=platform.machine())
    (dest/'generation.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
if __name__=='__main__':main()
