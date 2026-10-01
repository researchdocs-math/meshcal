"""Generate higher-order reference and terminal calibration enclosures."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json,time,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .reference import reference_solve
GRADING={'square':4,'notch':12,'hole':4}
FIELDS=['lower','upper','seconds','triangles','dofs','residual_l2']
def task(args):
    case,x=args;r=reference_solve(case,x,grading=GRADING[case]);return [r[k] for k in FIELDS]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--workers',type=int,default=12);args=ap.parse_args();p=Path(args.data);start=time.perf_counter();times={}
    for case in GRADING:
        d=np.load(p/f'{case}.npz');arrays={}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for split in ['cal','test']:
                X=d[split+'_x'].reshape(-1,6);path=p/f'{case}_{split}_reference.npz';t=time.perf_counter()
                if path.exists():
                    z=np.load(path);assert np.array_equal(z['x'],X);R=z['results']
                else:
                    R=np.array(list(pool.map(task,((case,x) for x in X),chunksize=4)));np.savez_compressed(path,x=X,results=R,fields=FIELDS,grading=GRADING[case])
                arrays[split]=R;times[case+'_'+split]=time.perf_counter()-t
                print(case,split,len(X),'seconds',times[case+'_'+split],'log gap max',np.log(R[:,1]/R[:,0]).max(),flush=True)
        np.savez_compressed(p/f'{case}_reference.npz',**arrays,grading=GRADING[case])
    (p/'reference_generation.json').write_text(json.dumps(dict(grading=GRADING,base_n=32,workers=args.workers,wall_seconds=time.perf_counter()-start,split_seconds=times,platform=platform.platform()),indent=2))
if __name__=='__main__':main()
