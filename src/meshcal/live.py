"""Fresh-solve calibration benchmark, independent of the response cache."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from .fem import solve,spaces
from .reference import reference_solve,reference_spaces
from .reference_generate import GRADING
from .calibration import log_enclosures,refine_query,score_bounds,bracket
LEVELS=[8,16,32,64,128]

def run(task):
    case,model,rep,policy,datadir,resultdir=task;d=np.load(Path(datadir)/f'{case}.npz');X=d['cal_x'][rep];pred=np.load(Path(resultdir)/f'{case}_{model}_predictions.npz')['cal'][rep];n=len(X)
    # Warm fixed spaces before timing, equally for every policy.
    cost=[]
    for level in LEVELS:
        mesh,p,rt,z,_,_=spaces(case,level);cost.append(p.N+rt.N+z.N)
    _,p,rt,z,_,_=reference_spaces(case,grading=GRADING[case]);cost.append(p.N+rt.N+z.N);cost=np.tile(cost,(n,1))
    requests=[]
    def query(i,j):
        requests.append((i,j))
        if j==5:
            r=reference_solve(case,X[i],grading=GRADING[case]);return np.log(r['lower']),np.log(r['upper'])
        r=solve(case,LEVELS[j],X[i]);v=np.array([[r[k] for k in ['lower','upper','seconds','primal_dofs','flux_dofs','triangles','residual']]])
        a,b=log_enclosures(v,X[i:i+1]);return float(a[0]),float(b[0])
    t=time.perf_counter()
    if policy=='reference':
        labels=np.array([query(i,5) for i in range(n)]);a,b=score_bounds(labels[:,0],labels[:,1],pred);ql,qh=bracket(a,b,.1);r=dict(q_lower=ql,q_upper=qh,work=float(cost[:,5].sum()),calls=n,gap=qh-ql)
    else:r=refine_query(n,6,pred,cost,query,policy=policy,tol=.005)
    seconds=time.perf_counter()-t
    cached=pd.read_csv(Path(resultdir)/f'{case}_metrics.csv');method='Reference bounds' if policy=='reference' else policy.title();row=cached[(cached.model==model)&(cached.replicate==rep)&(cached.method==method)].iloc[0]
    difference=abs(r['q_upper']-row.q)
    if difference>1e-8 or r['work']!=row.work:raise AssertionError((case,model,rep,policy,difference,r['work'],row.work))
    return dict(case=case,model=model,replicate=rep,policy=policy,wall_seconds=seconds,work=r['work'],calls=r['calls'],gap=r['gap'],q=r['q_upper'],cache_difference=difference)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--results',default='results');ap.add_argument('--workers',type=int,default=12);ap.add_argument('--replicates',type=int,default=2);args=ap.parse_args();tasks=[(c,m,r,p,str(Path(args.data).resolve()),str(Path(args.results).resolve())) for c in ['square','notch','hole'] for m in ['coarse','midpoint'] for r in range(args.replicates) for p in ['rank','uniform','reference']];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(run,tasks,chunksize=1))
    pd.DataFrame(rows).to_csv(Path(args.results)/'live.csv',index=False);meta=dict(workers=args.workers,replicates=args.replicates,wall_seconds=time.perf_counter()-t,simultaneous_runs=True,space_construction_timed=False);(Path(args.results)/'live_metadata.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
if __name__=='__main__':main()
