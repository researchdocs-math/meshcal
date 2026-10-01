"""Evaluate calibration with a six-stage numerical-fidelity hierarchy."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration import log_enclosures,score_bounds,bracket,refine,coverage_bracket,quantile_rank
CASES=['square','notch','hole']
MODELS=['coarse','midpoint','neural']

def hierarchy(case,data):
    d=np.load(data/f'{case}.npz');r=np.load(data/f'{case}_reference.npz');X=d['cal_x'];nr,nc,_=X.shape
    L=[];U=[];C=[];T=[]
    for n in d['levels']:
        v=d[f'cal_{n}'];l,u=log_enclosures(v,X.reshape(-1,6));L.append(l);U.append(u);C.append(v[:,3]+v[:,4]+v[:,5]);T.append(v[:,2])
    v=r['cal'];L.append(np.log(v[:,0]));U.append(np.log(v[:,1]));C.append(v[:,4]);T.append(v[:,2])
    raw_l=np.stack(L,1).reshape(nr,nc,6);raw_u=np.stack(U,1).reshape(nr,nc,6)
    l=np.maximum.accumulate(raw_l,axis=-1);u=np.minimum.accumulate(raw_u,axis=-1)
    if np.any(l>u):raise RuntimeError('Disjoint numerical enclosures')
    tl,tu=log_enclosures(d['test_128'],d['test_x']);tl=np.maximum(tl,np.log(r['test'][:,0]));tu=np.minimum(tu,np.log(r['test'][:,1]))
    if np.any(tl>tu):raise RuntimeError('Disjoint test enclosures')
    return d,l,u,np.stack(C,1).reshape(nr,nc,6),np.stack(T,1).reshape(nr,nc,6),tl,tu,raw_l,raw_u

def evaluate_case(case,data,out,predictions=None):
    predictions=out if predictions is None else predictions
    d,L,U,C,T,tl,tu,RL,RU=hierarchy(case,data);nr,nc,_=L.shape
    rows=[];sens=[];honly=[];traces={};allocations={};diagnostics={}
    for model in MODELS:
        z=np.load(predictions/f'{case}_{model}_predictions.npz');pc=z['cal'];pt=z['test']
        trainmesh=8 if model=='coarse' else 32
        txl,txu=log_enclosures(d[f'test_{trainmesh}'],d['test_x']);testlabel=txl if model=='coarse' else (txl+txu)/2
        diagnostics[model]={'test_log_rmse_reference_midpoint':float(np.sqrt(np.mean((pt-(tl+tu)/2)**2))),'test_log_mae_reference_midpoint':float(np.mean(np.abs(pt-(tl+tu)/2)))}
        for rep in range(nr):
            l,u,p,c,t=L[rep],U[rep],pc[rep],C[rep],T[rep]
            def add(method,q,work,seconds,gap=np.nan,converged=False,calls=nc):
                cl,cu,cm=coverage_bracket(pt,q,tl,tu)
                rows.append(dict(case=case,model=model,replicate=rep,method=method,q=q,coverage_lower=cl,coverage_upper=cu,coverage_midpoint=cm,coverage_training_mesh=float(np.mean(np.abs(testlabel-pt)<=q)),relative_width=float(2*np.sinh(q)),work=work,solve_seconds=seconds,gap=gap,converged=converged,calls=calls))
            nl=RL[rep,:,0] if model=='coarse' else (RL[rep,:,2]+RU[rep,:,2])/2
            k=quantile_rank(nc,.1);q=float(np.partition(np.abs(nl-p),k)[k]);add('Naive',q,np.nan,np.nan)
            for label,j in [('Coarse bounds',0),('Fine bounds',4),('Reference bounds',5)]:
                a,b=score_bounds(RL[rep,:,j],RU[rep,:,j],p);qa,qb=bracket(a,b,.1);add(label,qb,float(c[:,j].sum()),float(t[:,j].sum()),qb-qa,qb-qa<=.005)
            q=float(np.partition(np.abs((RL[rep,:,5]+RU[rep,:,5])/2-p),k)[k]);add('Reference midpoint',q,float(c[:,5].sum()),float(t[:,5].sum()))
            for policy in ['rank','width','random','uniform']:
                r=refine(l,u,p,c,policy=policy,seed=1901+rep,tol=.005,trace=rep<5)
                seconds=sum(t[i,:v+1].sum() for i,v in enumerate(r['levels']))
                add(policy.title(),r['q_upper'],r['work'],seconds,r['gap'],r['converged'],r['calls'])
                allocations[f'{model}_{policy}_{rep}']=r['levels']
                if rep<5:traces[f'{model}_{policy}_{rep}']=r['trace']
            for alpha in [.05,.1,.2]:
                for tol in [.002,.005,.01,.02]:
                    for policy in ['rank','uniform']:
                        r=refine(l,u,p,c,alpha=alpha,tol=tol,policy=policy)
                        cl,cu,cm=coverage_bracket(pt,r['q_upper'],tl,tu)
                        sens.append(dict(case=case,model=model,replicate=rep,alpha=alpha,tolerance=tol,policy=policy,q=r['q_upper'],gap=r['gap'],work=r['work'],converged=r['converged'],coverage_lower=cl,coverage_upper=cu,coverage_midpoint=cm,relative_width=2*np.sinh(r['q_upper'])))
            for policy in ['rank','uniform']:
                r=refine(l[:,:5],u[:,:5],p,c[:,:5],tol=.005,policy=policy);cl,cu,cm=coverage_bracket(pt,r['q_upper'],tl,tu)
                honly.append(dict(case=case,model=model,replicate=rep,policy=policy,work=r['work'],gap=r['gap'],converged=r['converged'],coverage_lower=cl,coverage_upper=cu,relative_width=2*np.sinh(r['q_upper'])))
        print('Evaluated',case,model,diagnostics[model],flush=True)
    pd.DataFrame(rows).to_csv(out/f'{case}_metrics.csv',index=False);pd.DataFrame(sens).to_csv(out/f'{case}_sensitivity.csv',index=False);pd.DataFrame(honly).to_csv(out/f'{case}_h_only.csv',index=False)
    np.savez_compressed(out/f'{case}_traces.npz',**traces);np.savez_compressed(out/f'{case}_allocations.npz',**allocations)
    diagnostics['test_reference_log_gap']={'median':float(np.median(tu-tl)),'max':float(np.max(tu-tl))}
    (out/f'{case}_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--out',default='results');ap.add_argument('--case',choices=CASES);args=ap.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for case in ([args.case] if args.case else CASES):rows+=evaluate_case(case,Path(args.data),out)
    if not args.case:pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
if __name__=='__main__':main()
