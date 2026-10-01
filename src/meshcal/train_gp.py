"""Refit the two Gaussian-process baselines from the training data."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF
from .calibration import log_enclosures

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--out',default='results-refitted');args=ap.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    for case in ['square','notch','hole']:
        d=np.load(Path(args.data)/f'{case}.npz');X=d['train_x']
        for model,n in [('coarse',8),('midpoint',32)]:
            l,u=log_enclosures(d[f'train_{n}'],X);y=l if model=='coarse' else (l+u)/2
            gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,100))*RBF(np.ones(6),(.2,20)),alpha=1e-8,normalize_y=True,n_restarts_optimizer=2,random_state=461);t=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:gp.fit(X,y)
            meta=dict(kernel=str(gp.kernel_),hyperparameters=gp.kernel_.theta.tolist(),training_seconds=time.perf_counter()-t,warnings=[str(w.message) for w in ws]);cal=gp.predict(d['cal_x'].reshape(-1,6)).reshape(d['cal_x'].shape[:2]);test=gp.predict(d['test_x']);np.savez_compressed(out/f'{case}_{model}_predictions.npz',cal=cal,test=test,metadata=json.dumps(meta));print(case,model,meta,flush=True)
if __name__=='__main__':main()
