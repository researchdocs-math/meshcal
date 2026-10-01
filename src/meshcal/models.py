"""Portable GP inference from recorded fitted kernel parameters."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .calibration import log_enclosures

def gp_predict(d,metadata,model,x):
    tr=d['train_x'];n=8 if model=='coarse' else 32;l,u=log_enclosures(d[f'train_{n}'],tr);y=l if model=='coarse' else (l+u)/2
    hp=np.exp(np.asarray(metadata['hyperparameters']));amplitude=hp[0];length=hp[1:]
    def kernel(a,b):return amplitude*np.exp(-.5*np.sum(((a[:,None,:]-b[None,:,:])/length)**2,axis=2))
    K=kernel(tr,tr)+1e-8*np.eye(len(tr));alpha=cho_solve(cho_factor(K,lower=True),(y-y.mean())/y.std())
    return kernel(x,tr)@alpha*y.std()+y.mean()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--results',default='results');args=ap.parse_args();data=Path(args.data);out=Path(args.results)
    report={}
    from .neural import predict
    for case in ['square','notch','hole']:
        d=np.load(data/f'{case}.npz')
        for model in ['coarse','midpoint','neural']:
            saved=np.load(out/f'{case}_{model}_predictions.npz');meta=json.loads(str(saved['metadata']));X=np.concatenate([d['cal_x'].reshape(-1,6),d['test_x']])
            y=predict(np.load(out/f'{case}_neural_weights.npz'),X) if model=='neural' else gp_predict(d,meta,model,X)
            expected=np.r_[saved['cal'].ravel(),saved['test']];error=float(np.max(np.abs(y-expected)));report[case+'_'+model]=error
            if error>1e-7:raise AssertionError((case,model,error))
    (out/'prediction_verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
