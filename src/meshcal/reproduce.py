"""Reproduce all numerical comparisons with frozen predictors."""
import argparse,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd

def call(module,*args):subprocess.run([sys.executable,'-m','meshcal.'+module,*map(str,args)],check=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['replay','full'],default='replay');ap.add_argument('--data',default='data');ap.add_argument('--out',default='results-reproduced');ap.add_argument('--workers',type=int,default=12);ap.add_argument('--live',action='store_true');args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.mode=='full':
        call('generate','--out',args.data,'--workers',args.workers);call('reference_generate','--data',args.data,'--workers',args.workers)
    for pattern in ['*_predictions.npz','*_neural_weights.npz','*_models.json']:
        for p in Path('results').glob(pattern):
            if p.resolve()!=(out/p.name).resolve():shutil.copy2(p,out/p.name)
    call('models','--data',args.data,'--results',out)
    call('evaluate','--data',args.data,'--out',out)
    expected=pd.read_csv('results/metrics.csv');actual=pd.read_csv(out/'metrics.csv');keys=['case','model','replicate','method'];a=actual.sort_values(keys).reset_index(drop=True);e=expected.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(e[keys])
    for col in ['q','coverage_lower','coverage_upper','relative_width','work','gap','calls']:
        np.testing.assert_allclose(a[col],e[col],rtol=1e-6,atol=1e-8,equal_nan=True,err_msg=col)
    print('All numerical comparisons agree with the archived results. Timing fields are platform dependent.')
    if args.live:call('live','--data',args.data,'--results',out,'--workers',args.workers)
if __name__=='__main__':main()
