"""Optional Metal-accelerated ensemble training; portable NumPy inference."""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from .calibration import log_enclosures

def predict(weights,x):
    z=np.broadcast_to(x/1.3,(4,len(x),6)).copy()
    for i in range(3):
        z=np.einsum('bni,bio->bno',z,weights[f'w{i}'].astype(float))+weights[f'b{i}'].astype(float)
        if i<2:z=np.tanh(z)
    return z[:,:,0].mean(axis=0)*float(weights['yscale'])+float(weights['ymean'])

def train(case,data,out,epochs=6000):
    out.mkdir(exist_ok=True,parents=True)
    import torch
    torch.set_num_threads(1);torch.manual_seed(9172)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    d=np.load(data/f'{case}.npz');X=d['train_x'];l,u=log_enclosures(d['train_32'],X);y=(l+u)/2;ym=y.mean();ys=y.std()
    path=out/f'{case}_neural_weights.npz'
    if not path.exists():
        layers=[6,64,64,1];params=[]
        for a,b in zip(layers[:-1],layers[1:]):
            w=torch.empty(4,a,b);torch.nn.init.uniform_(w,-np.sqrt(6/(a+b)),np.sqrt(6/(a+b)))
            params.extend([torch.nn.Parameter(w.to(device)),torch.nn.Parameter(torch.zeros(4,1,b,device=device))])
        xt=torch.tensor(X/1.3,dtype=torch.float32,device=device).expand(4,-1,-1);yt=torch.tensor((y-ym)/ys,dtype=torch.float32,device=device)
        opt=torch.optim.Adam(params,lr=.003);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00003)
        t=time.perf_counter()
        for epoch in range(epochs):
            z=xt
            for i in range(3):
                z=torch.bmm(z,params[2*i])+params[2*i+1]
                if i<2:z=torch.tanh(z)
            loss=((z[:,:,0]-yt)**2).mean();opt.zero_grad();loss.backward();opt.step();sched.step()
        if device.type=='mps':torch.mps.synchronize()
        meta=dict(device=str(device),epochs=epochs,seconds=time.perf_counter()-t,training_mse=float(loss.detach().cpu()),seed=9172,architecture=[6,64,64,1],members=4,torch=torch.__version__)
        weights={}
        for i in range(3):weights[f'w{i}']=params[2*i].detach().cpu().numpy();weights[f'b{i}']=params[2*i+1].detach().cpu().numpy()
        np.savez_compressed(path,**weights,ymean=ym,yscale=ys,metadata=json.dumps(meta))
    w=np.load(path);pc=predict(w,d['cal_x'].reshape(-1,6)).reshape(d['cal_x'].shape[:2]);pt=predict(w,d['test_x'])
    np.savez_compressed(out/f'{case}_neural_predictions.npz',cal=pc,test=pt,metadata=w['metadata'])
    print(case,str(w['metadata']),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--out',default='results');ap.add_argument('--case',default='square');ap.add_argument('--epochs',type=int,default=6000);args=ap.parse_args();train(args.case,Path(args.data),Path(args.out),args.epochs)
if __name__=='__main__':main()
