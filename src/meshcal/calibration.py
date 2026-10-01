"""Conformal quantile brackets and adaptive refinement of deterministic labels."""
from __future__ import annotations
import numpy as np

def log_enclosures(results,x):
    """Correct flux energy for equilibrium residual and enclose log compliance.

    For a domain contained in (0,1)^2, Poincare's constant is at most
    1/(pi*sqrt(2)).  Equal-area triangles yield ||div(q)+f|| <= rho*fmax.
    A small outward numerical guard is applied; this is not interval arithmetic.
    """
    amin=np.exp(np.min(x[:,:4],axis=1))
    fmax=1+np.exp(x[:,4])+np.exp(x[:,5])
    correction=results[:,-1]*fmax/(np.pi*np.sqrt(2*amin))
    upper=(np.sqrt(results[:,1])+correction)**2
    guard=1e-11
    return np.log(results[:,0]*(1-guard)),np.log(upper*(1+guard))

def score_bounds(lower,upper,pred,scale=None):
    scale=1. if scale is None else scale
    return np.maximum.reduce([lower-pred,pred-upper,np.zeros_like(lower)])/scale,np.maximum(np.abs(lower-pred),np.abs(upper-pred))/scale

def quantile_rank(n,alpha):return int(np.ceil((n+1)*(1-alpha)))-1

def bracket(a,b,alpha):
    k=quantile_rank(len(a),alpha)
    if k>=len(a):return np.inf,np.inf
    return float(np.partition(a,k)[k]),float(np.partition(b,k)[k])

def refine_query(n,nlev,pred,cost,query,alpha=.1,tol=.005,policy='rank',seed=0,trace=False,scale=None):
    """Calibrate by requesting query(index, level) -> (log lower, log upper).

    The predictor and optional positive scale must be fixed independently of
    calibration. Cost contains the known work of every potential request.
    """
    if not 0<alpha<1 or tol<0:raise ValueError('Invalid coverage or tolerance')
    if policy not in ['rank','width','random','uniform']:raise ValueError(policy)
    if scale is not None and np.any(np.asarray(scale)<=0):raise ValueError('Scale must be positive')
    levels=np.zeros(n,dtype=int)
    initial=np.array([query(i,0) for i in range(n)])
    lo=initial[:,0];hi=initial[:,1];spent=float(cost[:,0].sum());calls=n
    if np.any(lo>hi):raise RuntimeError('Invalid initial enclosures')
    rng=np.random.default_rng(seed);history=[]
    if quantile_rank(n,alpha)>=n:
        return dict(q_lower=np.inf,q_upper=np.inf,gap=np.nan,work=spent,calls=calls,levels=levels,converged=False,trace=np.array([[spent,np.inf,np.inf,calls]]))
    while True:
        a,b=score_bounds(lo,hi,pred,scale);ql,qh=bracket(a,b,alpha)
        if trace:history.append([spent,ql,qh,calls])
        if qh-ql<=tol or np.all(levels==nlev-1):break
        can=levels<nlev-1
        if policy=='uniform':chosen=np.flatnonzero(can)
        else:
            if policy=='rank':
                can &= (a<=qh)&(b>=ql)
                if not np.any(can):break
            candidates=np.flatnonzero(can)
            if policy=='random':chosen=np.array([rng.choice(candidates)])
            else:
                utility=(b[candidates]-a[candidates])/cost[candidates,levels[candidates]+1]
                chosen=np.array([candidates[np.argmax(utility)]])
        for i in chosen:
            levels[i]+=1;lnew,unew=query(int(i),int(levels[i]))
            lo[i]=max(lo[i],lnew);hi[i]=min(hi[i],unew)
        if np.any(lo>hi):raise RuntimeError('Inconsistent deterministic label enclosures')
        spent+=float(cost[chosen,levels[chosen]].sum());calls+=len(chosen)
    return dict(q_lower=ql,q_upper=qh,gap=qh-ql,work=spent,calls=calls,levels=levels,converged=bool(qh-ql<=tol),trace=np.asarray(history))

def refine(lower,upper,pred,cost,alpha=.1,tol=.005,policy='rank',seed=0,trace=False,scale=None):
    """Replay the same request interface using a stored fidelity database."""
    n,nlev=lower.shape
    return refine_query(n,nlev,pred,cost,lambda i,j:(lower[i,j],upper[i,j]),alpha,tol,policy,seed,trace,scale)

def coverage_bracket(pred,q,lower,upper):
    certain=(lower>=pred-q)&(upper<=pred+q)
    possible=(upper>=pred-q)&(lower<=pred+q)
    midpoint=np.abs((lower+upper)/2-pred)<=q
    return float(certain.mean()),float(possible.mean()),float(midpoint.mean())
