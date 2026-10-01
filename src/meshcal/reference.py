"""Independent higher-order, graded-mesh primal--dual reference enclosures."""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from skfem import Basis,ElementTriP2,ElementTriP1DG,ElementTriRT2,asm
from .fem import spaces,fields,stiffness,forcing,flux_mass,divergence

@lru_cache(maxsize=3)
def reference_spaces(case,n=32,grading=12):
    mesh=spaces(case,n)[0]
    points=[(.5,.5),(.5,0),(.5,1),(0,.5),(1,.5)]
    if case=='hole':points +=[(.375,.375),(.375,.625),(.625,.375),(.625,.625),(.5,.375),(.5,.625),(.375,.5),(.625,.5)]
    for j in range(grading):
        c=mesh.p[:,mesh.t].mean(axis=1)
        dist=np.min(np.stack([np.sum((c-np.array(p)[:,None])**2,axis=0) for p in points]),axis=0)**.5
        selected=np.flatnonzero(dist<.10*.5**j)
        if len(selected):mesh=mesh.refined(selected)
    p=Basis(mesh,ElementTriP2(),intorder=4);rt=Basis(mesh,ElementTriRT2(),intorder=4);z=Basis(mesh,ElementTriP1DG(),intorder=4)
    I=np.setdiff1d(np.arange(p.N),p.get_dofs().all());B=asm(divergence,rt,z).tocsc()
    return mesh,p,rt,z,I,B

def reference_solve(case,theta,n=32,grading=12):
    t=perf_counter();mesh,p,rt,z,I,B=reference_spaces(case,n,grading)
    a,f=fields(theta,p.global_coordinates());K=asm(stiffness,p,a=a);F=asm(forcing,p,f=f);u=np.zeros(p.N);u[I]=spsolve(K[I][:,I].tocsc(),F[I]);lower=float(2*F@u-u@(K@u))
    ar,fr=fields(theta,rt.global_coordinates());M=asm(flux_mass,rt,a=ar);az,fz=fields(theta,z.global_coordinates());g=-asm(forcing,z,f=fz)
    S=bmat([[M,B.T],[B,None]],format='csc');v=spsolve(S,np.r_[np.zeros(rt.N),g]);q=v[:rt.N];eq=float(q@(M@q))
    residual_l2=float(np.sqrt(np.sum((fr+rt.interpolate(q).div)**2*rt.dx)))
    amin=float(ar.min());upper=(np.sqrt(eq)+residual_l2/(np.pi*np.sqrt(2*amin)))**2
    return dict(lower=lower*(1-1e-11),upper=upper*(1+1e-11),seconds=perf_counter()-t,triangles=mesh.nelements,dofs=p.N+rt.N+z.N,residual_l2=residual_l2)
