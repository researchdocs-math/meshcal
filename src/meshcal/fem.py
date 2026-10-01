"""Conforming P1 / equilibrated RT0 bounds for elliptic compliance."""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri, Basis, BilinearForm, LinearForm, ElementTriP1, ElementTriP0, ElementTriRT0, asm
from skfem.helpers import dot, grad, div

@BilinearForm
def stiffness(u,v,w): return w.a*dot(grad(u),grad(v))
@LinearForm
def forcing(v,w): return w.f*v
@BilinearForm
def flux_mass(u,v,w): return dot(u,v)/w.a
@BilinearForm
def divergence(u,v,w): return div(u)*v

@lru_cache(maxsize=24)
def spaces(case, n):
    mesh=MeshTri.init_tensor(np.linspace(0,1,n+1),np.linspace(0,1,n+1))
    c=mesh.p[:,mesh.t].mean(axis=1)
    if case=='notch':
        mesh=mesh.remove_elements(np.where((c[0]>.5)&(c[1]>.5))[0])
    elif case=='hole':
        mesh=mesh.remove_elements(np.where((c[0]>.375)&(c[0]<.625)&(c[1]>.375)&(c[1]<.625))[0])
    p=Basis(mesh,ElementTriP1(),intorder=2)
    rt=Basis(mesh,ElementTriRT0(),intorder=2)
    z=Basis(mesh,ElementTriP0(),intorder=2)
    interior=np.setdiff1d(np.arange(p.N),p.get_dofs().all())
    B=asm(divergence,rt,z).tocsc()
    return mesh,p,rt,z,interior,B

def fields(theta, points):
    x,y=points[0],points[1]
    region=(x>=.5).astype(int)+2*(y>=.5).astype(int)
    a=np.exp(np.asarray(theta[:4]))[region]
    f=1.+np.exp(theta[4])*(x<.25)+np.exp(theta[5])*(y>.75)
    return a,f

def solve(case, n, theta, solution=False):
    t0=perf_counter()
    mesh,p,rt,z,inside,B=spaces(case,int(n))
    a,f=fields(theta,p.global_coordinates())
    K=asm(stiffness,p,a=a)
    F=asm(forcing,p,f=f)
    u=np.zeros(p.N)
    u[inside]=spsolve(K[inside][:,inside].tocsc(),F[inside])
    lower=float(2*F@u-u@(K@u))
    ar,fr=fields(theta,rt.global_coordinates())
    M=asm(flux_mass,rt,a=ar)
    az,fz=fields(theta,z.global_coordinates())
    g=-asm(forcing,z,f=fz)
    S=bmat([[M,B.T],[B,None]],format='csc')
    rhs=np.r_[np.zeros(rt.N),g]
    mixed=spsolve(S,rhs)
    q=mixed[:rt.N]
    upper=float(q@(M@q))
    residual=float(np.max(np.abs(B@q-g))/max(np.max(np.abs(g)),1e-15))
    out=dict(lower=lower,upper=upper,seconds=perf_counter()-t0,primal_dofs=p.N,flux_dofs=rt.N,triangles=mesh.nelements,residual=residual)
    if solution: out.update(points=mesh.p,triangles_array=mesh.t,u=u)
    if not (0<lower<=upper*(1+1e-10)) or residual>1e-8:
        raise RuntimeError(out)
    return out

def manufactured(n):
    """u=sin(pi*x)sin(pi*y), integrate forcing at high order: convergence test."""
    mesh=MeshTri.init_tensor(np.linspace(0,1,n+1),np.linspace(0,1,n+1))
    p=Basis(mesh,ElementTriP1(),intorder=8)
    x=p.global_coordinates()
    f=2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
    K=asm(stiffness,p,a=np.ones_like(f));F=asm(forcing,p,f=f)
    I=np.setdiff1d(np.arange(p.N),p.get_dofs().all())
    u=np.zeros(p.N);u[I]=spsolve(K[I][:,I],F[I])
    return float(F@u),np.pi**2/2
