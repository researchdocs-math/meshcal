"""Numerical and statistical invariants of mesh-aware calibration."""
import unittest
import numpy as np
from meshcal.calibration import score_bounds,bracket,refine,refine_query,log_enclosures
from meshcal.fem import solve,manufactured
from meshcal.reference import reference_solve

class CalibrationTests(unittest.TestCase):
    def test_adaptive_domination_and_precision(self):
        rng=np.random.default_rng(829)
        for repeat in range(20):
            y=rng.normal(size=99);p=.7*y+rng.normal(size=99)*.15
            widths=rng.uniform(.1,.4,(99,1))*np.array([1,.2,.04,.008,0])[None,:]
            l=y[:,None]-widths;u=y[:,None]+widths;cost=np.tile([1,4,16,64,256],(99,1));qstar=np.partition(abs(y-p),89)[89]
            for policy in ['rank','width','random','uniform']:
                r=refine(l,u,p,cost,tol=.005,policy=policy,trace=True)
                self.assertLessEqual(r['q_lower'],qstar+1e-13);self.assertGreaterEqual(r['q_upper'],qstar-1e-13)
                self.assertTrue(r['converged']);self.assertLessEqual(r['q_upper']-qstar,.005+1e-13)
                self.assertTrue(np.all(np.diff(r['trace'][:,1])>=-1e-13));self.assertTrue(np.all(np.diff(r['trace'][:,2])<=1e-13))
    def test_ties_and_cap(self):
        l=np.zeros((19,2));u=np.ones((19,2));p=np.zeros(19);c=np.ones((19,2))
        r=refine(l,u,p,c,tol=0);self.assertFalse(r['converged']);self.assertEqual(r['q_upper'],1)
        r=refine(l,l,p,c,tol=0);self.assertTrue(r['converged']);self.assertEqual(r['calls'],19)
    def test_unbounded_small_sample(self):
        r=refine(np.zeros((3,2)),np.ones((3,2)),np.zeros(3),np.ones((3,2)),alpha=.01)
        self.assertTrue(np.isinf(r['q_upper']))
    def test_inconsistent_bounds_rejected(self):
        with self.assertRaises(RuntimeError):refine(np.ones((19,2)),np.zeros((19,2)),np.zeros(19),np.ones((19,2)))
    def test_nonunit_scale(self):
        a,b=score_bounds(np.array([1.,-1.]),np.array([2.,1.]),np.zeros(2),np.array([2.,.5]));np.testing.assert_allclose(a,[.5,0]);np.testing.assert_allclose(b,[1,2])

class FiniteElementTests(unittest.TestCase):
    def test_analytic_compliance_enclosure(self):
        # Positive Fourier series with a conservative omitted-energy tail.
        odd=np.arange(1,2048,2,dtype=float);m=odd[:,None];n=odd[None,:]
        exact=float(np.sum(64/(np.pi**6*m*m*n*n*(m*m+n*n))))
        tail=1/(np.pi**2*(2049**2+1));theta=[0,0,0,0,-np.inf,-np.inf]
        for level in [8,16,32]:
            r=solve('square',level,theta);self.assertLess(r['lower'],exact);self.assertGreater(r['upper'],exact+tail)
        r=reference_solve('square',theta,grading=4)
        self.assertLess(r['lower'],exact);self.assertGreater(r['upper'],exact+tail)
    def test_manufactured_second_order_compliance(self):
        errors=[abs(manufactured(n)[0]-manufactured(n)[1]) for n in [8,16,32]]
        self.assertGreater(errors[0]/errors[1],3.5);self.assertGreater(errors[1]/errors[2],3.5)
    def test_nested_energy_bounds(self):
        x=np.array([.8,-.5,.3,-.9,.4,.1])
        for case in ['square','notch','hole']:
            r=[solve(case,n,x) for n in [8,16,32]]
            self.assertTrue(all(a['lower']<b['lower'] for a,b in zip(r,r[1:])))
            self.assertTrue(all(a['upper']>b['upper'] for a,b in zip(r,r[1:])))
            self.assertLess(max(v['residual'] for v in r),1e-9)
if __name__=='__main__':unittest.main()
