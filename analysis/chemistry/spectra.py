import clusterfock as cf
import matplotlib.pyplot as plt
import numpy as np

dt = 0.01
F_str = 0.02
time = (0,10,dt)

def delta_kick(t, basis, dt=dt, F_str=F_str, u=np.array([0,0,1])):
    if t < dt:
        return F_str*np.einsum("xij,x->ij", basis.r, u)
    else:
        return 0
    
def sampler(basis):
    return {"r": basis.r}
basis = cf.PyscfBasis("H 0 0 -0.5; H 0 0 0.5", "sto-3g")
hf = cf.HF(basis).run()
basis.change_basis(hf.C)
basis.from_restricted()

cc = cf.CCD(basis)
tdcc = cf.td.TimeDependentCoupledCluster(cc, time=time)
tdcc.external_one_body = delta_kick
tdcc.run(vocal=True)