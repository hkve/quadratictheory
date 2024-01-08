import clusterfock as cf
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

dt = 0.01
F_str = 0.2
time = (0,0.5,dt) # 225

def delta_kick(t, basis, dt=dt, F_str=F_str, u=np.array([0,0,1])):
    if 0 < t < dt:
        return F_str*np.einsum("xij,x->ij", basis.r, u)
    else:
        return 0
    
def pulse_li_et_al(t, basis, omega=0.1, Em=0.07):
    ot = omega*t
    
    def f(t):
        if ot < 2*np.pi:
            return ot/(2*np.pi)*Em
        elif 2*np.pi < ot < 4*np.pi:
            return Em
        elif 4*np.pi < ot < 6*np.pi:
            return (3-ot/(2*np.pi))*Em
        else:
            return 0

    return -basis.r[2]*f(t)

def sampler(basis):
    return {"r": basis.r[2]}

def run_helium_test():
    Re = 0.7354
    basis = cf.PyscfBasis(f"H 0 0 {-Re/2}; H 0 0 {Re/2}", "6-311++Gss")
    hf = cf.HF(basis).run()
    basis.change_basis(hf.C)
    basis.from_restricted()

    cc = cf.CCSD(basis)
    tdcc = cf.td.TimeDependentCoupledCluster(cc, time=time)
    tdcc.external_one_body = pulse_li_et_al #delta_kick
    tdcc.one_body_sampler = sampler
    tdcc.run(vocal=True)

    return tdcc.results

def plot(results):

    # fig, ax = plt.subplots()
    # ax.set(ylim=(0,1.1))
    # ax.plot(results["t"], results["overlap"].real)
    # print("Shows overlap")
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(results["t"], results["energy"].real)
    # print("Shows energy")
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(results["t"][1:], -results["r"].real)
    print("Shows pos")
    plt.show()

    embed()


def main():
    results = run_helium_test()
    plot(results)

if __name__ == "__main__":
    main()
