import clusterfock as cf

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def run_with_pertubation(sign, geometry, basis, CC, expval, epsilon, component, run_args):
    b = cf.PyscfBasis(geometry, basis).pyscf_hartree_fock()
    b.from_restricted()

    if expval == "r":
        b.h += sign*epsilon * b.r[component, ...]
    if expval == "Q":
        i, j = component
        b.h += sign*epsilon * b.Q[i, j, ...]

    cc = CC(b)
    cc.run(**run_args)

    return cc.energy()

def two_point_energies(geometry, basis, CC, expval, epsilon, component, **kwargs):
    tol = kwargs.pop("tol", 1e-8)

    run_args = {"tol": tol}
    if not CC.__name__.startswith("Q"):
        run_args["include_l"] = True    
    
    assert expval in ["r", "Q"]
    if expval == "r":
        assert type(component) == int  and 0 <= component <= 2
    if expval == "Q":
        assert type(component) in [list, tuple] and 0 <= component[0] <= 2 and 0 <= component[1] <= 2

    e_plus = run_with_pertubation(+1.0, geometry, basis, CC, expval, epsilon, component, run_args)
    e_minus = run_with_pertubation(-1.0, geometry, basis, CC, expval, epsilon, component, run_args)

    b = cf.PyscfBasis(geometry, basis).pyscf_hartree_fock()
    b.from_restricted()
    cc = CC(b).run(**run_args)


    return e_plus, e_minus

def get_expected(geometry, basis, CC, expval, component, **kwargs):
    tol = kwargs.pop("tol", 1e-8)

    run_args = {"tol": tol}
    if not CC.__name__.startswith("Q"):
        run_args["include_l"] = True    
    
    assert expval in ["r", "Q"]
    if expval == "r":
        assert type(component) == int  and 0 <= component <= 2
    if expval == "Q":
        assert type(component) in [list, tuple] and 0 <= component[0] <= 2 and 0 <= component[1] <= 2

    b = cf.PyscfBasis(geometry, basis).pyscf_hartree_fock()
    b.from_restricted()
    cc = CC(b).run(**run_args)

    expected = 0
    if expval == "r":
        expected =  cc.one_body_expval(b.r[component, ...])
    if expval == "Q":
        i, j = component
        expected =  cc.one_body_expval(b.Q[i, j, ...])

    return expected

def main():
    geometry = "Li 0 0 0; H 0 0 2.0;"
    basis = "sto-3g"
    expval = "Q"
    component = (1,1)
    CC = cf.QCCSD

    epsilon = np.logspace(-2, -5, 6)
    # epsilon = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    estimates= np.zeros_like(epsilon)

    for i, eps in enumerate(epsilon):
        e_plus, e_minus = two_point_energies(geometry, basis, CC, expval, eps, component)    
        estimates[i] = (e_plus - e_minus)/(2*eps)
        print(f"Done {eps = }, estimate = {estimates[i]}")

    expected = get_expected(geometry, basis, CC, expval, component)

    fig, ax = plt.subplots()
    error = np.abs(estimates-expected)
    ax.scatter(epsilon, error)
    
    log_epsilon, log_error = np.log10(epsilon), np.log10(error)

    # coeffs = np.polyfit(epsilon,log_error,deg=1)
    # print(coeffs)
    # poly = np.poly1d(coeffs)
    func = lambda x, a, b: a*x + b
    popt, pcov = curve_fit(func, log_epsilon, log_error)
    a, b = popt

    print(a)
    error_fit = np.power(10, func(log_epsilon, a, b))
    
    ax.plot(epsilon, error_fit, ls="--")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.invert_xaxis()
    plt.show()

if __name__ == "__main__":
    main()