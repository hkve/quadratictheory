from clusterfock.basis import Basis
import numpy as np


class Sampler:
    def __init__(self, one_body=False, two_body=False, misc=False):
        self.has_one_body = one_body
        self.has_two_body = two_body
        self.has_misc = misc

        self.has_overlap = False

    def setup_sampler(self, basis, tdcc):
        one_body_keys = list(self.one_body(basis).keys())
        two_body_keys = list(self.two_body(basis).keys())
        misc_keys = list(self.misc(tdcc).keys())

        return one_body_keys + two_body_keys + misc_keys

    def one_body(self, basis: Basis) -> dict:
        return {}

    def two_body(self, basis: Basis) -> dict:
        return {}

    def misc(self, tdcc) -> dict:
        return {}

class ImagTimeSampler(Sampler):
    def __init__(self, one_body=False, two_body=False, misc=True):
        super().__init__(one_body, two_body, misc)

    def misc(self, tdcc) -> dict:
        cc, cc_gs = tdcc.cc, tdcc.cc_gs
        
        t_norms = {}
        l_norms = {}
        for order in cc._t.orders:
            t_norms[f"delta_t{order}"] = np.linalg.norm(cc_gs._t[order] - cc._t[order])
            l_norms[f"delta_l{order}"] = np.linalg.norm(cc_gs._l[order] - cc._l[order])

        result = {
            "energy": cc.time_dependent_energy(),
        }

        result.update(t_norms)
        result.update(l_norms)

        return result


class DipoleSampler(Sampler):
    def __init__(self, one_body=True, two_body=False, misc=True):
        super().__init__(one_body, two_body, misc)

    def one_body(self, basis: Basis) -> dict:
        return {"r": basis.r}

    def misc(self, tdcc) -> dict:
        return {
            "energy": tdcc.cc.time_dependent_energy(),
            "delta_rho1": np.linalg.norm(tdcc.cc.rho_ob - tdcc.cc.rho_ob.conj().T),
        }


class DipoleSamplerExpanded(Sampler):
    def __init__(self, one_body=True, two_body=True, misc=True):
        super().__init__(one_body, two_body, misc)

    def one_body(self, basis: Basis) -> dict:
        return {"r": basis.r, "h": basis.h}

    def two_body(self, basis: Basis) -> dict:
        return {"u": basis.u}

    def misc(self, tdcc) -> dict:
        return {
            "energy": tdcc.cc.time_dependent_energy(),
            "delta_rho1": np.linalg.norm(tdcc.cc.rho_ob - tdcc.cc.rho_ob.conj().T),
            "delta_rho2": np.linalg.norm(
                tdcc.cc.rho_tb - tdcc.cc.rho_tb.conj().transpose(2, 3, 0, 1)
            ),
        }


class OverlapSampler(Sampler):
    def __init__(self, one_body=False, two_body=False, misc=True):
        super().__init__(one_body, two_body, misc)
        self.has_overlap = True

    def misc(self, tdcc) -> dict:
        return {
            "energy": tdcc.cc.time_dependent_energy(),
            "overlap": tdcc.cc.overlap(tdcc._t0, tdcc._l0, tdcc.cc._t, tdcc.cc._l),
        }
