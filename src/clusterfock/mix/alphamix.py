from clusterfock.mix import Mixer


class AlphaMixer(Mixer):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, old, new):
        return (1 - self.alpha) * new + self.alpha * old
