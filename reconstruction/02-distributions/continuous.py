from distribution import Continuous
import numpy as np
import theano.tensor as tt


class Normal(Continuous):

  def __init__(self, mu=0., sigma=1., **kwargs):
    self.mu = mu
    self.sigma = sigma
    self.tau = self.sigma ** -2
    self.variance = 1. / self.tau

    super().__init__(**kwargs)

  def logp(self, value):
    """Calculate log-probability of Normal distribution at specified value."""
    sigma = self.sigma
    tau = self.tau
    mu = self.mu

    if sigma > 0:
      return (-tau * (value - mu)**2 + tt.log(tau / np.pi / 2.)) / 2.

    return -np.inf
