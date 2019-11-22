from pymc3 import Model, Normal


def simple_model():
  mu = 2.
  tau = 10.
  with Model() as model:
    x = Normal('x', mu, tau=tau)

  return model


if __name__ == '__main__':
  model = simple_model()
  print(model.x.logp({ 'x' : 2.01 }))
