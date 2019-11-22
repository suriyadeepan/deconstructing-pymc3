from model import Model
from continuous import Normal


def simple_model():
  mu = 2.
  sigma = 10.
  with Model() as model:
    x = Normal('x', mu, sigma=sigma)

  return model


if __name__ == '__main__':
  model = simple_model()
  print('logp', model.x.logp_elemwiset)
