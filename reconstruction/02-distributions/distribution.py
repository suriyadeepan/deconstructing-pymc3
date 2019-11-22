from model import Model
import numpy as np
import theano
import theano.tensor as tt


class Distribution:

  def __new__(cls, name, *args, **kwargs):
    try:  # a distribution must be created only inside a model's context
      model = Model.get_context()
    except TypeError:
      raise Exception("Distributions must be created inside the model's context!")
    
    # check if we have any data associated with this variable
    data = kwargs.pop('observed', None)
    cls.data = data  # attach to class `Normal` (cls)
    dist = cls.dist(*args, **kwargs)  # create an instance of `Normal`
    return model.Var(name, dist, data)

  @classmethod
  def dist(cls, *args, **kwargs):
    # create an object of type `Normal` in a cool way
    dist = object.__new__(cls)
    dist.__init__(*args, **kwargs)  # call `Normal.__init__`
    return dist

  def __init__(self, shape, dtype, broadcastable=None):
    self.shape = np.atleast_1d(shape)  # array([]) <- ()
    self.dtype = dtype
    self.type = TensorType(self.dtype, self.shape, broadcastable)

  def logp_sum(self, *args, **kwargs):
    return tt.sum(self.logp(*args, **kwargs))


def TensorType(dtype, shape, broadcastable=None):
  if broadcastable is None:
    broadcastable = np.atleast_1d(shape) == 1
  return tt.TensorType(str(dtype), broadcastable)


class Continuous(Distribution):

  def __init__(self, shape=(), dtype=None, *args, **kwargs):
    dtype = theano.config.floatX
    super().__init__(shape, dtype, *args, **kwargs)
