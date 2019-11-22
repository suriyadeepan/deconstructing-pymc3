import threading
import numpy as np

from theano.tensor.var import TensorVariable


class Context:
  contexts = threading.local()  # thread-local storage

  def __enter__(self):
    cls = type(self)  # get a handle to Class `Context`
    contexts = cls.get_contexts()  # call @classmethod `get_contexts`
    contexts.append(self)  # add instance to contexts
    return self

  def __exit__(self, typ, value, traceback):
    cls = type(self)  # get a handle to Class `Context`
    contexts = cls.get_contexts()  # call @classmethod `get_contexts`
    contexts.pop()  # remove instance from contexts stack

  @classmethod
  def get_contexts(cls):
    if not hasattr(cls.contexts, 'stack'):  # does `Context.contexts.stack` exist?
      cls.contexts.stack = []  # create and return an empty stack
    return cls.contexts.stack

  @classmethod
  def get_context(cls):
    contexts = cls.get_contexts()  # get all contexts
    if len(contexts) == 0:
      raise Exception("Context stack is empty!")
    return contexts[-1]  # return the deepest context


class InitContextMeta(type):
  """Metaclass that runs Model.__init__ in its context"""
  def __call__(cls, *args, **kwargs):
    instance = cls.__new__(cls, *args, **kwargs)  # create an instance of `Model`
    with instance:  # run __init__ in context
      instance.__init__(*args, **kwargs)  # `Model.__init__`
    return instance


class Model(Context, metaclass=InitContextMeta):

  def __new__(cls, *args, **kwargs):  # class method that creates an instance
    instance = super().__new__(cls)
    # resolve parent instance
    if cls.get_contexts():  # if contexts stack isn't empty
      instance._parent = cls.get_context()  # get the deepest context
    else:
      instance._parent = None
    return instance

  def __init__(self, name=''):
    self.name = name
    if self.parent is None:
      self.named_vars = {}  # using python dict instead of pymc's treedict
      self.free_RVs = []    # using python list instead of pymc's treelist
      self.observed_RVs = []
      self.deterministics = []
      self.potentials = []
      self.missing_values = []
    else:
      raise NotImplementedError('We dont care about this case yet!')

  @property
  def parent(self):
    return self._parent

  def Var(self, name, dist, data=None):
    """Create and add (un)observed variable to the model"""
    if data is None:  # Unobserved variable
      with self:  # create a free RV in context
        var = FreeRV(name=name, distribution=dist, model=self)
      # add to the list of free RVs
      self.free_RVs.append(var)
    else:  # Observed variable
      with self:
        var = ObservedRV(name=name, distribution=dist, data=data, model=self)
      # add to the list of observed RVs
      self.observed_RVs.append(var)
    # add to model.named_vars
    self.add_random_variable(var)

  def add_random_variable(self, var):
    # check if variable already exists in named_vars
    if var.name in self.named_vars:
      raise Exception("Variable {} already exists in model!".format(var.name))
    self.named_vars[var.name] = var
    # add variable as attribute of model
    if not hasattr(self, var.name):
      setattr(self, var.name, var)


class Factor:

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


def incorporate_methods(src, dest, methods, override=False):
  for method in methods:
    if hasattr(dest, method) and not override:
      raise Exception("Method {} already exists in {}".format(method, dest))

    if hasattr(src, method):
      setattr(dest, method, getattr(src, method))
    else:
      setattr(dest, method, None)


class FreeRV(Factor, TensorVariable):
  """Unobserved Random Variable"""
  def __init__(self, name=None, distribution=None, model=None,
      type=None, owner=None, index=None):

    if type is None:
      type = distribution.type
    super().__init__(type, owner, index, name)

    self.name = name
    self.dshape = tuple(distribution.shape)  # dims
    self.dsize = int(np.prod(distribution.shape))  # product of dims
    self.distribution = distribution
    self.logp_elemwiset = distribution.logp(self)
    self.logp_sum_unscaledt = distribution.logp_sum(self)
    self.model = model
    incorporate_methods(src=distribution, dest=self, methods=['random'])
