import threading


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
  
  
if __name__ == '__main__':
  with Model() as model:
    pass
