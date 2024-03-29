# Deconstructing PyMC3

This is my attempt to understand the internals of PyMC3
by deconstructing one module at a time and reconstructing them incrementally.

## Setup

Debugging setup

```bash
pip install -r requirements-dev.txt
git clone https://github.com/pymc-devs/pymc3
cp tests/*.py pymc3/ && cd pymc3/
ipdb {test_x}.py  # run debugger
```

Reconstruction setup

```bash
pip install requirements.txt
python reconstruction/{x}/{y}.py
```

## TODO

- [x] pm.Model
  - [Deconstructing PyMC3 : Part I](http://antithesis.pub/pymc3/deconstruction/2019/11/06/deconstructing-pymc3-part-i.html)
  - [reconstruction/01-model/](reconstruction/01-model/)
- [x] pm.Normal
  - [Deconstructing PyMC3 Part 2 : Random Variables and Prior Distributions](http://antithesis.pub/pymc3/deconstruction/2019/11/22/deconstructing-pymc3-part-2.html)
  - [reconstruction/02-distributions/](reconstruction/02-distributions/)
