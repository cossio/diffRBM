# DiffRBM Python package

Python module to train a Differential RBM (DiffRBM), as described in the paper

> Learning the differences: a machine learning approach to predicting antigen immunogenicity and T-cell receptor specificity

Usage of this repo or modifications derived from it should cite the above publication.

This code is based on a fork of https://github.com/jertubiana/PGM, which implements training of a Restricted Boltzmann machine.

On top of that, we implemented here the class `DiffRBM` (in the file `source/diffrbm.py`). The following code gives an example of how to construct it:

```
RBM_back # pre-trained background RBM
# ....

# construct the full RBM (back + diff units)
RBM_post = rbm.RBM(
  visible = RBM_back.visible, # nature of visible units
  hidden = RBM_back.hidden, # nature of hidden units
  n_v = RBM_back.n_v, # number of visible units
  n_cv = RBM_back.n_cv, # number of states
  n_h = RBM_back.n_h + diff_n_h # hidden units = background hidden units + diffRBM units
)

# construct a `DiffRBM` object
dRBM = diffrbm.DiffRBM(RBM_back, RBM_post)
# ensure parameters of the post and back models are in-sync
dRBM.update_post_from_back(vlayer=True, hlayer=True)
```

At this point `dRBM` has been initialized. The background RBM parameters have been copied from the pre-trained `RBM_back`. Now we need to train the diffRBM units on "selected data". To do this, use the `fit_top` function,

```
dRBM.fit_top(sel_data)
```

See the example notebooks in https://github.com/bravib/diffRBM_immunogenicity_TCRspecificity for more details.

# Requirements

Python (tested with v3.9), Numpy (tested with v1.23), Numba (tested with v0.56). See also the Requirements section in https://github.com/jertubiana/PGM.