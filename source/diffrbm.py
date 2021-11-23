import numpy as np
import layer
import pgm
import moi
import utilities
from utilities import check_random_state, gen_even_slices, logsumexp, log_logistic, average, average_product, saturate, check_nan, get_permutation, rand_shuffle_data
import time
import copy
import rbm

from float_precision import double_precision, curr_float, curr_int


# %%

# used internally to break away from a callback
class BreakTrain(Exception):
    pass


class DiffRBM:
    def __init__(self, RBMback, RBMpost, update_back=False, update_back_vlayer=False, update_back_hlayer=False):
        self.RBMback = RBMback
        self.RBMpost = RBMpost
        assert RBMback.visible == RBMpost.visible
        assert RBMback.hidden == RBMpost.hidden
        assert RBMback.n_cv == RBMpost.n_cv
        assert RBMback.n_ch == RBMpost.n_ch
        assert RBMback.n_v <= RBMpost.n_v
        assert RBMback.n_h <= RBMpost.n_h
        assert RBMback.n_v < RBMpost.n_v or RBMback.n_h < RBMpost.n_h
        self.n_v_ = RBMback.n_v
        self.n_h_ = RBMback.n_h
        if update_back:
            self.update_back_from_post(vlayer=update_back_vlayer, hlayer=update_back_hlayer)
    
    # returns true if back and post RBMs parameters are in sync
    def insync(self, vlayer=False, hlayer=True):
        weights_insync = (self.RBMback.weights[:] == self.RBMpost.weights[:self.n_h_, :self.n_v_]).all()
        vlayer_insync = hlayer_insync = True
        if vlayer:
            for key in self.RBMback.vlayer.list_params:
                c = (self.RBMback.vlayer.__dict__[key][:] == self.RBMpost.vlayer.__dict__[key][:self.n_v_]).all()
                vlayer_insync = c and vlayer_insync
        if hlayer:
            for key in self.RBMback.hlayer.list_params:
                c = (self.RBMback.hlayer.__dict__[key][:] == self.RBMpost.hlayer.__dict__[key][:self.n_h_]).all()
                hlayer_insync = c and hlayer_insync
        return weights_insync and vlayer_insync and hlayer_insync

    # updates back RBM from post parameters
    def update_back_from_post(self, vlayer=False, hlayer=True):
        self.RBMback.weights[:] = self.RBMpost.weights[:self.n_h_, :self.n_v_]
        if vlayer:
            for key in self.RBMback.vlayer.list_params:
                self.RBMback.vlayer.__dict__[key][:] = self.RBMpost.vlayer.__dict__[key][:self.n_v_]
        if hlayer:
            for key in self.RBMback.hlayer.list_params:
                self.RBMback.hlayer.__dict__[key][:] = self.RBMpost.hlayer.__dict__[key][:self.n_h_]
            self.RBMback.hlayer.recompute_params(which='')
    
    # updates post RBM from back parameters
    def update_post_from_back(self, vlayer=False, hlayer=True):
        self.RBMpost.weights[:self.n_h_, :self.n_v_] = self.RBMback.weights
        if vlayer:
            for key in self.RBMpost.vlayer.list_params:
                self.RBMpost.vlayer.__dict__[key][:self.n_v_] = self.RBMback.vlayer.__dict__[key]
        if hlayer:
            for key in self.RBMpost.hlayer.list_params:
                self.RBMpost.hlayer.__dict__[key][:self.n_h_] = self.RBMback.hlayer.__dict__[key]
            self.RBMpost.hlayer.recompute_params(which='')

    # updates post RBM from top parameters in 'topRBM'
    def update_post_from_top(self, topRBM, vlayer=False, hlayer=True):
        self.RBMpost.weights[self.n_h_:] = topRBM.weights
        if vlayer:
            for key in self.RBMpost.vlayer.list_params:
                self.RBMpost.vlayer.__dict__[key][:self.n_v_] = topRBM.vlayer.__dict__[key][:self.n_v_] + self.RBMback.vlayer.__dict__[key]
                self.RBMpost.vlayer.__dict__[key][self.n_v_:] = topRBM.vlayer.__dict__[key][self.n_v_:]
        if hlayer:
            for key in self.RBMpost.hlayer.list_params:
                self.RBMpost.hlayer.__dict__[key][self.n_h_:] = topRBM.hlayer.__dict__[key]
            self.RBMpost.hlayer.recompute_params(which='')
    
    # updates topRBM with parmeters from 'top' part of the post-RBM
    def update_top_from_post(self, topRBM, vlayer=True, hlayer=True):
        topRBM.weights[:] = self.RBMpost.weights[self.n_h_:]
        if vlayer:
            for key in self.RBMpost.vlayer.list_params:
                topRBM.vlayer.__dict__[key][:self.n_v_] = self.RBMpost.vlayer.__dict__[key][:self.n_v_] - self.RBMback.vlayer.__dict__[key]
                topRBM.vlayer.__dict__[key][self.n_v_:] = self.RBMpost.vlayer.__dict__[key][self.n_v_:]
        if hlayer:
            for key in self.RBMpost.hlayer.list_params:
                topRBM.hlayer.__dict__[key][:] = self.RBMpost.hlayer.__dict__[key][self.n_h_:]
            topRBM.hlayer.recompute_params(which='')

    # returns the top RBM
    def top_rbm(self):
        RBMtop = rbm.RBM(n_v=self.RBMpost.n_v, n_h=self.RBMpost.n_h - self.n_h_,
                         n_cv=self.RBMpost.n_cv, n_ch=self.RBMpost.n_ch,
                         visible=self.RBMpost.visible, hidden=self.RBMpost.hidden)
        self.update_top_from_post(RBMtop, vlayer=True, hlayer=True)
        return RBMtop
    
    # alpha_post * Lpost + alpha_back * Lback
    def likelihood(self, data_post, data_back, omega_post=1, omega_back=1, recompute_Z=False):
        Lback = self.RBMback.likelihood(data_back, recompute_Z = recompute_Z)
        Lpost = self.RBMpost.likelihood(data_post, recompute_Z = recompute_Z)
        alpha_back = data_back.shape[0] / (data_back.shape[0] + data_post.shape[0]) * omega_post
        alpha_post = data_post.shape[0] / (data_back.shape[0] + data_post.shape[0]) * omega_back
        return alpha_post * Lpost.mean() + alpha_back * Lback.mean()
    
    # alpha_post * Lpost + alpha_back * Lback
    # pseudo-likelihood approximation
    def pseudo_likelihood(self, data_post, data_back, omega_post=1, omega_back=1):
        Lback = self.RBMback.pseudo_likelihood(data_back)
        Lpost = self.RBMpost.pseudo_likelihood(data_post)
        alpha_back = data_back.shape[0] / (data_back.shape[0] + data_post.shape[0]) * omega_post
        alpha_post = data_post.shape[0] / (data_back.shape[0] + data_post.shape[0]) * omega_back
        return alpha_post * Lpost.mean() + alpha_back * Lback.mean()
            
    # fits top RBM from post data, with background RBM frozen
    def fit_top(self, data_post, l2_fields_top = 0, zero_back_grad_on_post=False,
                callback=None, modify_gradients_callback=None, modify_regularization_callback=None, 
                **kwargs):
        def _callback():
            self.update_post_from_back(vlayer=False, hlayer=True)
            if callback is not None:
                callback()
        def _modify_gradients_callback():
            if zero_back_grad_on_post:
                self.zero_back_grad_on_post(vlayer=False, hlayer=True)
            if modify_gradients_callback is not None:
                modify_gradients_callback()
        def _modify_regularization_callback():
            self.RBMpost.gradient['vlayer']['fields'][:self.n_v_] -= l2_fields_top * (self.RBMpost.vlayer.fields[:self.n_v_] - self.RBMback.vlayer.fields)
            if modify_regularization_callback is not None:
                modify_regularization_callback()

        self.RBMpost.fit(data_post, callback=_callback,
                         modify_regularization_callback=_modify_regularization_callback,
                         modify_gradients_callback=_modify_gradients_callback, 
                         **kwargs)
    
    # sets to zero the gradient of background parameters on postRBM
    def zero_back_grad_on_post(self, vlayer=False, hlayer=True):
        self.RBMpost.gradient['weights'][:self.n_h_]
        if vlayer:
            for key, g in self.RBMpost.gradient['vlayer'].items():
                g[:self.n_v_] = 0
        if hlayer:
            for key, g in self.RBMpost.gradient['hlayer'].items():
                g[:self.n_h_] = 0

    # fits background RBM from back data (doesn't change top RBM)
    def fit_back(self, data_back, **kwargs):
        self.RBMback.fit(data_back, **kwargs)
        self.update_post_from_back(vlayer=False, hlayer=True)
    
    # fits post RBM from post data (normal fit), and updates backRBM
    def fit_post(self, data_post, **kwargs):
        self.RBMpost.fit(data_post, **kwargs)
        self.update_back_from_post(vlayer=False, hlayer=True)
    
    # fits background RBM from post data, with top RBM frozen
    def fit_back_from_post(self, data_post, callback=None, **kwargs):
        topRBM = self.top_rbm()
        def cb():
            self.update_post_from_top(topRBM, vlayer=False, hlayer=True)
            if callback is not None:
                callback()
        self.RBMpost.fit(data_post, callback=cb, **kwargs)
        self.update_back_from_post(vlayer=True, hlayer=True)
        self.update_post_from_top(topRBM, vlayer=True, hlayer=True)

    # fits top and back RBMs simultaneously
    def fit_diff(self, data_post, data_back, weights_post=None, weights_back=None, l2_fields_back=0, l2_fields_top=0, omega_post=1, omega_back=1,
                 reg_diff=True, batch_size=100, n_iter=10, shuffle_data=True, modify_gradients_callback=None, **kwargs):
        
        assert self.RBMpost.n_v == self.RBMback.n_v # I don't know how to handle RBMpost.n_v > RBMback.n_v

        alpha_back = data_back.shape[0] / (data_back.shape[0] + data_post.shape[0]) * omega_post
        alpha_post = data_post.shape[0] / (data_back.shape[0] + data_post.shape[0]) * omega_back

        data_post = np.asarray(data_post, dtype=self.RBMpost.vlayer.type, order="c")
        data_back = np.asarray(data_back, dtype=self.RBMback.vlayer.type, order="c")

        n_samples_back = data_back.shape[0]
        n_batches_back = int(np.ceil(float(n_samples_back) / batch_size))
        back_batch_slices = list(gen_even_slices(n_batches_back * batch_size, n_batches_back, n_samples_back))
        assert len(back_batch_slices) == n_batches_back
        back_batch_slice_idx = 0

        # initialize fit (n_iter = 0 does nothing)
        self.RBMpost.fit(data_post, weights=weights_post, shuffle_data=False, batch_size=batch_size, n_iter=0, l2_fields=l2_fields_back, **kwargs)
        self.RBMback.fit(data_back, weights=weights_back, shuffle_data=False, batch_size=batch_size, n_iter=0, l2_fields=l2_fields_top, **kwargs)

        # in RBMback we only update the visible layer
        self.RBMback.do_grad_updates['weights'] = False
        for key in self.RBMback.do_grad_updates['hlayer']:
            self.RBMback.do_grad_updates['hlayer'][key] = False

        def _modify_gradients_callback():
            nonlocal back_batch_slice_idx, data_back, weights_back

            self.update_back_from_post(hlayer=True)

            if back_batch_slice_idx == 0 and shuffle_data:
                data_back, weights_back = rand_shuffle_data(data_back, weights_back)
            
            back_batch_slice = back_batch_slices[back_batch_slice_idx]
            
            # compute back gradients on back data
            if weights_back is None:
                no_nans = self.RBMback.minibatch_fit(data_back[back_batch_slice], weights=None)
            else:
                no_nans = self.RBMback.minibatch_fit(data_back[back_batch_slice], weights=weights_back[back_batch_slice])
            
            self.RBMpost.gradient['weights'] *= alpha_post
            self.RBMpost.gradient['weights'][:self.n_h_] += alpha_back * self.RBMback.gradient['weights']
            for key in self.RBMpost.gradient['hlayer']:
                self.RBMpost.gradient['hlayer'][key] *= alpha_post
                self.RBMpost.gradient['hlayer'][key][:self.n_h_] += alpha_back * self.RBMback.gradient['hlayer'][key]
            
            if reg_diff and self.RBMpost.tmp_l2_fields > 0: # regularize field differences
                self.RBMpost.gradient['vlayer']['fields'][:self.n_v_] += self.RBMpost.tmp_l2_fields * self.RBMback.vlayer.fields
            
            back_batch_slice_idx = (back_batch_slice_idx + 1) % n_batches_back

        return self.RBMpost.fit(data_post, weights=weights_post, shuffle_data=False, batch_size=batch_size, n_iter=n_iter, l2_fields=l2_fields_top,
                                modify_gradients_callback=_modify_gradients_callback, **kwargs)
        

# constructs a diff RBM from parameters
def construct_diff_rbm(n_v_post, n_h_post, n_v_back=None, n_h_back=None, n_cv=1, n_ch=1, visible='Bernoulli', hidden='Bernoulli'):
    if n_v_back is None:
        n_v_back = n_v_post
    if n_h_back is None:
        n_h_back = n_h_post
    RBMback = rbm.RBM(n_v=n_v_back, n_h=n_h_back, visible=visible, hidden=hidden, n_cv=n_cv, n_ch=n_ch)
    RBMpost = rbm.RBM(n_v=n_v_post, n_h=n_h_post, visible=visible, hidden=hidden, n_cv=n_cv, n_ch=n_ch)
    RBMdiff = DiffRBM(RBMback, RBMpost, update_back=True, update_back_vlayer=True, update_back_hlayer=True)
    return RBMdiff
# %%
