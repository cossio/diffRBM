import numpy as np
import layer
import pgm
import moi
import utilities
from utilities import check_random_state, gen_even_slices, logsumexp, log_logistic, average, average_product, saturate, check_nan, get_permutation
import time
import copy
import rbm

from float_precision import double_precision, curr_float, curr_int


# %%


class DiffRBM:
    def __init__(self, RBMback, RBMpost, update_back=True, update_back_vlayer=False, update_back_hlayer=True):
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
    
    # updates back RBM from post parameters
    def update_back_from_post(self, vlayer=False, hlayer=True):
        self.RBMback.weights[:] = self.RBMpost.weights[:self.n_h_, :self.n_v_]
        if vlayer:
            for key in self.RBMback.vlayer.list_params:
                self.RBMback.vlayer.__dict__[key][:] = self.RBMpost.vlayer.__dict__[key][:self.n_v_]
        if hlayer:
            for key in self.RBMback.hlayer.list_params:
                self.RBMback.hlayer.__dict__[key][:] = self.RBMpost.hlayer.__dict__[key][:self.n_h_]
    
    # updates post RBM from back parameters
    def update_post_from_back(self, vlayer=False, hlayer=True):
        self.RBMpost.weights[:self.n_h_, :self.n_v_] = self.RBMback.weights
        if vlayer:
            for key in self.RBMpost.vlayer.list_params:
                self.RBMpost.vlayer.__dict__[key][:self.n_v_] = self.RBMback.vlayer.__dict__[key]
        if hlayer:
            for key in self.RBMpost.hlayer.list_params:
                self.RBMpost.hlayer.__dict__[key][:self.n_h_] = self.RBMback.hlayer.__dict__[key]

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

    # returns the top RBM
    def top_rbm(self):
        RBMtop = rbm.RBM(n_v=self.RBMpost.n_v, n_h=self.RBMpost.n_h - self.n_h_,
                         n_cv=self.RBMpost.n_cv, n_ch=self.RBMpost.n_ch,
                         visible=self.RBMpost.visible, hidden=self.RBMpost.hidden)
        self.update_top_from_post(RBMtop, vlayer=True, hlayer=True)
        return RBMtop
            
    # fits top RBM from post data, with background RBM frozen
    def fit_top(self, data_post, callback=None, **kwargs):
        def cb():
            self.update_post_from_back(vlayer=False, hlayer=True)
            if callback is not None:
                callback()
        self.RBMpost.fit(data_post, callback=cb, **kwargs)
    
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
    def fit_diff(self, data_post, data_back, batch_size=100, learning_rate=None, extra_params=None, init='independent', optimizer='ADAM', batch_norm=True, CD=False, N_MC=1, nchains=None, n_iter=10,
                interpolate_z=True, degree_interpolate_z=5,
                lr_decay=True, lr_final=None, decay_after=0.5, l1=0, l1b=0, l1c=0, l2=0, l2_fields=0, reg_delta=0, weights_post=None, weights_back=None,
                learning_rate_multiplier=1, epsilon=1e-6, verbose=1, vverbose=0, l1_custom=None, l1b_custom=None,
                callback=None # callback function, called after each minibatch fit
                ):

        self.RBMpost.batch_size = self.RBMback.batch_size = batch_size
        self.RBMpost.optimizer = self.RBMback.optimizer = optimizer
        self.RBMpost.batch_norm = self.RBMback.batch_norm = batch_norm
        self.RBMpost.hlayer.batch_norm = self.RBMback.hlayer.batch_norm = batch_norm
        self.RBMpost.record_swaps = self.RBMback.record_swaps = False
        self.RBMpost.only_sampling = self.RBMback.only_sampling = False
        self.RBMpost.zero_track_RBM = self.RBMback.zero_track_RBM = False
        self.RBMpost.adapt_PT = self.RBMback.adapt_PT = False
        self.RBMpost.adapt_MC = self.RBMback.adapt_MC = False
        self.RBMpost.N_PT = self.RBMback.N_PT = 1
        self.RBMpost.N_MC = self.RBMback.N_MC = N_MC
        self.RBMpost.from_hidden = self.RBMback.from_hidden = False
        self.RBMpost.no_fields = self.RBMback.no_fields = False
        self.RBMpost.n_iter = self.RBMback.n_iter = n_iter
        self.RBMpost.CD = self.RBMback.CD = CD
        self.RBMpost.l1 = self.RBMback.l1 = l1
        self.RBMpost.l1b = self.RBMback.l1b = l1b
        self.RBMpost.l1c = self.RBMback.l1c = l1c
        self.RBMpost.l1_custom = self.RBMback.l1_custom = l1_custom
        self.RBMpost.l1b_custom = self.RBMback.l1b_custom = l1b_custom
        self.RBMpost.l2 = self.RBMback.l2 = l2
        self.RBMpost.tmp_l2_fields = self.RBMback.tmp_l2_fields = l2_fields
        self.RBMpost.tmp_reg_delta = self.RBMback.tmp_reg_delta = reg_delta
        self.RBMpost.from_MoI = self.RBMback.from_MoI = False
        self.RBMpost.from_MoI_h = self.RBMback.from_MoI_h = False
        self.RBMpost.interpolate_z = self.RBMback.interpolate_z = False
        self.RBMpost._update_betas = self.RBMback._update_betas = False
        self.RBMpost.PTv = self.RBMback.PTv = False
        self.RBMpost.PTh = self.RBMback.PTh = False

        if weights_post is None:
            weights_post = np.ones(len(data_post))
        if weights_back is None:
            weights_back = np.ones(len(data_back))
        
        data_post = np.asarray(data_post, dtype=self.RBMpost.vlayer.type, order="c")
        data_back = np.asarray(data_back, dtype=self.RBMback.vlayer.type, order="c")
        weights_post = np.asarray(weights_post, dtype=curr_float)
        weights_back = np.asarray(weights_back, dtype=curr_float)
        self.RBMpost.mu_data = utilities.average(data_post, c=self.RBMpost.n_cv, weights=weights_post)
        self.RBMback.mu_data = utilities.average(data_back, c=self.RBMback.n_cv, weights=weights_back)
        self.RBMpost.moments_data = self.RBMpost.vlayer.get_moments(data_post, value='data', weights=weights_post, beta=1)
        self.RBMback.moments_data = self.RBMback.vlayer.get_moments(data_back, value='data', weights=weights_back, beta=1)

        n_samples_back = data_back.shape[0]
        n_samples_post = data_post.shape[0]
        n_samples_max = max(n_samples_back, n_samples_post)
        n_batches = int(np.ceil(float(n_samples_max) / batch_size))
        batch_slices = list(gen_even_slices(n_batches * batch_size, n_batches, n_samples_max))

        # learning rate should be smaller for the smaller dataset
        if n_samples_back > n_samples_post:
            lr_back_factor = 1
            lr_post_factor = n_samples_post / n_samples_back
        else:
            lr_back_factor = n_samples_back / n_samples_post
            lr_post_factor = 1

        if n_iter <= 1:
            lr_decay = False

        if learning_rate is None:
            if optimizer in ['SGD', 'momentum']:
                if self.RBMpost.hidden in ['Gaussian', 'ReLU+', 'ReLU', 'dReLU']:
                    if batch_norm:
                        learning_rate = 0.05
                    else:
                        learning_rate = 5e-3
                else:
                    learning_rate = 0.05
            elif optimizer == 'ADAM':
                learning_rate = 5e-3
            else:
                print('Need to specify learning rate for optimizer.')

        self.RBMback.learning_rate_init = learning_rate * lr_back_factor
        self.RBMpost.learning_rate_init = learning_rate * lr_post_factor
        self.RBMback.learning_rate = learning_rate * lr_back_factor
        self.RBMpost.learning_rate = learning_rate * lr_post_factor
        self.RBMpost.lr_decay = self.RBMback.lr_decay = lr_decay
        if lr_decay:
            self.RBMpost.decay_after = self.RBMback.decay_after = decay_after
            self.RBMpost.start_decay = self.RBMback.start_decay = start_decay = int(n_iter * decay_after)
            if lr_final is None:
                lr_final = 1e-2 * learning_rate
            self.RBMback.lr_final = lr_final * lr_back_factor
            self.RBMpost.lr_final = lr_final * lr_post_factor
            decay_gamma = (float(lr_final) / float(learning_rate))**(1 / float(n_iter * (1 - decay_after)))
        else:
            decay_gamma = 1
        self.RBMpost.decay_gamma = self.RBMback.decay_gamma = decay_gamma

        if init != 'previous':
            self.RBMback.init_weights(np.sqrt(0.1 / self.RBMback.n_v))
            self.RBMpost.init_weights(np.sqrt(0.1 / self.RBMpost.n_v))
            if init == 'independent':
                self.RBMback.vlayer.init_params_from_data(self.RBMback.moments_data, eps=epsilon, value='moments')
                self.RBMpost.vlayer.init_params_from_data(self.RBMpost.moments_data, eps=epsilon, value='moments')
            self.RBMpost.hlayer.init_params_from_data(None)
            self.RBMback.hlayer.init_params_from_data(None)
            self.update_back_from_post(vlayer=False, hlayer=True)

        if nchains is None:
            nchains = batch_size
        self.RBMpost.nchains = self.RBMback.nchains = nchains

        self.RBMpost.hlayer.update0 = self.RBMback.hlayer.update0 = True
        self.RBMpost.vlayer.update0 = self.RBMback.vlayer.update0 = False
        self.RBMpost.hlayer.target0 = self.RBMback.hlayer.target0 = 'pos'
        self.RBMpost.vlayer.target0 = self.RBMback.vlayer.target0 = 'pos'

        self.RBMpost.fantasy_v = self.RBMpost.vlayer.random_init_config(nchains)
        self.RBMpost.fantasy_h = self.RBMpost.hlayer.random_init_config(nchains)
        self.RBMback.fantasy_v = self.RBMback.vlayer.random_init_config(nchains)
        self.RBMback.fantasy_h = self.RBMback.hlayer.random_init_config(nchains)

        self.RBMpost.gradient = self.RBMpost.initialize_gradient_dictionary()
        self.RBMback.gradient = self.RBMback.initialize_gradient_dictionary()
        self.RBMpost.do_grad_updates = {'vlayer': self.RBMpost.vlayer.do_grad_updates, 'weights': True}
        self.RBMback.do_grad_updates = {'vlayer': self.RBMback.vlayer.do_grad_updates, 'weights': True}
        if batch_norm:
            self.RBMpost.do_grad_updates['hlayer'] = self.RBMpost.hlayer.do_grad_updates_batch_norm
            self.RBMback.do_grad_updates['hlayer'] = self.RBMback.hlayer.do_grad_updates_batch_norm
        else:
            self.RBMpost.do_grad_updates['hlayer'] = self.RBMpost.hlayer.do_grad_updates
            self.RBMback.do_grad_updates['hlayer'] = self.RBMback.hlayer.do_grad_updates

        if optimizer == 'momentum':
            if extra_params is None:
                extra_params = 0.9
            self.RBMpost.momentum = self.RBMback.momentum = extra_params
            self.RBMpost.previous_update = self.RBMpost.initialize_gradient_dictionary()
            self.RBMback.previous_update = self.RBMback.initialize_gradient_dictionary()
        elif optimizer == 'ADAM':
            if extra_params is None:
                extra_params = [0.99, 0.99, 1e-3]
            self.RBMpost.beta1 = self.RBMback.beta1 = extra_params[0]
            self.RBMpost.beta2 = self.RBMback.beta2 = extra_params[1]
            self.RBMpost.epsilon = self.RBMback.epsilon = extra_params[2]
            self.RBMpost.gradient_moment1 = self.RBMpost.initialize_gradient_dictionary()
            self.RBMpost.gradient_moment2 = self.RBMpost.initialize_gradient_dictionary()
            self.RBMback.gradient_moment1 = self.RBMback.initialize_gradient_dictionary()
            self.RBMback.gradient_moment2 = self.RBMback.initialize_gradient_dictionary()

        self.RBMpost.learning_rate_multiplier = {}
        self.RBMback.learning_rate_multiplier = {}
        for key in self.RBMpost.gradient.keys():
            if type(self.RBMpost.gradient[key]) == dict:
                self.RBMpost.learning_rate_multiplier[key] = {}
                self.RBMback.learning_rate_multiplier[key] = {}
                for key_ in self.RBMpost.gradient[key].keys():
                    if ('0' in key_) | ('1' in key_):
                        self.RBMpost.learning_rate_multiplier[key][key_] = learning_rate_multiplier
                        self.RBMback.learning_rate_multiplier[key][key_] = learning_rate_multiplier
                    else:
                        self.RBMpost.learning_rate_multiplier[key][key_] = 1
                        self.RBMback.learning_rate_multiplier[key][key_] = 1
            else:
                self.RBMpost.learning_rate_multiplier[key] = 1
                self.RBMback.learning_rate_multiplier[key] = 1

        self.RBMpost.has_momentum = {}
        self.RBMback.has_momentum = {}
        for key in self.RBMpost.gradient.keys():
            if type(self.RBMpost.gradient[key]) == dict:
                self.RBMpost.has_momentum[key] = {}
                self.RBMback.has_momentum[key] = {}
                for key_ in self.RBMpost.gradient[key].keys():
                    if ('0' in key_):
                        self.RBMpost.has_momentum[key][key_] = self.RBMback.has_momentum[key][key_] = True
                    else:
                        self.RBMpost.has_momentum[key][key_] = self.RBMback.has_momentum[key][key_] = False
            else:
                self.RBMpost.has_momentum[key] = self.RBMback.has_momentum[key] = False
        
        weights_post /= weights_post.mean()
        weights_back /= weights_back.mean()
        
        self.RBMpost.count_updates = self.RBMback.count_updates = 0

        if verbose:
            lik_back = (self.RBMback.pseudo_likelihood(data_back) * weights_back).sum() / weights_back.sum()
            lik_post = (self.RBMpost.pseudo_likelihood(data_post) * weights_post).sum() / weights_post.sum()
            print('Iteration number 0, pseudo-likelihood back: %.2f' % lik_back, 'pseudo-likelihood post: %.2f' % lik_post)

        num_samples_max = max(n_samples_back, n_samples_post)
        enlarge_back_idx = self.RBMback.random_state.choice(range(n_samples_back), size=(n_iter, num_samples_max - n_samples_back), p=weights_back / weights_back.sum())
        enlarge_post_idx = self.RBMpost.random_state.choice(range(n_samples_post), size=(n_iter, num_samples_max - n_samples_post), p=weights_post / weights_post.sum())

        for epoch in range(1, n_iter + 1):
            if verbose:
                begin = time.time()
            if lr_decay:
                if (epoch > start_decay):
                    learning_rate *= decay_gamma
                    self.RBMback.learning_rate = learning_rate * lr_back_factor
                    self.RBMpost.learning_rate = learning_rate * lr_post_factor

            if (verbose | vverbose):
                print('Starting epoch %s' % (epoch))
            
            # enlarge datasets to common number of samples and shuffle
            permute_back = np.arange(num_samples_max)
            permute_post = np.arange(num_samples_max)
            self.RBMback.random_state.shuffle(permute_back)
            self.RBMpost.random_state.shuffle(permute_post)
            enlarge_back_idx_ = np.concatenate([range(n_samples_back), enlarge_back_idx[epoch - 1]])
            enlarge_post_idx_ = np.concatenate([range(n_samples_post), enlarge_post_idx[epoch - 1]])
            enlarge_back_idx_ = enlarge_back_idx_[permute_back]
            enlarge_post_idx_ = enlarge_post_idx_[permute_post]
            data_back_ = data_back[enlarge_back_idx_]
            data_post_ = data_post[enlarge_post_idx_]
            weights_back_ = weights_back[enlarge_back_idx_]
            weights_post_ = weights_post[enlarge_post_idx_]
            assert data_back_.shape[0] == data_post_.shape[0] == num_samples_max
            assert weights_back_.shape[0] == weights_post_.shape[0] == num_samples_max

            for batch_slice in batch_slices:
                # update back
                no_nans_back = self.RBMback.minibatch_fit(data_back_[batch_slice], weights=weights_back_[batch_slice], verbose=vverbose)
                self.update_post_from_back(vlayer=False, hlayer=True)
                # update post
                no_nans_post = self.RBMpost.minibatch_fit(data_post_[batch_slice], weights=weights_post_[batch_slice], verbose=vverbose)
                self.update_back_from_post(vlayer=False, hlayer=True)
                
                if callback is not None:
                    callback()
                if no_nans_back and no_nans_post:
                    done = False
                else:
                    done = True
                    break
                
            if done:
                break

            if verbose:
                end = time.time()
                lik_back = (self.RBMback.pseudo_likelihood(data_back) * weights_back).sum() / weights_back.sum()
                lik_post = (self.RBMpost.pseudo_likelihood(data_post) * weights_post).sum() / weights_post.sum()
                print("[%s] Iteration %d, time = %.2fs, pseudo-likelihood back = %.2f, pseudo-likelihood post: %.2f" % (type(self).__name__, epoch, end - begin, lik_back, lik_post))

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
