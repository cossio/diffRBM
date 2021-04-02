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
        
        self.count_updates = self.RBMpost.count_updates = self.RBMback.count_updates = 0

        if verbose:
            lik_back = (self.RBMback.pseudo_likelihood(data_back) * weights_back).sum() / weights_back.sum()
            lik_post = (self.RBMpost.pseudo_likelihood(data_post) * weights_post).sum() / weights_post.sum()
            print('Iteration number 0, pseudo-likelihood back: %.2f' % lik_back, 'pseudo-likelihood post: %.2f' % lik_post)

        n_samples_max = max(n_samples_back, n_samples_post)
        enlarge_back_idx = self.RBMback.random_state.choice(range(n_samples_back), size=(n_iter, n_samples_max - n_samples_back), p=weights_back / weights_back.sum())
        enlarge_post_idx = self.RBMpost.random_state.choice(range(n_samples_post), size=(n_iter, n_samples_max - n_samples_post), p=weights_post / weights_post.sum())

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
            permute_back = np.arange(n_samples_max)
            permute_post = np.arange(n_samples_max)
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
            assert data_back_.shape[0] == weights_back_.shape[0] == n_samples_max
            assert data_post_.shape[0] == weights_post_.shape[0] == n_samples_max

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

    def minibatch_fit_diff(self, V_pos_back, V_pos_post, weights=None, verbose=True, modify_gradients_callback=None):
        self.count_updates += 1
        assert self.N_PT == 0
        assert self.N_MC > 0
        assert not self.from_hidden
        if self.CD:  # Contrastive divergence: initialize the Markov chain at the data point.
            # Copy the value, not the pointer. DO NOT USE self.fantasy_v = V_pos
            self.RBMback.fantasy_v[:V_pos_back.shape[0]] = V_pos_back
            self.RBMpost.fantasy_v[:V_pos_post.shape[0]] = V_pos_post
        # Else: use previous value.
        for _ in range(self.N_MC):
            (self.RBMback.fantasy_v, self.RBMback.fantasy_h) = self.RBMback.markov_step((self.RBMback.fantasy_v, self.RBMback.fantasy_h))
            (self.RBMpost.fantasy_v, self.RBMpost.fantasy_h) = self.RBMpost.markov_step((self.RBMpost.fantasy_v, self.RBMpost.fantasy_h))

        for attr in ['fantasy_v', 'fantasy_h', 'fantasy_z', 'fantasy_E']:
            if hasattr(self.RBMback, attr):
                if np.isnan(getattr(self.RBMback, attr)).max():
                    print('NAN in RBMback %s (before gradient computation). Breaking' % attr)
                    return False
            if hasattr(self.RBMpost, attr):
                if np.isnan(getattr(self.RBMpost, attr)).max():
                    print('NAN in RBMpost %s (before gradient computation). Breaking' % attr)
                    return False

        if self.CD:
            weights_neg = weights
        else:
            weights_neg = None
    
        V_neg_back = self.RBMback.fantasy_v
        V_neg_post = self.RBMpost.fantasy_v
        I_neg_back = self.RBMback.vlayer.compute_output(V_neg_back, self.RBMback.weights)
        I_neg_post = self.RBMpost.vlayer.compute_output(V_neg_post, self.RBMpost.weights)
        I_pos_back = self.RBMback.vlayer.compute_output(V_pos_back, self.RBMback.weights)
        I_pos_post = self.RBMpost.vlayer.compute_output(V_pos_post, self.RBMpost.weights)

        if self.batch_norm:
            if (self.n_cv > 1) & (self.n_ch == 1):
                mu_I = np.tensordot(self.weights, self.mu_data, axes=[(1, 2), (0, 1)])
            elif (self.n_cv > 1) & (self.n_ch > 1):
                mu_I = np.tensordot(self.weights, self.mu_data, axes=[(1, 3), (0, 1)])
            elif (self.n_cv == 1) & (self.n_ch > 1):
                mu_I = np.tensordot(self.weights, self.mu_data, axes=[1, 0])
            else:
                mu_I = np.dot(self.weights, self.mu_data)

            self.hlayer.batch_norm_update(
                mu_I, I_pos, lr=0.25 * self.learning_rate / self.learning_rate_init, weights=weights)

        H_pos = self.hlayer.mean_from_inputs(I_pos)

        if (self.from_MoI & self.zero_track_RBM):
            data_0v = np.swapaxes(self.weights_MoI[0], 0, 1)
            weights_0v = self.zlayer.mu[0]
            data_0h = None
            weights_0h = None
        elif self.from_MoI_h:
            data_0v = None
            weights_0v = None
            data_0h = np.swapaxes(self.weights_MoI[0], 0, 1)
            weights_0h = self.zlayer.mu[0]
        else:
            data_0v = None
            data_0h = None
            weights_0v = None
            weights_0h = None

        if self.from_MoI & self.zero_track_RBM:
            Z = self.zlayer.mean_from_inputs(
                None, I0=self.vlayer.compute_output(V_neg, self.weights_MoI), beta=0)
            self.gradient['weights_MoI'] = utilities.average_product(
                Z, V_neg, mean1=True, mean2=False, c1=self.zlayer.n_c, c2=self.vlayer.n_c) - self.muzx
        elif self.from_MoI_h:
            if self.zero_track_RBM:
                Z = self.zlayer.mean_from_inputs(None, I0=self.hlayer.compute_output(
                    self.fantasy_h[0], self.weights_MoI), beta=0)
                self.gradient['weights_MoI'] = utilities.average_product(
                    Z, self.fantasy_h[0], mean1=True, mean2=False, c1=self.zlayer.n_c, c2=self.hlayer.n_c) - self.muzx
            else:
                H_pos_sample = self.hlayer.sample_from_inputs(I_pos)
                Z = self.zlayer.mean_from_inputs(None, I0=self.hlayer.compute_output(
                    H_pos_sample, self.weights_MoI), beta=0)
                self.gradient['weights_MoI'] = utilities.average_product(
                    Z, H_pos_sample, mean1=True, mean2=False, c1=self.zlayer.n_c, c2=self.hlayer.n_c) - self.muzx

        if self.from_hidden:
            self.gradient['vlayer'] = self.vlayer.internal_gradients(self.moments_data, I_neg, data_0=data_0v,
                                                                     weights=None, weights_neg=weights_neg, weights_0=weights_0v,
                                                                     value='moments', value_neg='input', value_0='input')

            self.gradient['hlayer'] = self.hlayer.internal_gradients(I_pos, H_neg, data_0=data_0h,
                                                                     weights=weights, weights_neg=weights_neg, weights_0=weights_0h,
                                                                     value='input', value_neg='data', value_0='input')

            V_neg = self.vlayer.mean_from_inputs(I_neg)
            self.gradient['weights'] = pgm.couplings_gradients_h(self.weights, H_pos, H_neg, V_pos, V_neg, self.n_ch, self.n_cv, l1=self.l1,
                                                                 l1b=self.l1b, l1c=self.l1c, l2=self.l2, weights=weights, weights_neg=weights_neg, l1_custom=self.l1_custom, l1b_custom=self.l1b_custom)

        else:
            self.gradient['vlayer'] = self.vlayer.internal_gradients(self.moments_data, V_neg, data_0=data_0v,
                                                                     weights=weights, weights_neg=weights_neg, weights_0=weights_0v,
                                                                     value='moments', value_neg='data', value_0='input')

            self.gradient['hlayer'] = self.hlayer.internal_gradients(I_pos, I_neg, data_0=data_0h,
                                                                     weights=weights, weights_neg=weights_neg, weights_0=weights_0h,
                                                                     value='input', value_neg='input', value_0='input')

            H_neg = self.hlayer.mean_from_inputs(I_neg)
            self.gradient['weights'] = pgm.couplings_gradients(self.weights, H_pos, H_neg, V_pos, V_neg, self.n_ch, self.n_cv, mean1=True, l1=self.l1,
                                                               l1b=self.l1b, l1c=self.l1c, l2=self.l2, weights=weights, weights_neg=weights_neg, l1_custom=self.l1_custom, l1b_custom=self.l1b_custom)
       
        # this callback allows the user to modify gradients (via the RBM.gradients dictionary)
        # before they are given to the optimizer
        if modify_gradients_callback is not None:
            modify_gradients_callback()

        if self.interpolate & (self.N_PT > 2):
            self.gradient['vlayer'] = self.vlayer.internal_gradients_interpolation(
                self.fantasy_v, self.betas, gradient=self.gradient['vlayer'], value='data')
            self.gradient['hlayer'] = self.hlayer.internal_gradients_interpolation(
                self.fantasy_h, self.betas, gradient=self.gradient['hlayer'], value='data')
        if self.interpolate_z & (self.N_PT > 2):
            self.gradient['zlayer'] = self.zlayer.internal_gradients_interpolation(
                self.fantasy_z, self.betas, value='data')

        if check_nan(self.gradient, what='gradient', location='before batch norm'):
            self.vproblem = V_pos
            self.Iproblem = I_pos
            return False

        if self.batch_norm:  # Modify gradients.
            self.hlayer.batch_norm_update_gradient(
                self.gradient['weights'], self.gradient['hlayer'], V_pos, I_pos, self.mu_data, self.n_cv, weights=weights)

        if check_nan(self.gradient, what='gradient', location='after batch norm'):
            self.vproblem = V_pos
            self.Iproblem = I_pos
            return False

        for key, item in self.gradient.items():
            if type(item) == dict:
                for key_, item_ in item.items():
                    saturate(item_, 1.0)
            else:
                saturate(item, 1.0)

        if self.tmp_l2_fields > 0:
            self.gradient['vlayer']['fields'] -= self.tmp_l2_fields * \
                self.vlayer.fields
        if not self.tmp_reg_delta == 0:
            self.gradient['hlayer']['delta'] -= self.tmp_reg_delta

        for key, item in self.gradient.items():
            if type(item) == dict:
                for key_, item_ in item.items():
                    current = getattr(getattr(self, key), key_)
                    do_update = self.do_grad_updates[key][key_]
                    lr_multiplier = self.learning_rate_multiplier[key][key_]
                    has_momentum = self.has_momentum[key][key_]
                    gradient = item_
                    if do_update:
                        if self.optimizer == 'SGD':
                            current += self.learning_rate * lr_multiplier * gradient
                        elif self.optimizer == 'momentum':
                            self.previous_update[key][key_] = (
                                1 - self.momentum) * self.learning_rate * lr_multiplier * gradient + self.momentum * self.previous_update[key][key_]
                            current += self.previous_update[key][key_]
                        elif self.optimizer == 'ADAM':
                            if has_momentum:
                                self.gradient_moment1[key][key_] *= self.beta1
                                self.gradient_moment1[key][key_] += (
                                    1 - self.beta1) * gradient
                                self.gradient_moment2[key][key_] *= self.beta2
                                self.gradient_moment2[key][key_] += (
                                    1 - self.beta2) * gradient**2
                                current += self.learning_rate * lr_multiplier / (1 - self.beta1) * (self.gradient_moment1[key][key_] / (
                                    1 - self.beta1**self.count_updates)) / (self.epsilon + np.sqrt(self.gradient_moment2[key][key_] / (1 - self.beta2**self.count_updates)))
                            else:
                                self.gradient_moment2[key][key_] *= self.beta2
                                self.gradient_moment2[key][key_] += (
                                    1 - self.beta2) * gradient**2
                                current += self.learning_rate * lr_multiplier * gradient / \
                                    (self.epsilon + np.sqrt(self.gradient_moment2[key][key_] / (
                                        1 - self.beta2**self.count_updates)))

                            # self.gradient_moment1[key][key_] *= self.beta1
                            # self.gradient_moment1[key][key_] += (1- self.beta1) * gradient
                            # self.gradient_moment2[key][key_] *= self.beta2
                            # self.gradient_moment2[key][key_] += (1- self.beta2) * gradient**2
                            # current += self.learning_rate * (self.gradient_moment1[key][key_]/(1-self.beta1**self.count_updates)) /(self.epsilon + np.sqrt( self.gradient_moment2[key][key_]/(1-self.beta2**self.count_updates ) ) )
            else:
                current = getattr(self, key)
                do_update = self.do_grad_updates[key]
                lr_multiplier = self.learning_rate_multiplier[key]
                has_momentum = self.has_momentum[key]
                gradient = item
                if do_update:
                    if self.optimizer == 'SGD':
                        current += self.learning_rate * lr_multiplier * gradient
                    elif self.optimizer == 'momentum':
                        self.previous_update[key] = (
                            1 - self.momentum) * self.learning_rate * lr_multiplier * gradient + self.momentum * self.previous_update[key]
                        current += self.previous_update[key]
                    elif self.optimizer == 'ADAM':
                        if has_momentum:
                            self.gradient_moment1[key] *= self.beta1
                            self.gradient_moment1[key] += (
                                1 - self.beta1) * gradient
                            self.gradient_moment2[key] *= self.beta2
                            self.gradient_moment2[key] += (
                                1 - self.beta2) * gradient**2
                            current += self.learning_rate * lr_multiplier / (1 - self.beta1) * (self.gradient_moment1[key] / (
                                1 - self.beta1**self.count_updates)) / (self.epsilon + np.sqrt(self.gradient_moment2[key] / (1 - self.beta2**self.count_updates)))
                        else:
                            self.gradient_moment2[key] *= self.beta2
                            self.gradient_moment2[key] += (
                                1 - self.beta2) * gradient**2
                            current += self.learning_rate * lr_multiplier * gradient / \
                                (self.epsilon + np.sqrt(self.gradient_moment2[key] / (
                                    1 - self.beta2**self.count_updates)))

                        # self.gradient_moment1[key] *= self.beta1
                        # self.gradient_moment1[key] += (1- self.beta1) * gradient
                        # self.gradient_moment2[key] *= self.beta2
                        # self.gradient_moment2[key] += (1- self.beta2) * gradient**2
                        # current += self.learning_rate * (self.gradient_moment1[key]/(1-self.beta1**self.count_updates)) /(self.epsilon + np.sqrt( self.gradient_moment2[key]/(1-self.beta2**self.count_updates ) ) )

        if (self.n_cv > 1) | (self.n_ch > 1):
            pgm.gauge_adjust_couplings(
                self.weights, self.n_ch, self.n_cv, gauge=self.gauge)

        self.hlayer.recompute_params()
        self.vlayer.ensure_constraints()
        self.hlayer.ensure_constraints()

        if check_nan(self.hlayer.__dict__, what='hlayer', location='after recompute parameters'):
            return False

        if (self.from_MoI & self.zero_track_RBM) | self.from_MoI_h:
            if self.from_MoI:
                layer_id = 0
                n_cx = self.vlayer.n_c
                # fantasy_x = self.fantasy_v[0]
            else:
                layer_id = 1
                n_cx = self.hlayer.n_c
                # if self.zero_track_RBM:
                # fantasy_x = self.fantasy_h[0]
                # else:
                # fantasy_x = H_pos_sample

            if self.zero_track_RBM:
                weights_Z = weights_neg
            else:
                weights_Z = weights

            pgm.gauge_adjust_couplings(
                self.weights_MoI, self.zlayer.n_c, n_cx, gauge=self.gauge)
            # muz = utilities.average(Z,weights=weights_Z)
            # if weights_Z is None:
            #     likz =  np.dot(self.likelihood_mixture(fantasy_x),Z[:,0])/(muz[0] * fantasy_x.shape[0] )
            # else:
            #     likz =  np.dot(self.likelihood_mixture(fantasy_x) * weights_Z,Z[:,0])/(muz[0] * fantasy_x.shape[0] )

            self.zlayer.mu = (1 - self.update_zlayer_mu_lr) * self.zlayer.mu + \
                self.update_zlayer_mu_lr * \
                utilities.average(Z, weights=weights_Z)
            # self.zlayer.average_likelihood = 0. * self.zlayer.average_likelihood + 1 * likz
            self.update_params_MoI(
                layer_id=layer_id, eps=1e-4, verbose=verbose)

        if self.N_PT > 1:
            if self._update_betas:
                self.update_betas()

        return True

        



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
