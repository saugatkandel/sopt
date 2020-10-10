from typing import Callable, NamedTuple
import tensorflow as tf
import numpy as np

__all__ = ['BackTrackingLineSearch', 'AdaptiveLineSearch']


class LSState(NamedTuple):
    newf: tf.Tensor
    newx: tf.Tensor
    alpha: tf.Tensor
    step_count: tf.Tensor


class BackTrackingLineSearch:
    """Adapted from the backtracking line search in the manopt package"""
    def __init__(self, contraction_factor: float = 0.5,
                 optimism: float = 3.,
                 suff_decr: float = 1e-4,
                 initial_stepsize: float = 10.0,
                 stepsize_threshold_low: float = 1e-10,
                 dtype: str = 'float32',
                 maxiter: int = None,
                 name='backtracking_linesearch') -> None:
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.suff_decr = suff_decr
        self.initial_stepsize = initial_stepsize
        self.stepsize_threshold_low = stepsize_threshold_low

        self._dtype = dtype
        self._machine_eps = np.finfo(np.dtype(dtype)).eps

        self._name = name

        machine_maxiter = np.ceil(np.log(self._machine_eps) / np.log(self.contraction_factor))

        if maxiter is None:
            maxiter = np.inf
        self.maxiter = np.minimum(maxiter, machine_maxiter).astype('int32')

        with tf.variable_scope(name):
            self._oldf0 = tf.Variable(-np.inf, dtype=self._dtype, name='old_f0')
            self._alpha = tf.Variable(0., dtype=self._dtype, name='alpha')

        self._variables = [self._oldf0, self._alpha]

    @property
    def reset(self):
        reset_ops = [v.assign(v.initial_value) for v in self._variables]
        return tf.group(reset_ops)

    def search(self, objective_and_update: Callable,
               x0: tf.Tensor,
               descent_dir: tf.Tensor,
               gradient: tf.Tensor,
               f0: tf.Tensor = None):

        with tf.name_scope(self._name + '_search'):
            if f0 is None:
                f0, _ = objective_and_update(x0, tf.zeros_like(x0))

            # Calculating the directional derivative along the descent direction
            descent_norm = tf.linalg.norm(descent_dir)
            df0 = tf.reduce_sum(descent_dir * gradient)

            def _alphafn_true():
                # Pick initial step size based on where we were last time
                alpha = 2 * (f0 - self._oldf0) / df0

                # Look a little further
                alpha *= self.optimism
                alpha = tf.cond(alpha * descent_norm < self._machine_eps,
                                lambda: self.initial_stepsize / descent_norm,
                                lambda: alpha)
                return alpha

            def _alphafn_false():
                return self.initial_stepsize / descent_norm

            alpha = tf.cond(self._oldf0 >= f0, _alphafn_true, _alphafn_false)

            # Make the chosen sten and compute the cost there
            newf, newx = objective_and_update(x0, alpha * descent_dir)
            step_count = 1

            # Backtrack while the Armijo criterion is not satisfied


            def _cond(state: LSState):

                cond1 = state.newf > f0 + self.suff_decr * state.alpha * df0
                cond2 = state.step_count <= self.maxiter
                cond3 = state.alpha > self.stepsize_threshold_low
                return tf.logical_and(tf.logical_and(cond1, cond2), cond3)

            def _body(state: LSState):
                alpha = self.contraction_factor * state.alpha
                newf, newx = objective_and_update(x0, alpha * descent_dir)
                return [LSState(newf=newf,
                                newx=newx,
                                alpha=alpha,
                                step_count=state.step_count + 1)]

            lsstate0 = LSState(newf=newf, newx=newx, alpha=alpha, step_count=tf.constant(step_count, dtype='int32'))
            [lsstate_new] = tf.while_loop(cond=_cond,
                                          body=_body,
                                          loop_vars=[lsstate0],
                                          back_prop=False)

            sanity_check_true = lambda: lsstate_new
            sanity_check_false = lambda: LSState(newf=f0, newx=x0, alpha=0., step_count=lsstate_new.step_count)

            with tf.control_dependencies([lsstate_new.step_count]):
                assign_ops = tf.group([self._oldf0.assign(f0), self._alpha.assign(lsstate_new.alpha)])

            with tf.control_dependencies([assign_ops]):
                lsstate_updated = tf.cond(lsstate_new.newf <= f0, sanity_check_true, sanity_check_false)

        return lsstate_updated


class AdaptiveLineSearch:
    """Adapted from the backtracking line search in the manopt package"""
    def __init__(self, contraction_factor: float = 0.5,
                 optimism: float = 2.,
                 suff_decr: float = 1e-4,
                 initial_stepsize: float = 10.0,
                 stepsize_threshold_low: float = 1e-10,
                 dtype: np.dtype = np.float32,
                 maxiter: int = None,
                 name='backtracking_linesearch') -> None:
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.suff_decr = suff_decr
        self.initial_stepsize = initial_stepsize
        self.stepsize_threshold_low = stepsize_threshold_low

        self._dtype = dtype
        self._machine_eps = np.finfo(dtype).eps

        self._name = name

        machine_maxiter = np.ceil(np.log(self._machine_eps) / np.log(self.contraction_factor))

        if maxiter is None:
            maxiter = np.inf
        self.maxiter = np.minimum(maxiter, machine_maxiter).astype('int32')

        with tf.variable_scope(name):
            self._alpha = tf.Variable(0., dtype=self._dtype, name='alpha')
            self._alpha_suggested = tf.Variable(0., dtype=self._dtype, name='alpha_suggested')

        self._variables = [self._alpha, self._alpha_suggested]

    @property
    def reset(self):
        reset_ops = [v.assign(v.initial_value) for v in self._variables]
        return tf.group(reset_ops)

    def search(self, objective_and_update: Callable,
               x0: tf.Tensor,
               descent_dir: tf.Tensor,
               gradient: tf.Tensor,
               f0: tf.Tensor = None):

        with tf.name_scope(self._name + '_search'):
            if f0 is None:
                f0, _ = objective_and_update(x0, tf.zeros_like(x0))

            # Calculating the directional derivative along the descent direction
            descent_norm = tf.linalg.norm(descent_dir)
            df0 = tf.reduce_sum(descent_dir * gradient)

            def _alphafn_true():
                # Pick initial step size based on where we were last time
                return self._alpha_suggested

            def _alphafn_false():
                return self.initial_stepsize / descent_norm
            
            alpha = tf.cond(self._alpha_suggested > 0, _alphafn_true, _alphafn_false)

            # Make the chosen sten and compute the cost there
            newf, newx = objective_and_update(x0, alpha * descent_dir)
            step_count = 1

            # Backtrack while the Armijo criterion is not satisfied
            def _cond(state: LSState):
                cond1 = state.newf > f0 + self.suff_decr * state.alpha * df0
                #cond1 = state.newf > f0 + self.suff_decr * tf.reduce_sum(gradient * (state.newx - x0))
                cond2 = (state.step_count <= self.maxiter)
                cond3 = state.alpha > self.stepsize_threshold_low
                return tf.logical_and(tf.logical_and(cond1, cond2), cond3)

            def _body(state: LSState):
                alpha = self.contraction_factor * state.alpha
                newf, newx = objective_and_update(x0, alpha * descent_dir)
                return [LSState(newf=newf,
                                newx=newx,
                                alpha=alpha,
                                step_count=state.step_count + 1)]

            lsstate0 = LSState(newf=newf, newx=newx, alpha=alpha, step_count=tf.constant(step_count, dtype='int32'))
            [lsstate_new] = tf.while_loop(cond=_cond,
                                          body=_body,
                                          loop_vars=[lsstate0],
                                          back_prop=False)

            sanity_check_true = lambda: lsstate_new
            sanity_check_false = lambda: LSState(newf=f0, newx=x0,
                                                 alpha=tf.constant(0., dtype=self._dtype),
                                                 step_count=lsstate_new.step_count)

            # New suggestion for step size
            # case 1: if things go very well (step count is 1), push your luck
            case1 = lambda: self.optimism * lsstate_new.alpha

            # case 2: if things go reasonably well (step count is 2), try to keep pace
            case2 = lambda: lsstate_new.alpha

            # case 3: if we backtracked a lot, the new stepsize is probably quite small:
            # try to recover
            case3 = lambda: self.optimism * lsstate_new.alpha


            # the (step_count - 1) is just a workaround for switch_case
            # switch case requires contiguous range starting form 0
            # we start from 1.

            suggested_alpha = tf.switch_case(branch_index=(lsstate_new.step_count - 1),
                                             branch_fns={0: case1, 1: case2},
                                             default=case3)

            with tf.control_dependencies([lsstate_new.step_count]):
                assign_ops = tf.group([self._alpha_suggested.assign(suggested_alpha),
                                       self._alpha.assign(lsstate_new.alpha)])
            with tf.control_dependencies([assign_ops]):
                lsstate_updated = tf.cond(lsstate_new.newf <= f0, sanity_check_true, sanity_check_false)

        return lsstate_updated