from typing import Callable, NamedTuple
import tensorflow as tf
import numpy as np

__all__ = ['BackTrackingLineSearch']


class BackTrackingLineSearch:
    """Adapted from the backtracking line search in the manopt package"""
    def __init__(self, contraction_factor: float = 0.5,
                 optimism: float = 10.,
                 suff_decr: float = 1e-4,
                 initial_stepsize: float = 10.0,
                 dtype: np.dtype = np.float32,
                 maxiter: int = None) -> None:
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.suff_decr = suff_decr
        self.initial_stepsize = initial_stepsize

        self._dtype = dtype
        self._machine_eps = np.finfo(dtype).eps

        machine_maxiter = np.ceil(np.log(self._machine_eps) / np.log(self.contraction_factor))

        if maxiter is None:
            maxiter = np.inf
        self.maxiter = np.minimum(maxiter, machine_maxiter).astype('int32')

        self._oldf0 = tf.Variable(-np.inf, dtype='float32', name='old_f0')

    def search(self, objective_and_update: Callable,
               x0: tf.Tensor,
               descent_dir: tf.Tensor,
               gradient: tf.Tensor,
               f0: tf.Tensor = None):

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

        class LSState(NamedTuple):
            newf: tf.Tensor
            newx: tf.Tensor
            alpha: tf.Tensor
            step_count: tf.Tensor

        def _cond(state: LSState):

            cond1 = state.newf > f0 + self.suff_decr * state.alpha * df0
            cond2 = state.step_count <= self.maxiter
            return tf.logical_and(cond1, cond2)

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
            oldf0_assign_op = self._oldf0.assign(f0)

        with tf.control_dependencies([oldf0_assign_op]):
            lsstate_updated = tf.cond(lsstate_new.newf <= f0, sanity_check_true, sanity_check_false)

        return lsstate_updated