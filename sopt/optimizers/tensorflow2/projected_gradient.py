# Non linear conjugate gradient method
# Only float data types supported right now.
# This is fairly easy to change to add complex data types.
import numpy as np
import tensorflow as tf
from typing import Callable

from sopt.optimizers.tensorflow import BackTrackingLineSearch, AdaptiveLineSearch

__all__ = ['ProjectedGradient']

class ProjectedGradient(object):
    """This is a convenience wrapper around the linesearch"""
    linesearch_map = {'backtracking': BackTrackingLineSearch,
                      'adaptive': AdaptiveLineSearch}

    def __init__(self,
                 name: str,
                 input_var: tf.Variable,  # has to have constraint
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 diag_precond_fn: Callable[[], tf.Tensor] = None,
                 max_linesearch_iter: int = None,
                 linesearch_type: str='adaptive') -> None:
        self._name = name
        self._input_var = input_var
        self._dtype = self._input_var.dtype.base_dtype.name

        self._machine_eps = np.finfo(np.dtype(self._dtype)).eps

        self._loss_fn = loss_fn

        self._grads_t = grads_t
        if grads_t is None:
            self._grads_t = tf.gradients(self._loss_t, self._input_var)[0]

        self._descent_dir_t = descent_dir_t
        if descent_dir_t is None:
            self._descent_dir_t = -self._grads_t

        self._max_linesearch_iter = max_linesearch_iter
        self._diag_precond_fn = diag_precond_fn


        self._iters = tf.Variable(0, dtype='int32', name='iterations')
        self._linesearch_iters = tf.Variable(0, dtype='int32', name='linesearch_iterations')
        self._linesearch = self.linesearch_map[linesearch_type](maxiter=self._max_linesearch_iter,
                                                                initial_stepsize=1.0,
                                                                dtype=self._dtype)


        self._variables = [self._iters, self._linesearch_iters]
        reset_ops = [v.assign(v.initial_value) for v in self._variables]
        self._reset_op = tf.group([*reset_ops, self._linesearch.reset])
    @property
    def reset(self):
        return self._reset_op

    def _lossAndUpdateFn(self, var, update):
        new_var = var + update
        if var.constraint is not None:
            new_var = var.constraint(new_var)
        new_loss = self._loss_fn(new_var)
        return new_loss, new_var

    def _search(self, objective_grad: tf.Tensor=None, descent_dir: tf.Tensor=None, f0: tf.Tensor=None):

        if objective_grad is None:
            with tf.GradientTape() as gt:
                loss = self._loss_fn(self._input_var)
            objective_grad = gt.gradient(loss, self._input_var)

        if descent_dir is None:
            descent_dir = -objective_grad

        if self._diag_precond_fn is not None:
            descent_dir = self._diag_precond_fn() * descent_dir

        linesearch_state = self._linesearch.search(objective_and_update=self._lossAndUpdateFn,
                                                   x0=self._input_var,
                                                   descent_dir=descent_dir,
                                                   gradient=objective_grad,
                                                   f0=f0)
        self._linesearch_iters.assign_add(linesearch_state.step_count)
        self._iters.assign_add(1)
        return linesearch_state.newx, linesearch_state.newf

    def minimize(self, **kwargs):
        with tf.name_scope(self._name + '_minimize_step'):
            var_new, loss_new =  self._search(**kwargs)
        self._input_var.assign(var_new)
        return loss_new

