# Non linear conjugate gradient method
# Only float data types supported right now.
# This is fairly easy to change to add complex data types.
import numpy as np
import tensorflow as tf
from typing import Callable

from sopt.optimizers.tensorflow.utils.linesearch import BackTrackingLineSearch, AdaptiveLineSearch

__all__ = ['ProjectedGradient']

class ProjectedGradient(object):
    """This is a convenience wrapper around the linesearch"""
    linesearch_map = {'backtracking': BackTrackingLineSearch,
                      'adaptive': AdaptiveLineSearch}

    def __init__(self,
                 input_var: tf.Variable,  # has to have constraint
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 name: str,
                 grads_t: tf.Tensor = None,
                 descent_dir_t: tf.Tensor = None,
                 diag_precond_t: tf.Tensor = None,
                 max_linesearch_iter: int = None,
                 linesearch_type: str='adaptive') -> None:
        self._name = name
        self._input_var = input_var
        self._dtype = self._input_var.dtype.base_dtype.name

        self._machine_eps = np.finfo(np.dtype(self._dtype)).eps

        self._loss_fn = loss_fn
        self._loss_t = self._loss_fn(self._input_var)

        self._grads_t = grads_t
        if grads_t is None:
            self._grads_t = tf.gradients(self._loss_t, self._input_var)[0]

        self._descent_dir_t = descent_dir_t
        if descent_dir_t is None:
            self._descent_dir_t = -self._grads_t

        self._max_linesearch_iter = max_linesearch_iter
        self._diag_precond_t = diag_precond_t
        if diag_precond_t is not None:
            self._descent_dir_t *= self._diag_precond_t

        with tf.variable_scope(name):
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

    def _search(self, f0: tf.Tensor=None):
        linesearch_state = self._linesearch.search(objective_and_update=self._lossAndUpdateFn,
                                                   x0=self._input_var,
                                                   descent_dir=self._descent_dir_t,
                                                   gradient=self._grads_t,
                                                   f0=f0)
        counter_ops = [self._linesearch_iters.assign_add(linesearch_state.step_count),
                       self._iters.assign_add(1)]
        with tf.control_dependencies(counter_ops):
            output = tf.identity(linesearch_state.newx)
        return output

    def minimize(self):
        with tf.name_scope(self._name + '_minimize_step'):
            var_new =  self._search()
        return self._input_var.assign(var_new)

