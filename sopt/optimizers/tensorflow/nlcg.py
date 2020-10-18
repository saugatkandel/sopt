# Non linear conjugate gradient method
# Only float data types supported right now.
# This is fairly easy to change to add complex data types.
import numpy as np
import tensorflow as tf
from typing import Callable

from sopt.optimizers.tensorflow import BackTrackingLineSearch, AdaptiveLineSearch

__all__ = ['NonLinearConjugateGradient']

class NonLinearConjugateGradient(object):
    beta_functions_map = {"PR": "_calculatePRBeta"}
    linesearch_map = {'backtracking': BackTrackingLineSearch,
                      'adaptive': AdaptiveLineSearch}

    def __init__(self,
                 input_var: tf.Variable,
                 predictions_fn: Callable[[tf.Tensor], tf.Tensor],
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 name: str,
                 max_backtracking_iter: int = None,
                 beta_type: str = 'PR',
                 linesearch_type: str='adaptive',
                 diag_precondition_t: tf.Tensor= None,) -> None:
        self._name = name
        self._input_var = input_var
        self._dtype = self._input_var.dtype.base_dtype.name

        self._machine_eps = np.finfo(np.dtype(self._dtype)).eps

        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        self._preds_t = self._predictions_fn(self._input_var)
        self._loss_t = self._loss_fn(self._preds_t)

        self._max_backtracking_iter = max_backtracking_iter
        self._beta_type = beta_type
        if beta_type != 'PR':
            raise ValueError("Only PR method supported for now.")

        if linesearch_type not in self.linesearch_map:
            raise ValueError("Only 'backtracking' and 'adaptive' linesearches supported.")

        self._diag_precondition_t = diag_precondition_t

        with tf.variable_scope(name):
            self._descent_dir_old_t = tf.Variable(tf.zeros_like(self._input_var),
                                                  name='descent_dir_old')
            self._s_t = tf.Variable(tf.zeros_like(self._input_var),
                                    name='s')

            self._steps = tf.Variable(0, dtype='int32', name='steps')
            self._linesearch_steps = tf.Variable(0, dtype='int32',
                                                 name='ls_steps')
            self._linesearch = self.linesearch_map[linesearch_type](maxiter=self._max_backtracking_iter,
                                                                    initial_stepsize=1.0,
                                                                    dtype=self._dtype)

        # Gradient calculation
        self._grads_t = tf.gradients(self._loss_t, self._input_var)[0]
        self._descent_dir_t = -self._grads_t
        self._variables = [self._descent_dir_old_t, self._s_t, self._steps, self._linesearch_steps]
        reset_ops = [v.assign(v.initial_value) for v in self._variables]
        self._reset_op = tf.group([*reset_ops, self._linesearch.reset])


    @property
    def reset(self):
        return self._reset_op

    def _calculatePRBeta(self):

        p = self._descent_dir_t
        p_old = self._descent_dir_old_t
        if self._diag_precondition_t is not None:
            p = self._diag_precondition_t * p
            p_old = self._diag_precondition_t * p_old
        beta_num = tf.reduce_sum(p * (self._descent_dir_t - self._descent_dir_old_t))
        beta_denom = tf.reduce_sum(p_old * self._descent_dir_old_t)
        beta = tf.cond(self._steps > 0, lambda: beta_num / beta_denom, lambda: tf.constant(0., dtype=self._dtype))
        beta = tf.maximum(beta, tf.constant(0., dtype=self._dtype))
        return beta

    @staticmethod
    def _applyConstraint(x, y):
        return x.constraint(x + y)

    def minimize(self):
        beta_function = getattr(self, self.beta_functions_map[self._beta_type])
        with tf.name_scope(self._name + '_minimize_step'):
            beta = beta_function()
            s_new = self._descent_dir_t + beta * self._s_t

            # Ensure that the calculated descent direction actually reduces the objective
            descent_check = tf.reduce_sum(s_new * self._grads_t)
            s_new = tf.cond(descent_check < 0, lambda: s_new, lambda: self._descent_dir_t)

            def _loss_and_update_fn(x, y):
                if self._input_var.constraint is not None:
                    update = self._applyConstraint(x, y)
                else:
                    update = x + y
                loss = self._loss_fn(self._predictions_fn(update))
                return loss, update

            linesearch_out = self._linesearch.search(_loss_and_update_fn,
                                                     x0=self._input_var,
                                                     descent_dir=s_new,
                                                     gradient=self._grads_t,
                                                     f0=self._loss_t)
            var_update = tf.identity(linesearch_out.newx)

            with tf.control_dependencies([var_update]):
                assign_ops = tf.group([self._input_var.assign(var_update),
                                       self._s_t.assign(s_new),
                                       self._descent_dir_old_t.assign(self._descent_dir_t),
                                       self._steps.assign_add(1),
                                       self._linesearch_steps.assign_add(linesearch_out.step_count)])
            return assign_ops




