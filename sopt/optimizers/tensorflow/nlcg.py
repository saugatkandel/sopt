# Non linear conjugate gradient method
# Only float32 data types supported right now.
# This is fairly easy to change to add complex data types.
import numpy as np
import tensorflow as tf
from typing import Callable, NamedTuple
from sopt.optimizers.tensorflow.utils import BackTrackingLineSearch

class NonLinearConjugateGradient(object):
    beta_functions_map = {"PR": "_calculatePRBeta"}

    def __init__(self,
                 input_var: tf.Variable,
                 predictions_fn: Callable[[tf.Tensor], tf.Tensor],
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 name: str,
                 max_backtracking_iter: int = None,
                 beta_type: str = 'PR') -> None:
        self._name = name
        self._input_var = input_var
        self._machine_eps = np.finfo(input_var.dtype.as_numpy_dtype).eps

        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        self._preds_t = self._predictions_fn(self._input_var)
        self._loss_t = self._loss_fn(self._preds_t)

        self._max_backtracking_iter = max_backtracking_iter
        self._beta_type = beta_type
        if beta_type != 'PR':
            raise ValueError("Only PR method supported for now.")

        with tf.variable_scope(name):
            self._descent_dir_old = tf.Variable(tf.zeros_like(self._input_var),
                                                name='descent_dir_old')
            self._s = tf.Variable(tf.zeros_like(self._input_var),
                                  name='s')
            self._steps = tf.Variable(0, dtype='int32', name='steps')
            self._linesearch_steps = tf.Variable(0, dtype='int32',
                                                 name='ls_steps')
            self._linesearch = BackTrackingLineSearch(maxiter=self._max_backtracking_iter)

        # Gradient calculation
        self._grads = tf.gradients(self._loss_t, self._input_var)[0]
        self._descent_dir = -self._grads

    def _calculatePRBeta(self):

        beta_num = tf.reduce_sum(self._descent_dir * (self._descent_dir - self._descent_dir_old))
        beta_denom = tf.reduce_sum(self._descent_dir_old * self._descent_dir_old)
        beta = tf.cond(self._steps > 0, lambda: beta_num / beta_denom, lambda: 0.)
        beta = tf.maximum(beta, 0.)
        return beta

    @staticmethod
    def _applyConstraint(x, y):
        return x.constraint(x + y)

    def minimize(self):
        beta_function = getattr(self, self.beta_functions_map[self._beta_type])
        with tf.name_scope(self._name + '_minimize_step'):
            beta = beta_function()
            s_new = self._descent_dir + beta * self._s

            # Ensure that the calculated descent direction actually reduces the objective
            descent_check = tf.reduce_sum(s_new * self._grads)
            s_new = tf.cond(descent_check < 0, lambda: s_new, lambda: self._descent_dir)

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
                                                     gradient=self._grads,
                                                     f0=self._loss_t)
            var_update = tf.identity(linesearch_out.newx)

            with tf.control_dependencies([var_update]):
                assign_ops = tf.group([self._input_var.assign(var_update),
                                       self._s.assign(s_new),
                                       self._descent_dir_old.assign(self._descent_dir),
                                       self._steps.assign_add(1),
                                       self._linesearch_steps.assign_add(linesearch_out.step_count)])
            return assign_ops




