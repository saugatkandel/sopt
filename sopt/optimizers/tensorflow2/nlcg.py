# Non linear conjugate gradient method
# This is fairly easy to change to add complex data types.
import numpy as np
import tensorflow as tf
from typing import Callable, NamedTuple
from sopt.optimizers.tensorflow2.utils import BackTrackingLineSearch, AdaptiveLineSearch

__all__ = ['NonLinearConjugateGradient']

class NonLinearConjugateGradient(object):
    # only implemented PR method for now
    beta_functions_map = {"PR": "_calculatePRBeta"}
    linesearch_map = {'backtracking': BackTrackingLineSearch,
                      'adaptive': AdaptiveLineSearch}

    def __init__(self,
                 input_var: tf.Variable,
                 objective_fn: Callable[[tf.Tensor], tf.Tensor],
                 name: str,
                 max_backtracking_iter: int = None,
                 beta_type: str = 'PR',
                 linesearch_type: str = 'adaptive',
                 diag_precondition_fn: Callable = None) -> None:
        self._name = name
        if len(input_var.shape) > 1:
            raise ValueError("The optimizer currently only supports a one-dimensional variable array. "
                             + "Reshaping into multidimensional arrays should can be wrapped into objective_fn.")
        self._input_var = input_var
        self._dtype = self._input_var.dtype.base_dtype.name
        self._machine_eps = np.finfo(self._dtype).eps

        self._objective_fn = objective_fn

        self._max_backtracking_iter = max_backtracking_iter
        self._beta_type = beta_type
        if beta_type != 'PR':
            raise ValueError("Only PR method supported for now.")

        if linesearch_type not in self.linesearch_map:
            raise ValueError("Only 'backtracking' and 'adaptive' linesearches supported.")

        self._diag_precondition_fn = diag_precondition_fn

        self._descent_dir_old = tf.Variable(tf.zeros_like(self._input_var), name='descent_dir_old',
                                            trainable=False)

        self._s = tf.Variable(tf.zeros_like(self._input_var), name='s', trainable=False)

        self._steps = tf.Variable(0, dtype='int32', name='steps', trainable=False)
        self._linesearch_steps = tf.Variable(0, dtype='int32', name='ls_steps', trainable=False)
        self._linesearch = self.linesearch_map[linesearch_type](maxiter=self._max_backtracking_iter,
                                                                initial_stepsize=1.0,
                                                                dtype=self._dtype)

        self._loss_old = tf.Variable(np.inf, dtype=self._dtype, trainable=False)
        self._loss_new = tf.Variable(np.inf, dtype=self._dtype, trainable=False)

        self._variables = [self._descent_dir_old, self._s, self._steps, self._linesearch_steps,
                           self._loss_old, self._loss_new]

    def reset(self) -> bool:
        for v in self._variables:
            v.assign(v.initial_value)
        return self._linesearch.reset()

    def _calculatePRBeta(self, descent_dir):

        p = descent_dir
        p_old = self._descent_dir_old
        if self._diag_precondition_fn is not None:
            precond = self._diag_precondition_fn()
            p = precond * p
            p_old = precond * p_old

        beta_num = tf.reduce_sum(p * (descent_dir - self._descent_dir_old))
        beta_denom = tf.reduce_sum(p_old * self._descent_dir_old)
        if self._steps > 0:
            beta = tf.maximum(beta_num / beta_denom, tf.constant(0., dtype=self._dtype))
        else:
            beta = tf.constant(0., dtype=self._dtype)
        return beta

    @staticmethod
    def _applyConstraint(x, y):
        return x.constraint(x + y)

    @tf.function
    def minimize(self):

        def _loss_and_update_fn(x, y):
            if self._input_var.constraint is not None:
                update = self._applyConstraint(x, y)
            else:
                update = x + y
            loss = self._objective_fn(update)
            return loss, update

        def _loss_and_gradient_fn():
            with tf.GradientTape() as gt:
                loss = self._objective_fn(self._input_var)
            return loss, gt.gradient(loss, self._input_var)

        beta_function = getattr(self, self.beta_functions_map[self._beta_type])

        with tf.name_scope(self._name + '_minimize_step'):
            loss, grads = _loss_and_gradient_fn()
            descent_dir = -grads

            beta = beta_function(descent_dir)

            s_new = descent_dir + beta * self._s

            # Ensure that the calculated descent direction actually reduces the objective
            descent_check = tf.reduce_sum(s_new * grads)
            s_new = tf.cond(descent_check < 0, lambda: s_new, lambda: descent_dir)

            linesearch_out = self._linesearch.search(_loss_and_update_fn,
                                                     x0=self._input_var,
                                                     descent_dir=s_new,
                                                     gradient=grads,
                                                     f0=loss)
        self._input_var.assign(linesearch_out.newx)
        self._s.assign(s_new)
        self._descent_dir_old.assign(descent_dir)
        self._steps.assign_add(1)
        self._linesearch_steps.assign_add(linesearch_out.step_count)
        self._loss_old.assign(loss)
        self._loss_new.assign(linesearch_out.newf)
        return self._loss_new




