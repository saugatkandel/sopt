import numpy as np
import tensorflow as tf
from typing import Callable, Tuple, Union
import sopt.optimizers.tensorflow2.utils.autodiff_helper as adh

__all__ = ['Curveball']


class Curveball(object):
    """Adapted from:
    https://github.com/jotaf98/curveball

    The alpha should generally just be 1.0 and doesn't change.
    The beta and rho values are updated at each cycle, so there is no intial value.

    SK (06/13/20): Adding diagonal scaling does not help, but actually hinders the convergence.
                    Preconditioning helps for the full batch case, but not for the minibatch case?

    """

    def __init__(self,
                 input_var: tf.Variable,
                 predictions_fn: Callable[[tf.Tensor], tf.Tensor],
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 name: str,
                 damping_factor: float = 1.0,
                 damping_update_factor: float = 0.999,
                 damping_update_frequency: int = 5,
                 update_cond_threshold_low: float = 0.5,
                 update_cond_threshold_high: float = 1.5,
                 damping_threshold_low: float = 1e-8,
                 damping_threshold_high: float = 1e8,
                 alpha_init: float = 1.0,
                 diag_hessian_fn: Callable[[tf.Tensor], tf.Tensor] = None,
                 new_version: bool = True) -> None:
        """New version of the matrix-vector-product code is faster."""
        self._new_version = new_version
        self._name = name
        if len(input_var.shape) > 1:
            raise ValueError("The optimizer currently only supports a one-dimensional variable array. "
                             + "Reshaping into multidimensional arrays should can be wrapped into predictions_fn.")
        self._input_var = input_var

        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        self._machine_eps = np.finfo(input_var.dtype.as_numpy_dtype).eps

        self._damping_factor = tf.Variable(damping_factor, dtype='float32')

        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_update_factor = damping_update_factor

        self._damping_update_frequency = damping_update_frequency
        self._update_cond_threshold_low = update_cond_threshold_low
        self._update_cond_threshold_high = update_cond_threshold_high
        self._damping_threshold_low = damping_threshold_low
        self._damping_threshold_high = damping_threshold_high

        self._alpha = alpha_init

        self._diag_hessian_fn = diag_hessian_fn

        # Momentum-like updates
        self._z = tf.Variable(tf.zeros_like(self._input_var), trainable=False)

        self._iteration = tf.Variable(0, dtype='int32', trainable=False)
        self._successful_iterations = tf.Variable(0, dtype='int32', trainable=False)
        self._loss_old = tf.Variable(np.inf, dtype='float32', trainable=False)
        self._loss_new = tf.Variable(np.inf, dtype='float32', trainable=False)

        self._variables = [self._damping_factor, self._z,
                           self._iteration, self._successful_iterations,
                           self._loss_old, self._loss_new]

    def reset(self) -> bool:
        for v in self._variables:
            v.assign(v.initial_value)
        return True

    def _hvp(self, prediction: tf.Tensor,
             vector:tf.Tensor,
             return_loss_grad:bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        if self._diag_hessian_fn is not None:
            diag_hessian = self._diag_hessian_fn(prediction)
            hvp = diag_hessian * vector
            if return_loss_grad:
                with tf.GradientTape() as gt:
                    gt.watch(prediction)
                    loss = self._loss_fn(prediction)
                lossgrad = gt.gradient(loss, prediction)
        else:
            loss, lossgrad, hvp = adh.hvp_forward_backward(self._loss_fn, prediction, vector)

        if return_loss_grad:
            return loss, lossgrad, hvp
        return hvp

    def _getStepDifferentialParams_v1(self):
        with tf.GradientTape() as gt:
            prediction, jvpz = adh.jvp_forward(self._predictions_fn, self._input_var, self._z)
            loss_old, lossgrad, hjvpz = self._hvp(prediction, jvpz, return_loss_grad=True)

        jthjvpz = gt.gradient(prediction, self._input_var, output_gradients=(hjvpz + lossgrad))
        deltaz = jthjvpz + self._damping_factor * self._z

        _, jvpdz = adh.jvp_forward(self._predictions_fn, self._input_var, deltaz)
        hjvpdz = self._hvp(prediction, jvpdz, return_loss_grad=False)
        return loss_old, lossgrad, deltaz, jvpz, hjvpz, jvpdz, hjvpdz

    def _getStepDifferentialParams_v2(self):
        with tf.GradientTape(persistent=True) as gt:
            with tf.GradientTape(persistent=True) as gt2:
                prediction = self._predictions_fn(self._input_var)
                gt.watch(prediction)
                loss_old = self._loss_fn(prediction)
            x = tf.ones_like(prediction)
            gt.watch(x)
            vjp_fn = lambda v: gt2.gradient(prediction, self._input_var, output_gradients=v)
            inner_vjp = vjp_fn(x)
        jvp_fn_this = lambda v: gt.gradient(inner_vjp, x, output_gradients=v)

        def hvp_aux_fn(v):
            jvp = jvp_fn_this(v)
            hjvp = self._hvp(prediction, jvp, return_loss_grad=False)
            return jvp, hjvp

        lossgrad = gt.gradient(loss_old, prediction)
        jvpz, hjvpz = hvp_aux_fn(self._z)
        jthjvpz = vjp_fn(hjvpz + lossgrad)
        deltaz = jthjvpz + self._damping_factor * self._z

        jvpdz, hjvpdz = hvp_aux_fn(deltaz)
        return loss_old, lossgrad, deltaz, jvpz, hjvpz, jvpdz, hjvpdz



    def _param_updates(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """There might be efficiencies to exploit in the repeat jvp calculations, but it doesn't seem worth the effort."""
        with tf.name_scope(self._name + '_param_updates'):

            if not self._new_version:
                #--This works fine but we want a more efficient version------------------------------------------------
                loss_old, lossgrad, deltaz, jvpz, hjvpz, jvpdz, hjvpdz = self._getStepDifferentialParams_v1()
                # ------------------------------------------------------------------------------------------------------
            else:
                loss_old, lossgrad, deltaz, jvpz, hjvpz, jvpdz, hjvpdz = self._getStepDifferentialParams_v2()

            a11 = tf.reduce_sum(hjvpdz * jvpdz)
            a12 = tf.reduce_sum(jvpz * hjvpdz)
            a22 = tf.reduce_sum(jvpz * hjvpz)

            b1 = tf.reduce_sum(lossgrad * jvpdz)
            b2 = tf.reduce_sum(lossgrad * jvpz)

            a11 = a11 + tf.reduce_sum(deltaz * deltaz * self._damping_factor)
            a12 = a12 + tf.reduce_sum(deltaz * self._z * self._damping_factor)
            a22 = a22 + tf.reduce_sum(self._z * self._z * self._damping_factor)

            A = tf.stack([[a11, a12], [a12, a22]])
            b = tf.stack([b1, b2])

            A_inv = tf.linalg.pinv(A, rcond=1e-15)
            m_b = tf.cond(tf.equal(b2, 0.),
                          lambda: tf.stack([b[0] / A[[0, 0]], 0.]),
                          lambda: tf.reshape(A_inv @ b[:, None], [-1]))

            beta = m_b[0]
            rho = -m_b[1]
            M = -0.5 * tf.reduce_sum(m_b * b)

        return loss_old, beta, rho, M, deltaz

    def _damping_update(self, loss_change_before_constraint: tf.Tensor,
                        expected_quadratic_change: tf.Tensor) -> tf.Tensor:
        if self._iteration % self._damping_update_frequency == 0:
            with tf.name_scope(self._name + '_damping_update'):
                gamma_val = loss_change_before_constraint / expected_quadratic_change

                if gamma_val < self._update_cond_threshold_low:
                    update_factor = 1 / self._damping_update_factor
                elif gamma_val > self._update_cond_threshold_high:
                    update_factor = self._damping_update_factor
                else:
                    update_factor = 1.0

                damping_factor = tf.clip_by_value(self._damping_factor * update_factor,
                                                  self._damping_threshold_low,
                                                  self._damping_threshold_high)
                return damping_factor
        return self._damping_factor

    @tf.function
    def minimize(self) -> tf.Tensor:
        with tf.name_scope(self._name + '_minimize_step'):
            loss_before_update, beta, rho, M, deltaz = self._param_updates()

            z_new = rho * self._z - beta * deltaz
            var_new = self._input_var + self._alpha * z_new
            loss_after_update = self._loss_fn(self._predictions_fn(var_new))

            loss_change_before_constraint = loss_after_update - loss_before_update

            update_condition = (loss_change_before_constraint < self._machine_eps * 10)

            if update_condition:
                self._z.assign(z_new)
                if self._input_var.constraint is not None:
                    var_new = self._input_var.constraint(var_new)
                    loss_after_update = self._loss_fn(self._predictions_fn(var_new))
                self._input_var.assign(var_new, name='var_update_op')
                self._successful_iterations.assign_add(1)
            else:
                self._z.assign(tf.zeros_like(self._z))

            self._damping_factor.assign(self._damping_update(loss_change_before_constraint, M))
            self._iteration.assign_add(1)
            self._loss_old.assign(loss_before_update)
            self._loss_new.assign(loss_after_update)
        return loss_after_update