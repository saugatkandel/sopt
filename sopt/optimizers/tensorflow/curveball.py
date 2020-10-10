#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from typing import Callable, Tuple
from sopt.optimizers.tensorflow import AdaptiveLineSearch


__all__ = ['Curveball', 'PreconditionedCurveball']

## This class is under construction. Attempt to chain the optimization step
# and the damping parameter update step into a single step.
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
                 diag_hessian_fn: Callable[[tf.Tensor], tf.Tensor]= None) -> None:


        self._name = name
        self._input_var = input_var
        
        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        self._machine_eps = np.finfo(input_var.dtype.as_numpy_dtype).eps
        
        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_update_factor = damping_update_factor
        self._damping_update_frequency = damping_update_frequency
        self._update_cond_threshold_low = update_cond_threshold_low
        self._update_cond_threshold_high =  update_cond_threshold_high
        self._damping_threshold_low = damping_threshold_low
        self._damping_threshold_high = damping_threshold_high
        self._alpha = alpha_init

        self._diag_hessian_fn = diag_hessian_fn
        
        with tf.variable_scope(name):
            self._predictions_fn_tensor = self._predictions_fn(self._input_var)
            self._loss_fn_tensor = self._loss_fn(self._predictions_fn_tensor)
            
            # Jacobian for the loss function wrt its inputs
            self._jloss = tf.gradients(self._loss_fn_tensor, self._predictions_fn_tensor,
                                       name='jloss')[0]
            if self._diag_hessian_fn is not None:
                self._diag_hessian_fn_tensor = self._diag_hessian_fn(self._predictions_fn_tensor)
            
            self._damping_factor = tf.get_variable("lambda", dtype=tf.float32, 
                                                  initializer=damping_factor)

            # Variable used for momentum-like updates
            self._z = tf.get_variable("z", dtype=tf.float32, 
                                     initializer=tf.zeros_like(self._input_var))

            self._dummy_var = tf.get_variable("dummy", dtype=tf.float32, 
                                             initializer=tf.zeros_like(self._predictions_fn_tensor))

            self._loss_before_update = tf.get_variable("loss_before_update", dtype=tf.float32,
                                                     initializer=0.)
            self._iteration = tf.get_variable("iteration", shape=[], dtype=tf.int32,
                                             initializer=tf.zeros_initializer)
            self._successful_iterations = tf.get_variable("successful_iters", shape=[], dtype=tf.int32,
                                             initializer=tf.zeros_initializer)

        # Set up the second order calculations
        self._second_order()
    
    def _second_order(self) -> None:
        with tf.name_scope(self._name + '_second_order'):
            self._vjp = tf.gradients(self._predictions_fn_tensor, self._input_var, self._dummy_var,
                                     name='vjp')[0]
            self._jvpz = tf.gradients(self._vjp, self._dummy_var, tf.stop_gradient(self._z),
                                      name='jvpz')[0]

            if self._diag_hessian_fn is not None:
                self._hjvpz = self._diag_hessian_fn_tensor * self._jvpz
            else:
                # I have commented out my implementation of the hessian-vector product. 
                # Using the tensorflow implementation instead.
                #self._hjvpz = tf.gradients(tf.gradients(self._loss_fn_tensor, 
                #                                       self._predictions_fn_tensor)[0][None, :] 
                #                          @ self._jvpz[:,None], self._predictions_fn_tensor,
                #                          stop_gradients=self._jvpz)[0]
                self._hjvpz = _hessian_vector_product(ys=[self._loss_fn_tensor],
                                                      xs=[self._predictions_fn_tensor],
                                                      v=[self._jvpz])[0]

            # J^T H J z
            self._jhjvpz = tf.gradients(self._predictions_fn_tensor, self._input_var,
                                        self._hjvpz + self._jloss,
                                        name='jhjvpz')[0]

            self._deltaz = self._jhjvpz + self._damping_factor * self._z

            self._grad_t = tf.gradients(self._loss_fn_tensor, self._input_var, name='grad')[0]

    def _param_updates(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        
        with tf.name_scope(self._name + '_param_updates'):              
            
            # This is for the beta and rho updates
            self._jvpdz = tf.gradients(self._vjp, self._dummy_var, tf.stop_gradient(self._deltaz),
                                       name='jvpdz')[0]
            
            if self._diag_hessian_fn is not None:
                #self._hjvpdz = self._diag_hessian_fn(self._predictions_fn_tensor) * self._jvpdz
                self._hjvpdz = self._diag_hessian_fn_tensor * self._jvpdz
            else:
                #self._hjvpdz = tf.gradients(tf.gradients(self._loss_fn_tensor, 
                #                                       self._predictions_fn_tensor)[0][None, :] 
                #                          @ self._jvpdz[:,None], self._predictions_fn_tensor,
                #                          stop_gradients=self._jvpdz)[0]
                self._hjvpdz = _hessian_vector_product(ys=[self._loss_fn_tensor],
                                                       xs=[self._predictions_fn_tensor],
                                                       v=[self._jvpdz])[0]

            a11 = tf.reduce_sum(self._hjvpdz * self._jvpdz)
            a12 = tf.reduce_sum(self._jvpz * self._hjvpdz)
            a22 = tf.reduce_sum(self._jvpz * self._hjvpz)

            b1 = tf.reduce_sum(self._jloss * self._jvpdz)
            b2 = tf.reduce_sum(self._jloss * self._jvpz)

            a11 = a11 + tf.reduce_sum(self._deltaz * self._deltaz * self._damping_factor)
            a12 = a12 + tf.reduce_sum(self._deltaz * self._z * self._damping_factor)
            a22 = a22 + tf.reduce_sum(self._z * self._z * self._damping_factor)

            A = tf.stack([[a11, a12],[a12, a22]])
            b = tf.stack([b1, b2])
            
            # Cannot use vanilla matrix inverse because the matrix is sometimes singular
            #m_b = tf.reshape(tf.matrix_inverse(A)  @ b[:, None], [-1])

            # I am using 1e-15 for rcond instead of the default value.
            # While this is a less robust choice, using a higher value of rcond seems to output approximate
            # inverse values which slow down the optimization significantly.
            # Instead, choosing a low value sometimes produces very bad outputs, but we can take care of that
            # using an additional update condition based on the change of the loss function,
            # by requiring that the loss function always decrease.

            def _two_by_two_pinv_sol():
                A_inv = tf.linalg.pinv(A, rcond=1e-15)
                m_b = tf.reshape(A_inv @ b[:, None], [-1])
                #with tf.control_dependencies([tf.print(m_b)]):
                #    m_b_0 = tf.clip_by_value(m_b[0], clip_value_min=1e-5, clip_value_max=1.0)
                #    m_b_1 = tf.clip_by_value(m_b[1], clip_value_min=-np.inf, clip_value_max=-1e-5)
                #    m_b = tf.stack([m_b_0, m_b_1])
                #m_b = tf.reshape(m_b, [-1])
                #m_b = tf.reshape(tf.linalg.lstsq(A, b[:,None], fast=False), [-1])
                #for i in range(10):
                #    db = A @ m_b[:,None] - b[:,None]
                #    m_db = tf.reshape(tf.linalg.lstsq(A, db, fast=False), [-1])
                #    m_b = m_b - m_db
                return m_b

            def _zero_z_sol():
                return tf.stack([b[0] / A[[0,0]], 0.])

            m_b = tf.cond(tf.equal(b2, 0.), _zero_z_sol, _two_by_two_pinv_sol)
            beta = m_b[0]
            rho = -m_b[1]
            M = -0.5 * tf.reduce_sum(m_b * b)
            #with tf.control_dependencies([tf.print(M)]):
            #    M = M + 0.
        return beta, rho, M
            
    def _damping_update(self, loss_change_before_constraint, expected_quadratic_change) -> tf.Operation:
        # It turns out that tensorflow can only calculate the value of a tensor *once* during a session.run() call.
        # This means that I cannot calculate the loss value *before* and *after* the variable update within the 
        # same session.run call. Since the damping update reuires both the values, I am separating this out.
        
        # Uses the placeholder "loss_after_update"
        # This might be a TOO COMPLICATED way to do the damping updates.
        with tf.name_scope(self._name + '_damping_update'):
            
            def update() -> tf.Tensor:
                #loss_after_update = self._loss_fn(self._predictions_fn(self._input_var))
                #actual_loss_change = loss_after_update - self._loss_before_update

                #gamma_val = actual_loss_change / self._expected_quadratic_change
                gamma_val = loss_change_before_constraint / expected_quadratic_change

                f1 = lambda: tf.constant(1.0 / self._damping_update_factor)
                f2 = lambda: tf.constant(self._damping_update_factor)
                f3 = lambda: tf.constant(1.0)
                #with tf.control_dependencies([tf.print(actual_loss_change, gamma_val)]):
                update_factor = tf.case({tf.less(gamma_val, self._update_cond_threshold_low):f1,
                                 tf.greater(gamma_val, self._update_cond_threshold_high):f2},
                                 default=f3, exclusive=True)

                damping_factor_new = tf.clip_by_value(self._damping_factor 
                                                      * update_factor, 
                                                      self._damping_threshold_low, 
                                                      self._damping_threshold_high)
                return damping_factor_new

            damping_new_op = lambda: tf.assign(self._damping_factor, update(), name='damping_new_op')
            damping_same = lambda: tf.identity(self._damping_factor)

            damping_update_op = tf.cond(tf.equal(self._iteration % self._damping_update_frequency, 0),
                                            damping_new_op, damping_same)
        return damping_update_op
        
    def minimize(self) -> tf.Operation:
        with tf.name_scope(self._name + '_minimize_step'):
            store_loss_op = self._loss_before_update.assign(self._loss_fn_tensor, name='store_loss_op')

            with tf.control_dependencies([store_loss_op]):
                # Update the beta and rho parameters
                beta, rho, M = self._param_updates()

            z_new = rho * self._z - beta * self._deltaz
            var_new = self._input_var + self._alpha * z_new
            loss_after_update = self._loss_fn(self._predictions_fn(var_new))
            with tf.control_dependencies([store_loss_op]):
                loss_change_before_constraint = loss_after_update - self._loss_before_update

            update_condition = (loss_change_before_constraint < self._machine_eps * 10)

            #with tf.control_dependencies([store_loss_op]):
            #    print_op = tf.print(self._iteration, 'loss', self._loss_before_update,
            #                        'change', loss_change_before_constraint,
            #                        'beta', beta,
            #                        'rho', rho,
            #                        'M', M,
            #                        'ratio', self._ratio)

            """Update the various variables in sequence"""
            with tf.control_dependencies([store_loss_op]):
                z_new = tf.cond(update_condition, lambda: z_new, lambda: tf.zeros_like(self._z))
                z_op = self._z.assign(z_new, name='z_op')
                #z_op = tf.assign(self._z, rho * self._z - beta *
                #                 self._deltaz, name='z_op')
                
            with tf.control_dependencies([z_op]):
                var_new = tf.cond(update_condition, lambda: var_new, lambda: self._input_var)

                if self._input_var.constraint is not None:
                    var_new = self._input_var.constraint(var_new)
                var_update_op = self._input_var.assign(var_new, name='var_update_op')
                
            with tf.control_dependencies([var_update_op]):
                damping_update_ops = [self._damping_update(loss_change_before_constraint, M)]

            with tf.control_dependencies(damping_update_ops):
                success_iter_new = tf.cond(update_condition, lambda: 1, lambda: 0)
                counter_op = tf.group([self._iteration.assign_add(1, name='counter_op'),
                                       self._successful_iterations.assign_add(success_iter_new)])

        return counter_op


class PreconditionedCurveball(object):
    """Modified Curveball method that includes a diagonal preconditioning.
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
                 # These two params are experimental only ############################################################
                 # Removing the scaling parameter. The scaling actually hinders the convergence.
                 diag_precond_t: tf.Tensor = None,  # Use preconditioning for CG steps
                 ####################################################################################################
                 ) -> None:
        """The alpha should generally just be 1.0 and doesn't change.
        The beta and rho values are updated at each cycle, so there is no intial value."""
        self._name = name
        self._input_var = input_var

        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        self._machine_eps = np.finfo(input_var.dtype.as_numpy_dtype).eps

        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_update_factor = damping_update_factor
        self._damping_update_frequency = damping_update_frequency
        self._update_cond_threshold_low = update_cond_threshold_low
        self._update_cond_threshold_high = update_cond_threshold_high
        self._damping_threshold_low = damping_threshold_low
        self._damping_threshold_high = damping_threshold_high
        self._alpha = alpha_init

        self._diag_hessian_fn = diag_hessian_fn
        self._diag_precond_t = diag_precond_t

        with tf.variable_scope(name):
            self._predictions_fn_tensor = self._predictions_fn(self._input_var)
            self._loss_fn_tensor = self._loss_fn(self._predictions_fn_tensor)

            # Jacobian for the loss function wrt its inputs
            self._jloss = tf.gradients(self._loss_fn_tensor, self._predictions_fn_tensor,
                                       name='jloss')[0]
            if self._diag_hessian_fn is not None:
                self._diag_hessian_fn_tensor = self._diag_hessian_fn(self._predictions_fn_tensor)

            self._damping_factor = tf.get_variable("lambda", dtype=tf.float32,
                                                   initializer=damping_factor)

            # Variable used for momentum-like updates
            self._z = tf.get_variable("z", dtype=tf.float32,
                                      initializer=tf.zeros_like(self._input_var))

            self._dummy_var = tf.get_variable("dummy", dtype=tf.float32,
                                              initializer=tf.zeros_like(self._predictions_fn_tensor))

            self._loss_before_update = tf.get_variable("loss_before_update", dtype=tf.float32,
                                                       initializer=0.)
            self._iteration = tf.get_variable("iteration", shape=[], dtype=tf.int32,
                                              initializer=tf.zeros_initializer)
            self._successful_iterations = tf.get_variable("successful_iters", shape=[], dtype=tf.int32,
                                                          initializer=tf.zeros_initializer)

            self._projected_gradient_linesearch = AdaptiveLineSearch(name='proj_ls_linesearch')
            self._projected_gradient_iterations = tf.get_variable("projected_grad_iterations",
                                                                  dtype=tf.int32, shape=[],
                                                                  initializer=tf.zeros_initializer,
                                                                  trainable=False)
            self._total_proj_ls_iterations = tf.get_variable("total_projection_line_search_iterations",
                                                             dtype=tf.int32, shape=[],
                                                             initializer=tf.zeros_initializer,
                                                             trainable=False)

        # Set up the second order calculations
        self._second_order()

    def _second_order(self) -> None:
        with tf.name_scope(self._name + '_second_order'):
            self._vjp = tf.gradients(self._predictions_fn_tensor, self._input_var, self._dummy_var,
                                     name='vjp')[0]
            self._jvpz = tf.gradients(self._vjp, self._dummy_var, tf.stop_gradient(self._z),
                                      name='jvpz')[0]

            if self._diag_hessian_fn is not None:
                self._hjvpz = self._diag_hessian_fn_tensor * self._jvpz
            else:
                self._hjvpz = _hessian_vector_product(ys=[self._loss_fn_tensor],
                                                      xs=[self._predictions_fn_tensor],
                                                      v=[self._jvpz])[0]

            # J^T H J z
            self._jhjvpz = tf.gradients(self._predictions_fn_tensor, self._input_var,
                                        self._hjvpz + self._jloss,
                                        name='jhjvpz')[0]


            self._precond_this_iter = 1.
            if self._diag_precond_t is not None:
                self._precond_this_iter = 1 / (self._diag_precond_t + self._damping_factor)
            self._deltaz = self._precond_this_iter * (self._jhjvpz + self._damping_factor * self._z)
            self._grads_tensor = tf.gradients(self._loss_fn_tensor, self._input_var)[0]

    def _param_updates(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        with tf.name_scope(self._name + '_param_updates'):

            # This is for the beta and rho updates
            ## I think the preconditioning cancels out during the matrix multiplication.
            ## But I am keeping this here in case it helps stabilize the matrix inverse.
            self._jvpdz = tf.gradients(self._vjp, self._dummy_var, tf.stop_gradient(self._deltaz),
                                       name='jvpdz')[0]

            if self._diag_hessian_fn is not None:
                self._hjvpdz = self._diag_hessian_fn_tensor * self._jvpdz
            else:
                self._hjvpdz = _hessian_vector_product(ys=[self._loss_fn_tensor],
                                                       xs=[self._predictions_fn_tensor],
                                                       v=[self._jvpdz])[0]

            v1 = tf.reduce_sum(self._diag_hessian_fn_tensor) / tf.reduce_sum(tf.abs(self._diag_hessian_fn_tensor))

            a110 = tf.reduce_sum(self._hjvpdz * self._jvpdz)
            a12 = tf.reduce_sum(self._jvpz * self._hjvpdz)
            a22 = tf.reduce_sum(self._jvpz * self._hjvpz)

            b1 = tf.reduce_sum(self._jloss * self._jvpdz)
            b2 = tf.reduce_sum(self._jloss * self._jvpz)

            a11 = a110 + tf.reduce_sum(self._deltaz * self._deltaz * self._damping_factor)
            a12 = a12 + tf.reduce_sum(self._deltaz * self._z * self._damping_factor)
            a22 = a22 + tf.reduce_sum(self._z * self._z * self._damping_factor)



            A = tf.stack([[a11, a12], [a12, a22]])
            b = tf.stack([b1, b2])

            # Cannot use vanilla matrix inverse because the matrix is sometimes singular
            # m_b = tf.reshape(tf.matrix_inverse(A)  @ b[:, None], [-1])

            # I am using 1e-15 for rcond instead of the default value.
            # While this is a less robust choice, using a higher value of rcond seems to output approximate
            # inverse values which slow down the optimization significantly.
            # Instead, choosing a low value sometimes produces very bad outputs, but we can take care of that
            # using an additional update condition based on the change of the loss function,
            # by requiring that the loss function always decrease.

            def _two_by_two_pinv_sol():
                A_inv = tf.linalg.pinv(A, rcond=1e-15)
                m_b = tf.reshape(A_inv @ b[:, None], [-1])
                # m_b = tf.reshape(tf.linalg.lstsq(A, b[:,None], fast=False), [-1])
                # for i in range(2):
                #    db = A @ m_b[:,None] - b[:,None]
                #    m_db = tf.reshape(tf.linalg.lstsq(A, db, fast=False), [-1])
                #    m_b = m_b - m_db
                return m_b

            def _zero_z_sol():
                return tf.stack([b[0] / A[[0, 0]], 0.])

            m_b = tf.cond(tf.equal(b2, 0.), _zero_z_sol, _two_by_two_pinv_sol)
            beta = m_b[0]
            rho = -m_b[1]
            M = -0.5 * tf.reduce_sum(m_b * b)
            #dot_prod = tf.reduce_sum(self._grad * self._deltaz / tf.linalg.norm(self._grad) / tf.linalg.norm(self._deltaz))
            #with tf.control_dependencies(
            #    [tf.print('beta', beta, 'rho', rho, 'M', M, 'b2', b2, "b1", b1, "a110", a110, "a11", a11, "v1", v1,
            #              "dot_prod", dot_prod)]):
            #    M = M + 0.

        return beta, rho, M

    def _damping_update(self, loss_change_before_constraint, expected_quadratic_change) -> tf.Operation:
        # It turns out that tensorflow can only calculate the value of a tensor *once* during a session.run() call.
        # This means that I cannot calculate the loss value *before* and *after* the variable update within the
        # same session.run call. Since the damping update reuires both the values, I am separating this out.

        # Uses the placeholder "loss_after_update"
        # This might be a TOO COMPLICATED way to do the damping updates.
        with tf.name_scope(self._name + '_damping_update'):
            def update() -> tf.Tensor:
                # loss_after_update = self._loss_fn(self._predictions_fn(self._input_var))
                # actual_loss_change = loss_after_update - self._loss_before_update

                # gamma_val = actual_loss_change / self._expected_quadratic_change
                gamma_val = loss_change_before_constraint / expected_quadratic_change

                f1 = lambda: tf.constant(1.0 / self._damping_update_factor)
                f2 = lambda: tf.constant(self._damping_update_factor)
                f3 = lambda: tf.constant(1.0)
                # with tf.control_dependencies([tf.print(actual_loss_change, gamma_val)]):
                update_factor = tf.case({tf.less(gamma_val, self._update_cond_threshold_low): f1,
                                         tf.greater(gamma_val, self._update_cond_threshold_high): f2},
                                        default=f3, exclusive=True)

                damping_factor_new = tf.clip_by_value(self._damping_factor
                                                      * update_factor,
                                                      self._damping_threshold_low,
                                                      self._damping_threshold_high)
                return damping_factor_new

            damping_new_op = lambda: tf.assign(self._damping_factor, update(), name='damping_new_op')
            damping_same = lambda: tf.identity(self._damping_factor)

            damping_update_op = tf.cond(tf.equal(self._iteration % self._damping_update_frequency, 0),
                                        damping_new_op, damping_same)
        return damping_update_op

    @staticmethod
    def _applyConstraint(input_var, update):
        return input_var.constraint(input_var + update)

    # def _applyProjectedGradient(self, cb_update):
    #     """
    #     Reference:
    #     Journal of Computational and Applied Mathematics 172 (2004) 375–397
    #     """
    #
    #     projected_var = self._applyConstraint(self._input_var, cb_update)
    #     projected_loss_new = self._loss_fn(self._predictions_fn(projected_var))
    #     projected_loss_change = self._loss_before_update - projected_loss_new
    #     projection_reduction_ratio = projected_loss_change / tf.abs(self._loss_before_update)
    #
    #     fconv = tf.abs(self._loss_before_update) <= (self._machine_eps)
    #
    #     no_projection_condition = ((projection_reduction_ratio > 1e-4) | fconv)
    #
    #
    #     print_op = tf.print('loss_before_LM', self._loss_fn_tensor,
    #     #                    'loss_after_lm', loss_new,
    #     #                    'reduction_ration', reduction_ratio,
    #                         #'loss_after_lm', lmstate.loss,
    #                         'loss_after_constraint', projected_loss_new,
    #                         'projected_loss_diff', projected_loss_change,
    #                         'projection_reduction_ratio', projection_reduction_ratio,
    #                         'projection_condition', no_projection_condition)
    #     #                    'test_rhs', test_rhs, 'dist', dist)
    #
    #
    #     def _loss_and_update_fn(x, y):
    #         update = self._applyConstraint(x, y)
    #         loss = self._loss_fn(self._predictions_fn(update))
    #         return loss, update
    #
    #     def _linesearch():
    #         linesearch_state = self._projected_gradient_linesearch.search(objective_and_update=_loss_and_update_fn,
    #                                                                       x0=self._input_var,
    #                                                                       descent_dir=cb_update, #-self._grads_tensor,
    #                                                                       gradient=self._grads_tensor,
    #                                                                       f0=self._loss_before_update)
    #         counter_ops = [self._total_proj_ls_iterations.assign_add(linesearch_state.step_count),
    #                       self._projected_gradient_iterations.assign_add(1)]
    #         with tf.control_dependencies([*counter_ops, print_op]):
    #             output = tf.identity(linesearch_state.newx)
    #         return output
    #
    #     #with tf.control_dependencies([print_op]):
    #     input_var_update = tf.cond(no_projection_condition,
    #                                    lambda: projected_var,
    #                                    _linesearch)
    #     return input_var_update

    def minimize(self) -> tf.Operation:
        with tf.name_scope(self._name + '_minimize_step'):
            store_loss_op = self._loss_before_update.assign(self._loss_fn_tensor, name='store_loss_op')

            with tf.control_dependencies([store_loss_op]):
                # Update the beta and rho parameters
                beta, rho, M = self._param_updates()

            z_new = rho * self._z - beta * self._deltaz

            def _loss_and_update_fn(x, y):
                update = x + y
                loss = self._loss_fn(self._predictions_fn(update))
                return loss, update

            def _linesearch():
                linesearch_state = self._projected_gradient_linesearch.search(objective_and_update=_loss_and_update_fn,
                                                                              x0=self._input_var,
                                                                              descent_dir=z_new,
                                                                              gradient=self._grads_tensor,
                                                                              f0=self._loss_before_update)
                counter_ops = [self._total_proj_ls_iterations.assign_add(linesearch_state.step_count),
                               self._projected_gradient_iterations.assign_add(1)]
                with tf.control_dependencies([*counter_ops, tf.print("alpha", linesearch_state.alpha)]):
                    output = tf.identity(linesearch_state.newx)
                    return output


            var_new = _linesearch()

            #var_new = self._input_var + self._alpha * z_new
            loss_after_update = self._loss_fn(self._predictions_fn(var_new))
            with tf.control_dependencies([store_loss_op]):
                loss_change_before_constraint = loss_after_update - self._loss_before_update
            update_condition = (loss_change_before_constraint < self._machine_eps * 10)

            # with tf.control_dependencies([store_loss_op]):
            #    print_op = tf.print(self._iteration, 'loss', self._loss_before_update,
            #                        'change', loss_change_before_constraint,
            #                        'beta', beta,
            #                        'rho', rho,
            #                        'M', M,
            #                        'ratio', self._ratio)

            """Update the various variables in sequence"""
            with tf.control_dependencies([store_loss_op]):
                z_new = tf.cond(update_condition, lambda: z_new, lambda: tf.zeros_like(self._z))
                z_op = self._z.assign(z_new, name='z_op')
                # z_op = tf.assign(self._z, rho * self._z - beta *
                #                 self._deltaz, name='z_op')

            with tf.control_dependencies([z_op]):
                var_new = tf.cond(update_condition, lambda: var_new, lambda: self._input_var)

                if self._input_var.constraint is not None:
                    var_new = self._input_var.constraint(var_new)
                    #var_new =  self._applyProjectedGradient(z_new)
                var_update_op = self._input_var.assign(var_new, name='var_update_op')

            with tf.control_dependencies([var_update_op]):
                damping_update_ops = [self._damping_update(loss_change_before_constraint, M)]


            with tf.control_dependencies(damping_update_ops):
                success_iter_new = tf.cond(update_condition, lambda: 1, lambda: 0)
                counter_op = tf.group([self._iteration.assign_add(1, name='counter_op'),
                                       self._successful_iterations.assign_add(success_iter_new)])

        return counter_op
