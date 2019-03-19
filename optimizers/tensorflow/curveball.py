#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from typing import Callable, Tuple



## This class is under construction. Attempt to chain the optimization step
# and the damping parameter update step into a single step.
class Curveball(object):
    """Adapted from:
    https://github.com/jotaf98/curveball
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
                 damping_threshold_low: float = 1e-7,
                 damping_threshold_high: float = 1e7,
                 alpha_init: float = 1.0, 
                 squared_loss: bool = True) -> None:
        """The alpha should generally just be 1.0 and doesn't change. 
        The beta and rho values are updated at each cycle, so there is no intial value."""
        self._name = name
        self._input_var = input_var
        
        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn
        
        self._predictions_fn_tensor = self._predictions_fn(self._input_var)
        self._loss_fn_tensor = self._loss_fn(self._predictions_fn_tensor)
        
        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_update_factor = damping_update_factor
        self._damping_update_frequency = damping_update_frequency
        self._update_cond_threshold_low = update_cond_threshold_low
        self._update_cond_threshold_high =  update_cond_threshold_high
        self._damping_threshold_low = damping_threshold_low
        self._damping_threshold_high = damping_threshold_high
        self._squared_loss = squared_loss
        self._alpha = alpha_init

        with tf.variable_scope(name):
            self._damping_factor = tf.get_variable("lambda", dtype=tf.float32, 
                                                  initializer=damping_factor)

            # Variable used for momentum-like updates
            self._z = tf.get_variable("z", dtype=tf.float32, 
                                     initializer=tf.zeros_like(self._input_var))

            self._dummy_var = tf.get_variable("dummy", dtype=tf.float32, 
                                             initializer=tf.zeros_like(self._predictions_fn_tensor))

            self._loss_before_update = tf.get_variable("loss_before_update", dtype=tf.float32,
                                                     initializer=0.)
            self._iteration = tf.get_variable("iteration", shape=[], dtype=tf.int64,
                                             initializer=tf.zeros_initializer)
            self._expected_quadratic_change = tf.get_variable("expected_quadratic_change", 
                                                         dtype=tf.float32,
                                                         initializer=0.)

        # Set up the second order calculations
        self._second_order()
    
    def _second_order(self) -> None:
        with tf.name_scope(self._name + '_second_order'):
            self._vjp = tf.gradients(self._predictions_fn_tensor, self._input_var, self._dummy_var,
                                     name='vjp')[0]
            self._jvpz = tf.gradients(self._vjp, self._dummy_var, tf.stop_gradient(self._z),
                                      name='jvpz')[0]

            if self._squared_loss:
                self._hjvpz = self._jvpz
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

            # Jacobian for the loss function wrt its inputs
            self._jloss = tf.gradients(self._loss_fn_tensor, self._predictions_fn_tensor,
                                       name='jloss')[0]

            # J^T H J z
            self._jhjvpz = tf.gradients(self._predictions_fn_tensor, self._input_var, self._hjvpz + self._jloss,
                                        name='jhjvpz')[0]
            
            self._deltaz = self._jhjvpz + self._damping_factor * self._z    
    
    def _param_updates(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        
        with tf.name_scope(self._name + '_param_updates'):              
            
            # This is for the beta and rho updates
            self._jvpdz = tf.gradients(self._vjp, self._dummy_var, tf.stop_gradient(self._deltaz),
                                       name='jvpdz')[0]

            if self._squared_loss:
                self._hjvpdz = self._jvpdz
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

            a11 = a11 + tf.reduce_sum(self._deltaz * self._deltaz) * self._damping_factor
            a12 = a12 + tf.reduce_sum(self._deltaz * self._z) * self._damping_factor
            a22 = a22 + tf.reduce_sum(self._z * self._z) * self._damping_factor

            A = tf.stack([[a11, a12],[a12, a22]])
            b = tf.stack([b1, b2])
    
            # Cannot use vanilla matrix inverse because the matrix is sometimes singular
            #m_b = tf.reshape(tf.matrix_inverse(A)  @ b[:, None], [-1])
            
            # Using this as a substitute for pinv
            # CANNOT USE this either - behaves badly for ill-conditioned matrices :()
            #m_b = tf.reshape(tf.linalg.lstsq(self.A, self.b[:, None], fast=False), [-1])
            
            #m_b = tf.reshape(tf.linalg.lstsq(self.A + tf.diag(tf.random_uniform([2], 0, 1e-2)), 
            #                                 self.b[:, None], fast=False), [-1])
            
            #m_b = tf.reshape(pinv(A) @ b[:, None], [-1])
            #m_b = tf.reshape(tf.linalg.solve(A + tf.eye(2) * 10 * np.finfo('float32').eps, b[:,None]),[-1])
            
            def pinv(a):
                """Adapted from 
                https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/linalg.py
                """
                
                dtype = a.dtype.as_numpy_dtype
                #rcond = np.finfo(dtype).eps * 2 * 10
                s, u, v = tf.svd(a)
                # Ignore singular values close to zero to prevent numerical overflow
                limit = np.finfo(dtype).eps * 10 
                non_zero = tf.greater(s, limit)

                reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(s.shape, dtype='float64'))
                lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
                return tf.matmul(lhs, u, transpose_b=True)
            
            #A_double = tf.cast(A, tf.float64)
            #b_double = tf.cast(b, tf.float64)
            
            #m_b_double = tf.reshape(tf.linalg.solve(A_double + tf.eye(2, dtype='float64') * 100 * np.finfo('float32').eps,
            #                                        b_double[:, None]), [-1])
            
            #m_b_double = tf.reshape(pinv(A_double) @ b_double[:,None], [-1])
            #m_b = tf.cast(m_b_double, tf.float32)
            
            # Among the many things I tried, this seemed to be  the only one that worked somewhat consistently
            m_b = tf.reshape(tf.linalg.solve(A + tf.eye(2) * 100 * np.finfo('float32').eps,
                                             b[:,None]), [-1])
            
            beta = m_b[0]
            rho = -m_b[1]
            M = -0.5 * tf.reduce_sum(m_b * b)
        return beta, rho, M
            
    def _damping_update(self) -> tf.Operation:
        # It turns out that tensorflow can only calculate the value of a tensor *once* during a session.run() call.
        # This means that I cannot calculate the loss value *before* and *after* the variable update within the 
        # same session.run call. Since the damping update reuires both the values, I am separating this out.
        
        # Uses the placeholder "loss_after_update"
        # This might be a TOO COMPLICATED way to do the damping updates.
        
        with tf.name_scope(self._name + '_damping_update'):
            def update() -> tf.Tensor:
                loss_after_update = self._loss_fn(self._predictions_fn(self._input_var))
                actual_loss_change = loss_after_update - self._loss_before_update
                gamma_val = actual_loss_change / self._expected_quadratic_change

                f1 = lambda: tf.constant(1.0 / self._damping_update_factor)
                f2 = lambda: tf.constant(self._damping_update_factor)
                f3 = lambda: tf.constant(1.0)

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
            #damping_factor_new = tf.cond(tf.equal(self._iteration % self._damping_update_frequency, 0),
            #                             update, damping_same)


            #damping_update_op = tf.assign(self._damping_factor, damping_factor_new,
            #                                    name='damping_update_op')
            damping_update_op = tf.cond(tf.equal(self._iteration % self._damping_update_frequency, 0),
                                            damping_new_op, damping_same)
        return damping_update_op
        
    def minimize(self) -> tf.Operation:
        with tf.name_scope(self._name + '_minimize_step'):
            # Update the beta and rho parameters
            beta, rho, M = self._param_updates()
            
            
            quadratic_change_op = tf.assign(self._expected_quadratic_change, M, 
                                            name='quadratic_change_assign_op')
            store_loss_op = tf.assign(self._loss_before_update, self._loss_fn_tensor,
                                             name='store_loss_op')

            """Update the various variables in sequence"""
            with tf.control_dependencies([quadratic_change_op, store_loss_op]):
                z_op = tf.assign(self._z, rho * self._z - beta * 
                                 self._deltaz, name='z_op')
                
            with tf.control_dependencies([z_op]):
                var_update_op = tf.assign(self._input_var, self._input_var + 
                                          self._alpha * self._z, 
                                          name='var_update_op')
            with tf.control_dependencies([var_update_op]):
                damping_update_op = self._damping_update()
            with tf.control_dependencies([damping_update_op]):
                counter_op = tf.assign(self._iteration, self._iteration + 1, name='counter_op')
        return counter_op



## IN CONSTRUCTION
class CurveballOrig(object):
    """Adapted from:
    https://github.com/jotaf98/curveball
    """
    def __init__(self, 
                 input_var: tf.Variable, 
                 loss_fn_input: tf.Tensor, 
                 loss_fn_tensor: tf.Tensor, 
                 name: str,
                 damping_factor: float = 1.0, 
                 damping_update_factor: float = 0.999, 
                 damping_update_frequency: int = 5,
                 alpha_init: float = 1.0, 
                 squared_loss: bool = True) -> None:
        """The alpha should generally just be 1.0 and doesn't change. 
        The beta and rho values are updated at each cycle, so there is no intial value."""
        self._name = name
        self._input_var = input_var
        self._loss_fn_input = loss_fn_input
        self._loss_fn_tensor = loss_fn_tensor
        
        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_update_factor = damping_update_factor
        self._damping_update_frequency = damping_update_frequency
        self._squared_loss = squared_loss
        self._alpha = alpha_init

        with tf.variable_scope(name):
            self._damping_factor = tf.get_variable("lambda", dtype=tf.float32, 
                                                  initializer=damping_factor)
            
            # Variable used for momentum-like updates
            self._z = tf.get_variable("z", dtype=tf.float32, 
                                     initializer=tf.zeros_like(self._input_var))#(tf.shape(self._input_var)))

            self._dummy_var = tf.get_variable("dummy", dtype=tf.float32, 
                                             initializer=tf.zeros_like(self._loss_fn_input))

            self._loss_before_update = tf.get_variable("loss_before_update", dtype=tf.float32,
                                                     initializer=0.,
                                                     trainable=False)
            self._iteration = tf.get_variable("iteration", shape=[], dtype=tf.int64,
                                             initializer=tf.zeros_initializer, 
                                             trainable=False)
            self._expected_quadratic_change = tf.get_variable("expected_quadratic_change", 
                                                         dtype=tf.float32,
                                                         initializer=0.,
                                                         trainable=False)

        # Update parameters

        self.loss_after_update_placeholder = tf.placeholder(tf.float32, shape=[])
        self._second_order()

        # Update the beta and rho parameters
        self._param_updates()
        
    
    def _second_order(self) -> None:
        with tf.name_scope(self._name + '_second_order'):
            self._vjp = tf.gradients(self._loss_fn_input, self._input_var, self._dummy_var, 
                                     name='vjp')[0]
            self._jvpz = tf.gradients(self._vjp, self._dummy_var, self._z,
                                      name='jvpz')[0]

            # this could be off by a factor of two
            if self._squared_loss:
                self._hjvpz = self._jvpz
            else:
                self._hjvpz = tf.gradients(tf.gradients(self._loss_fn_tensor, 
                                                       self._loss_fn_input)[0][None, :] 
                                          @ self._jvpz[:,None], self._loss_fn_input,
                                          stop_gradients=self._jvpz)[0]

            # Jacobian for the loss function wrt its inputs
            self._jloss = tf.gradients(self._loss_fn_tensor, self._loss_fn_input, name='jloss')[0]

            # J^T H J z
            self._jhjvpz = tf.gradients(self._loss_fn_input, self._input_var, self._hjvpz + self._jloss, 
                                        name='jhjvpz')[0]
            self._deltaz = self._jhjvpz + self._damping_factor * self._z    
    
    def _param_updates(self) -> None:
        
        with tf.name_scope(self._name + '_param_updates'):              
            
            # This is for the beta and rho updates
            self._jvpdz = tf.gradients(self._vjp, self._dummy_var, self._deltaz, name='jvpdz')[0]

            if self._squared_loss:
                self._hjvpdz = self._jvpdz
            else:
                self._hjvpdz = tf.gradients(tf.gradients(self._loss_fn_tensor, 
                                                       self._loss_fn_input)[0][None, :] 
                                          @ self._jvpdz[:,None], self._loss_fn_input,
                                          stop_gradients=self._jvpdz)[0]

            a11 = tf.reduce_sum(self._hjvpdz * self._jvpdz)
            a12 = tf.reduce_sum(self._jvpz * self._hjvpdz)
            a22 = tf.reduce_sum(self._jvpz * self._hjvpz)

            b1 = tf.reduce_sum(self._jloss * self._jvpdz)
            b2 = tf.reduce_sum(self._jloss * self._jvpz)

            a11 = a11 + tf.reduce_sum(self._deltaz * self._deltaz) * self._damping_factor
            a12 = a12 + tf.reduce_sum(self._deltaz * self._z) * self._damping_factor
            a22 = a22 + tf.reduce_sum(self._z * self._z) * self._damping_factor

            A = tf.stack([[a11, a12],[a12, a22]])
            b = tf.stack([b1, b2])
            
            # Cannot use vanilla matrix inverse because the matrix is sometimes singular
            #m_b = tf.reshape(tf.matrix_inverse(A)  @ b[:, None], [-1])
            
            # Using this as a substitute for pinv
            # CANNOT USE this either - behaves badly for ill-conditioned matrices :()
            #m_b = tf.reshape(tf.linalg.lstsq(self.A, self.b[:, None], fast=False), [-1])
            
            #m_b = tf.reshape(tf.linalg.lstsq(self.A + tf.diag(tf.random_uniform([2], 0, 1e-2)), 
            #                                 self.b[:, None], fast=False), [-1])
            
            #m_b = tf.reshape(pinv(A) @ b[:, None], [-1])
            #m_b = tf.reshape(tf.linalg.solve(A + tf.eye(2) * 10 * np.finfo('float32').eps, b[:,None]),[-1])
            
            def pinv(a):
                """Adapted from 
                https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/linalg.py
                """
                
                dtype = a.dtype.as_numpy_dtype
                rcond = np.finfo('float32').eps * 2 * 10
                s, u, v = tf.svd(a)
                # Ignore singular values close to zero to prevent numerical overflow
                limit = np.finfo(dtype).eps * 10 #rcond * tf.reduce_max(s) * 10.
                non_zero = tf.greater(s, limit)

                reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(s.shape))
                lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
                return tf.matmul(lhs, u, transpose_b=True)
            
            # Another approach
            m_b = tf.reshape(pinv(A) @ b[:,None], [-1])
                        
            self._beta = m_b[0]
            self._rho = -m_b[1]
            self._M = -0.5 * tf.reduce_sum(m_b * b)
            
    def damping_update(self, 
                       update_threshold_low: float = 0.5,
                       update_threshold_high: float = 1.5,
                       damping_threshold_low: float = 1e-7,
                       damping_threshold_high: float = 1e7) -> tf.Operation:
        # It turns out that tensorflow can only calculate the value of a tensor *once* during a session.run() call.
        # This means that I cannot calculate the loss value *before* and *after* the variable update within the 
        # same session.run call. Since the damping update reuires both the values, I am separating this out.
        
        # Uses the placeholder "loss_after_update"
        # This might be a TOO COMPLICATED way to do the damping updates.
        
        with tf.name_scope(self._name + '_damping_update'):
            
            def update() -> tf.Tensor:
                actual_loss_change = self.loss_after_update_placeholder - self._loss_before_update
                gamma_val = actual_loss_change / self._expected_quadratic_change

                f1 = lambda: tf.constant(1.0 / self._damping_update_factor)
                f2 = lambda: tf.constant(self._damping_update_factor)
                f3 = lambda: tf.constant(1.0)

                update_factor = tf.case({tf.less(gamma_val, update_threshold_low):f1, 
                                 tf.greater(gamma_val, update_threshold_high):f2},
                                 default=f3, exclusive=True)

                damping_factor_new = tf.clip_by_value(self._damping_factor 
                                                      * update_factor, 
                                                      damping_threshold_low, 
                                                      damping_threshold_high)
                return damping_factor_new

            damping_new_op = lambda: tf.assign(self._damping_factor, update(), name='damping_new_op')
            damping_same = lambda: tf.identity(self._damping_factor)
            #damping_factor_new = tf.cond(tf.equal(self._iteration % self._damping_update_frequency, 0),
            #                             update, damping_same)


            #damping_update_op = tf.assign(self._damping_factor, damping_factor_new,
            #                                    name='damping_update_op')
            damping_update_op = tf.cond(tf.equal(self._iteration % self._damping_update_frequency, 0),
                                        damping_new_op, damping_same)
        return damping_update_op
        
    def minimize(self) -> tf.Operation:
        with tf.name_scope(self._name + '_minimize_step'):
            counter_op = tf.assign(self._iteration, self._iteration + 1, name='counter_op')
            quadratic_change_op = tf.assign(self._expected_quadratic_change, self._M, 
                                           name='quadratic_change_assign_op')
            store_loss_op = tf.assign(self._loss_before_update, self._loss_fn_tensor,
                                      name='store_loss_op')

            """Update the various variables in sequence"""
            with tf.control_dependencies([quadratic_change_op, counter_op, store_loss_op]):
                z_op = tf.assign(self._z, self._rho * self._z - self._beta * 
                                 self._deltaz, name='z_op')
                
            with tf.control_dependencies([z_op]):
                var_update_op = tf.assign(self._input_var, self._input_var + 
                                          self._alpha * self._z, 
                                          name='var_update_op')
        return var_update_op

    def minimize_2(self) -> tf.Operation:
        with tf.name_scope(self._name + '_minimize_step'):
            counter_op = tf.assign(self._iteration, self._iteration + 1, name='counter_op')
            quadratic_change_op = tf.assign(self._expected_quadratic_change, self._M, 
                                           name='quadratic_change_assign_op')
            store_loss_op = tf.assign(self._loss_before_update, self._loss_fn_tensor,
                                      name='store_loss_op')

            """Update the various variables in sequence"""
            with tf.control_dependencies([quadratic_change_op, counter_op, store_loss_op]):
                z_op = tf.assign(self._z, self._rho * self._z - self._beta * 
                                 self._deltaz, name='z_op')
                
            with tf.control_dependencies([z_op]):
                var_update_op = tf.assign(self._input_var, self._input_var + 
                                          self._alpha * self._z, 
                                          name='var_update_op')
        return var_update_op

