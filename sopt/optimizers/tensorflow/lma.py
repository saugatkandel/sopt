#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from typing import Callable, List
from sopt.optimizers.tensorflow.utils import MatrixFreeLinearOp, conjugate_gradient



class LMA(object):
    def __init__(self, 
                 input_var: tf.Variable, 
                 predictions_fn: Callable[[tf.Tensor], tf.Tensor], 
                 loss_fn: Callable[[tf.Tensor], tf.Tensor], 
                 name: str,
                 damping_factor: float = 1.0, 
                 damping_update_factor: float = 2/3,
                 update_cond_threshold_low: float = 0.25, 
                 update_cond_threshold_high: float = 0.75,
                 damping_threshold_low: float = 1e-7,
                 damping_threshold_high: float = 1e7,
                 max_cg_iter: int = 20,
                 cg_tol: float = 1e-5, 
                 xtol: float = 1e-6,
                 ftol: float = 1e-6,
                 gtol: float = 1e-6,
                 squared_loss: bool = True) -> None:
        
        self._name = name
        self._input_var = input_var
        
        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn
        
        self._predictions_fn_tensor = self._predictions_fn(self._input_var)
        self._loss_fn_tensor = self._loss_fn(self._predictions_fn_tensor)
        
        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_update_factor = damping_update_factor
        self._update_cond_threshold_low = update_cond_threshold_low
        self._update_cond_threshold_high =  update_cond_threshold_high
        self._damping_threshold_low = damping_threshold_low
        self._damping_threshold_high = damping_threshold_high
        self._max_cg_iter = max_cg_iter
        
        self._cg_tol = cg_tol
        self._xtol = xtol
        self._ftol = ftol
        self._gtol = gtol
        
        self._squared_loss = squared_loss
        
        with tf.variable_scope(name):
            self._damping_factor = tf.get_variable("lambda", dtype=tf.float32, 
                                                   initializer=damping_factor,
                                                   trainable=False)
            self._update_var = tf.get_variable("delta", dtype=tf.float32,
                                               initializer=tf.ones_like(self._input_var),
                                               trainable=False)
            self._dummy_var = tf.get_variable("dummy", dtype=tf.float32, 
                                              initializer=tf.zeros_like(self._predictions_fn_tensor),
                                              trainable=False)
            
            self._loss_before_update = tf.get_variable("loss_before_update", dtype=tf.float32,
                                                     initializer=0.,
                                                     trainable=False)
            self._expected_quadratic_change = tf.get_variable("expected_quadratic_change", 
                                                         dtype=tf.float32,
                                                         initializer=0.,
                                                         trainable=False)
            self._iteration = tf.get_variable("iteration", shape=[], dtype=tf.int32,
                                              initializer=tf.zeros_initializer,
                                              trainable=False)
            
            self._total_cg_iterations = tf.get_variable("total_cg_iterations", 
                                                        dtype=tf.int32, shape=[],
                                                        initializer=tf.zeros_initializer,
                                                        trainable=False)
        # Set up the second order calculations to define matrix-free linear ops.
        self._setup_second_order()
    
    def _setup_hessian_vector_product(self, 
                                      jvp_fn: Callable[[tf.Tensor], tf.Tensor],
                                      x: tf.Tensor,
                                      v_constant: tf.Tensor) -> tf.Tensor:
        predictions_this = self._predictions_fn(v_constant)
        loss_this = self._loss_fn(predictions_this)
        hjvp = _hessian_vector_product(ys=[loss_this],
                                       xs=[predictions_this],
                                       v=[jvp_fn(x)])
        jhjvp = tf.gradients(predictions_this, v_constant, hjvp)[0]
        return jhjvp
        
    def _setup_second_order(self) -> None:
        with tf.name_scope(self._name + '_gngvp'):
            vjp = tf.gradients(self._predictions_fn_tensor, self._input_var, self._dummy_var,
                                stop_gradients=[self._dummy_var],
                                name='vjp')[0]
            
            jvp_fn = lambda x: tf.gradients(vjp, self._dummy_var, x, name='jvpz')[0]
            self.vjp = vjp
            self.jvp_fn = jvp_fn
            
            if self._squared_loss:
                hjvp_fn = jvp_fn
                # Ignore the v input
                self._jhjvp_fn = lambda x, v_constant: tf.gradients(self._predictions_fn_tensor, 
                                                     self._input_var,
                                                     hjvp_fn(x))[0]
            else:
                self._jhjvp_fn = lambda x, v_constant: self._setup_hessian_vector_product(jvp_fn, x, 
                                                                                    v_constant)
            
            self._grads = tf.gradients(self._loss_fn_tensor, self._input_var)[0]
            
    
    def minimize(self) -> tf.Operation:
        tf.logging.warning("The ftol, gtol, and xtol conditions are adapted from "
                           + "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html."
                           + "This is a test version, and there is no guarantee that these work as intended.")
        with tf.name_scope(self._name + '_minimize_step'):
            
            grads_norm = tf.norm(self._grads, ord=np.inf)
            xtol_norm = self._xtol * (self._xtol + tf.norm(self._input_var, ord=2))
            update_norm = tf.norm(self._update_var, ord=2)
            
            assert_gtol_op = tf.assert_greater(grads_norm, self._gtol, 
                                               message='Gradient norm lower than tolerance.')
            assert_xtol_op = tf.assert_greater(update_norm, xtol_norm,
                                               message='Damping factor lower than ')
            
            
            with tf.control_dependencies([assert_gtol_op, assert_xtol_op]):
                store_loss_op = tf.assign(self._loss_before_update, self._loss_fn_tensor,
                                          name='store_loss_op')
            jhjvp_fn_l_h = lambda l, h, v_constant: self._jhjvp_fn(h, v_constant) + l * h
            linear_b = -self._grads
                
            def _body(damping, update, reduction_ratio, loss_new, cg_iterations, v_constant):
                linear_ax = MatrixFreeLinearOp(lambda h: jhjvp_fn_l_h(damping, h, v_constant),
                                               tf.TensorShape((self._input_var.shape.dims[0],
                                                               self._input_var.shape.dims[0])))
                cg_solve = conjugate_gradient(operator=linear_ax, 
                                              rhs=linear_b, 
                                              x=tf.zeros_like(self._update_var),#self._update_var,
                                              tol=self._cg_tol,
                                              max_iter=self._max_cg_iter)
                update = tf.identity(cg_solve.x, name='cg_solved')
                expected_quadratic_change = -0.5 * tf.tensordot(update, damping * update + linear_b, 1)
                optimized_var = self._input_var + update
                loss_new = self._loss_fn(self._predictions_fn(optimized_var))
                loss_diff = loss_new - self._loss_before_update
                
                ftol_factor = tf.abs(self._ftol * self._loss_before_update)
                assert_ftol_op = tf.assert_greater(tf.abs(loss_diff), ftol_factor,
                                                   message='Function update norm lower than...')
                #ftol_cond = tf.logical_or(tf.math.greater(tf.abs(loss_new - self._loss_before_update), 
                #                                          ftol_factor),
                #                          tf.math.greater(tf.abs(expected_quadratic_change), ftol_factor))
                # 
                #assert_ftol_op = tf.Assert(ftol_cond, [ftol_factor])
                
                with tf.control_dependencies([assert_ftol_op]):
                    reduction_ratio = loss_diff / expected_quadratic_change
                    #reduction_ratio = (loss_new / self._loss_before_update - 1.) / 
                    #(expected_quadratic_change / self._loss_before_update)
                
                f1 = lambda: tf.constant(1.0 / self._damping_update_factor)
                f2 = lambda: tf.constant(self._damping_update_factor)
                f3 = lambda: tf.constant(1.0)

                update_factor = tf.case({tf.less(reduction_ratio, self._update_cond_threshold_low):f1, 
                                 tf.greater(reduction_ratio, self._update_cond_threshold_high):f2},
                                 default=f3, exclusive=True)
                
                damping_new = damping * update_factor

                damping_new = tf.clip_by_value(damping * update_factor, 
                                               self._damping_threshold_low, 
                                               self._damping_threshold_high)
                return (damping_new, update, reduction_ratio, loss_new, cg_iterations + cg_solve.i, v_constant)
            
            def _cond(damping, update, reduction_ratio, loss_new, cg_iterations, v_constant):
                return tf.math.logical_and(reduction_ratio <= 0, 
                                          self._loss_before_update > 10 * np.finfo('float32').eps)
            
            with tf.control_dependencies([store_loss_op]):
                damping_new, update, reduction_ratio, loss_new, cg_iterations, _ = tf.while_loop(_cond, _body,
                                                                                       (self._damping_factor, 
                                                                                        self._update_var, 0., 0., 
                                                                                        tf.constant(0, dtype=tf.int32),
                                                                                        self._input_var), 
                                                                                       back_prop=False)
            
            update_ops = [tf.assign(self._damping_factor, damping_new),
                              tf.assign(self._update_var, update),
                              tf.assign(self._input_var, self._input_var + update)]
            with tf.control_dependencies(update_ops):
                cg_counter_op = tf.assign(self._total_cg_iterations, self._total_cg_iterations + cg_iterations,
                                          name='cg_counter_op')
            with tf.control_dependencies([cg_counter_op]):
                counter_op = tf.assign(self._iteration, self._iteration + 1, name='counter_op')
        return counter_op





