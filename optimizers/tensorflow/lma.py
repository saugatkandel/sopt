#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf



## IN CONSTRUCTION
class LMA(object):
    def __init__(self, input_var, loss_fn_input, loss_fn, squared_loss=True):
        
        self._input_var = input_var
        self._loss_fn_input = loss_fn_input
        self._loss_fn = loss_fn
        #self.damping_factor = damping_factor
        self._squared_loss = squared_loss
        
        self._update_var = tf.Variable(tf.zeros_like(input_var))
        self._dummy_var = tf.Variable(tf.zeros_like(loss_fn_input))
        
        self._vjp = tf.gradients(loss_fn_input, input_var, self._dummy_var)[0]
        self._grads = tf.gradients(loss_fn, input_var)[0]
        
        #self.linear_op = type('MyLinearOperator', (tf.linalg.LinearOperator),
        #                      {"_shape": lambda self: tf.TensorShape([input_var.shape[0],
        #                                                              input_var.shape[0]]),
        #                       "_matvec": lambda self, x: self.matrix_free_linear_op(x)})
        self._reduction_ratio_denom = tf.Variable(0., trainable=False)
    
    def _matrix_free_linear_op(self, x, damping_factor):
        jvp = tf.gradients(self._vjp, self._dummy_var, x)[0]
        
        # this could be off by a factor of two
        if self._squared_loss:
            hjvp = jvp
        else:
            hjvp = tf.gradients(tf.gradients(self._loss_fn, self._loss_fn_input)[0][None, :] @ jvp[:,None], 
                                self._loss_fn_input, stop_gradients=jvp)[0]
        
        jhjvp = tf.gradients(self._loss_fn_input, self._input_var, hjvp)[0]
        jhjvp = jhjvp + damping_factor * x
        return jhjvp 
    
    def update(self, damping_factor=0.0, tol=5e-4, maxiter=100):
        linear_op = lambda x: self._matrix_free_linear_op(x, damping_factor)
        # Use the prescription in the Marten paper and initialize the cg procedure 
        # with output from previous iteration instead of zeros
        
        # quadratic approximation to the loss function 
        # this is for the reduction ratio calculation for the lambda update
        # follows the recipe in the Marten paper
        
        update_temp, _, quadratic_vals = linear_cg_solve_martens(linear_op, -self._grads, self._update_var, tol, maxiter)
        quadratic_final = quadratic_vals.stack()[-1]
        update_ops =  [self._update_var.assign(update_temp), 
                      self._reduction_ratio_denom.assign(quadratic_final)]#0.5 * tf.tensordot(update_temp, linear_op(update_temp),1) 
                                                         #+ tf.tensordot(update_temp, self._grads,1))]
        with tf.control_dependencies(update_ops):
            assign_op = self._input_var.assign_add(self._update_var)
            
        return assign_op
    
    def revert(self):
        assign_op = self._input_var.assign_sub(self._update_var)
        return assign_op



def linear_cg_solve(linear_op, b, x_init, tol=1e-7, maxiter=None):
    """Source:
    https://stanford.edu/~boyd/papers/pdf/cvxflow_pyhpc.pdf
    
    """
    
    if isinstance(linear_op, tf.Tensor):
        matvec_op = lambda x: tf.matmul(linear_op, x)
    elif isinstance(linear_op, tf.linalg.LinearOperator):
        matvec_op = linear_op.matvec
    else:
        matvec_op = linear_op
    
    delta = tol * tf.norm(b)

    def body(x, k, r_norm_sq, r, p):
        Ap = matvec_op(p)#A(p)
        alpha = r_norm_sq / tf.tensordot(p, Ap, 1)
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm_sq_prev = r_norm_sq
        r_norm_sq = tf.tensordot(r,r,1)
        beta = r_norm_sq / r_norm_sq_prev
        p = r + beta * p
        return (x, k + 1, r_norm_sq, r, p)

    def cond(x, k, r_norm_sq, r, p):
        return tf.sqrt(r_norm_sq) > delta

    r = b - matvec_op(x_init)
    loop_vars = (x_init, tf.constant(0), tf.tensordot(r, r, 1), r, r)
    return tf.while_loop(cond, body, loop_vars, maximum_iterations=maxiter, back_prop=False)[:3]



def linear_cg_solve_martens(linear_op, b,
                            x_init, 
                            tol=5e-4, 
                            maxiter=100, 
                            miniter=10):
    """Source:
    https://stanford.edu/~boyd/papers/pdf/cvxflow_pyhpc.pdf
    adapted for the algorithm in
    "Deep learning via Hessian-free optimization"
    by J. Martens
    
    """
    
    if isinstance(linear_op, tf.Tensor):
        matvec_op = lambda x: tf.matmul(linear_op, x)
    elif isinstance(linear_op, tf.linalg.LinearOperator):
        matvec_op = linear_op.matvec
    else:
        matvec_op = linear_op
    
    quadratic_vals = tf.TensorArray(tf.float32, 
                                    size=1, 
                                    element_shape=[], 
                                    clear_after_read=False,
                                    dynamic_size=True)
    
    def quadratic(x):
        Ax = matvec_op(x)
        xAx = tf.tensordot(x, Ax, 1)
        return 0.5 * xAx - tf.tensordot(x, b, 1)

    def body(x, k, quadratic_vals, r_norm_sq, r, p):
        Ap = matvec_op(p)#A(p)
        pAp = tf.tensordot(p, Ap, 1)
        alpha = r_norm_sq / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm_sq_prev = r_norm_sq
        r_norm_sq = tf.tensordot(r,r,1)
        beta = r_norm_sq / r_norm_sq_prev
        p = r + beta * p
        return (x, k+1, quadratic_vals.write(k, quadratic(x)), r_norm_sq, r, p)

    def cond(x, k, quadratic_vals, r_norm_sq, r, p):
        kmax = tf.maximum(miniter, tf.cast(0.1 * tf.cast(k-1, 'float32'), 'int32'))
        minindx = tf.maximum(0, k-1 - kmax)
        cond1 = tf.greater(k-1, kmax)
        cond2 = tf.less(quadratic_vals.read(k-1), 0.)
        delta = tf.cast(kmax, 'float32') * tol
        cond3 = tf.less((quadratic_vals.read(k-1) - quadratic_vals.read(minindx)) / quadratic_vals.read(k-1), delta)
        cond4 = tf.logical_not(tf.logical_and(cond1, tf.logical_and(cond2, cond3)))
        return cond4
        
    r = b - matvec_op(x_init)
    loop_vars = (x_init, tf.constant(0, dtype='int32'), quadratic_vals, tf.tensordot(r, r, 1), r, r)
    
    x_new, k_new, quadratic_vals, r_norm_sq, r, p = body(*loop_vars)
    loop_vars = (x_new, k_new, quadratic_vals, r_norm_sq, r, p)
    return tf.while_loop(cond, body, loop_vars, maximum_iterations=maxiter, back_prop=False)[:3]



# Need to fit the run conditions inside the LMA class
class tfLMAPhaseRetriever(tfPhaseRetriever):
    
    def __init__(self, *args, 
                 max_cg_iter_per_step=100,
                 lambda_init=1,
                 lambda_damping=[2,3],
                 **kwargs,):
        self.max_cg_iter_per_step = max_cg_iter_per_step
        self.obj_lambda = lambda_init
        self.probe_lambda = lambda_init
        self.lambda_damping = lambda_damping
        super().__init__(*args, **kwargs)

    
    def initLossAndOptimizers(self):
        self.lossFunction()
        self.obj_lma = LMA(self.tf_obj, self.batch_differences, self.loss_fn)
        self.obj_lambda_placeholder = tf.placeholder(tf.float32, shape=[])
        self.obj_update_op = self.obj_lma.update(self.obj_lambda_placeholder,
                                                 maxiter=self.max_cg_iter_per_step)
        self.obj_revert_op = self.obj_lma.revert()
        
        self.probe_lma = LMA(self.tf_probe, self.batch_differences, self.loss_fn)
        self.probe_lambda_placeholder = tf.placeholder(tf.float32, shape=[])
        self.probe_update_op = self.probe_lma.update(self.probe_lambda_placeholder,
                                                     maxiter=self.max_cg_iter_per_step)
        self.probe_revert_op = self.probe_lma.revert()

    def run(self, n_iterations: int, 
            n_probe_fixed_iterations: int=0,
            obj_clip: bool=False,
            disable_progress_bar: bool=False):
        
        
        self.session.run(self.assign_op)
        lossval_prev = self.session.run(self.loss_fn)
        for i in tqdm(range(n_iterations), disable=disable_progress_bar):
            self.session.run(self.assign_op)
            
            while True:
                self.session.run(self.obj_update_op, 
                                 feed_dict={self.obj_lambda_placeholder:self.obj_lambda})
                lossval_new, reduction_ratio_denom = self.session.run([self.loss_fn, 
                                                                       self.obj_lma._reduction_ratio_denom])
                # The reduction ratio framework is based on the per christian hansen book
                reduction_ratio = (lossval_new - lossval_prev) / reduction_ratio_denom 
                
                if reduction_ratio < 0.25:
                    self.obj_lambda *= self.lambda_damping[0]
                    
                elif reduction_ratio > 0.75:
                    self.obj_lambda /= self.lambda_damping[1]
                
                    
                self.obj_lambda = np.clip(self.obj_lambda, a_min=1e-7, a_max=1e7)
                
                if reduction_ratio > 0:
                    lossval_prev = lossval_new
                    break
                
                self.session.run(self.obj_revert_op)
                
                #print(self.session.run([reduction_ratio, self.obj_lma._reduction_ratio_denom]))
                """
                accept = True
                if lossval_new > lossval_prev:
                    self.obj_lambda = self.obj_lambda / self.lambda_damping
                    self.session.run(self.obj_revert_op)
                    accept = False
                else:
                    self.obj_lambda = self.obj_lambda * self.lambda_damping
                
                self.obj_lambda = np.clip(self.obj_lambda, a_min=1e-7, a_max=1e7)
                if accept: 
                    lossval_prev = lossval_new
                    break
                """
            if i >= n_probe_fixed_iterations:
                while True:
                    self.session.run(self.probe_update_op, 
                                     feed_dict={self.probe_lambda_placeholder:self.probe_lambda})
                    lossval_new, reduction_ratio_denom = self.session.run([self.loss_fn,
                                                                           self.probe_lma._reduction_ratio_denom])
                    
                    reduction_ratio = (lossval_new - lossval_prev) / reduction_ratio_denom
                    
                    if reduction_ratio < 0.25:
                        self.probe_lambda *= self.lambda_damping[0]
                    elif reduction_ratio > 0.75:
                        self.probe_lambda /= self.lambda_damping[1]
                        
                    self.probe_lambda = np.clip(self.probe_lambda, a_min=1e-7, a_max=1e7)
                    if reduction_ratio > 0:
                        lossval_prev = lossval_new
                        break
                    
                    self.session.run(self.probe_revert_op)
                
                    """
                    accept = True
                    if lossval_new > lossval_prev:
                        self.probe_lambda = self.probe_lambda / self.lambda_damping
                        self.session.run(self.probe_revert_op)
                        accept = False
                    else:
                        self.probe_lambda = self.probe_lambda * self.lambda_damping
                    self.probe_lambda = np.clip(self.probe_lambda, a_min=1e-7, a_max=1e7)
                    if accept: 
                        lossval_prev = lossval_new
                        break
                    """
            
            self.losses = np.append(self.losses, lossval_new)
        if obj_clip: self.session.run(self.obj_clip_op())
        self.obj, self.probe = self.session.run([self.tf_obj_cmplx, self.tf_probe_cmplx])

