#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from typing import Callable, NamedTuple

from sopt.optimizers.tensorflow.utils.linear_conjugate_gradient import MatrixFreeLinearOp, conjugate_gradient
from sopt.optimizers.tensorflow.utils.linesearch import AdaptiveLineSearch

__all__ = ['LMA']#, 'ScaledLMA', 'PCGLMA']


class LMState(NamedTuple):
    mu_old: tf.Tensor
    mu_new: tf.Tensor
    dx: tf.Tensor
    loss: tf.Tensor
    actual_reduction: tf.Tensor
    pred_reduction: tf.Tensor
    ratio: tf.Tensor
    cgi: tf.Tensor
    converged: tf.Tensor
    i: tf.Tensor
    v_const: tf.Tensor

class LMA(object):
    def __init__(self,
                 input_var: tf.Variable,
                 predictions_fn: Callable[[tf.Tensor], tf.Tensor],
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 name: str,
                 mu: float = 1e-5,
                 grad_norm_regularization_power: float = 1.0,
                 mu_contraction: float = 0.25,
                 mu_expansion: float = 4.,
                 update_cond_thres_low: float = 0.25,
                 update_cond_thres_high: float = 0.75,
                 mu_thres_low: float = 1e-8,
                 mu_thres_high: float = 1e10,
                 max_mu_linesearch_iters: int = None,
                 max_cg_iter: int = 100,
                 ftol: float = 1e-6,
                 gtol: float = 1e-6,
                 min_reduction_ratio: float = 1e-4,
                 proj_min_reduction_ratio: float = None,
                 diag_hessian_fn: Callable[[tf.Tensor], tf.Tensor]= None,
                 diag_mu_scaling_t: tf.Tensor = None,  # Use  Marquardt-Fletcher scaling
                 diag_precond_t: tf.Tensor = None,  # Use preconditioning for CG steps
                 stochastic_diag_estimator_type: str = None, # Use random sampling to estimate the diagonal elements
                 stochastic_diag_estimator_iters: int = 1, # Number of matrix-vector-product iterations to use for the estimation
                 #diag_mu_thres_low: float = 1e-8,
                 min_cg_tol: float = None,  # Force CG iterations to have at least this tolerance.s
                 warm_start: bool = True,
                 assert_tolerances: bool = False) -> None:

        self._name = name
        self._input_var = input_var
        self._dtype = input_var.dtype.base_dtype.name
        self._machine_eps = np.finfo(np.dtype(self._dtype)).eps

        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        self._preds_t = self._predictions_fn(self._input_var)
        self._loss_t = self._loss_fn(self._preds_t)

        # Multiplicating factor to update the damping factor at the end of each cycle
        self._mu_contraction = mu_contraction
        self._mu_expansion = mu_expansion
        self._update_cond_thres_low = update_cond_thres_low
        self._update_cond_thres_high =  update_cond_thres_high
        self._mu_thres_low = mu_thres_low
        self._mu_thres_high = mu_thres_high
        self._grad_norm_regularization_power = grad_norm_regularization_power

        self._max_linesearch_iters = max_mu_linesearch_iters
        if max_mu_linesearch_iters is None:
            range = self._mu_thres_high / self._mu_thres_low
            self._max_linesearch_iters = np.ceil(-np.log(range) / np.log(mu_contraction)).astype('int32')

        self._max_cg_iter = max_cg_iter

        # The recommended value from Nocedal and Wright is 0.5
        self._min_cg_tol = 0.5 if min_cg_tol is None else min_cg_tol
        self._warm_start = warm_start

        self._min_reduction_ratio = min_reduction_ratio

        self._proj_min_reduction_ratio = proj_min_reduction_ratio
        if proj_min_reduction_ratio is None:
            # The logic here is that if the projection = 0.1 * dx0 (which is a reasonable low bound),
            # then the function value becomes approximately 0.01 * fn0.
            # Since the LM routine is calibrated to generate updates such that (1 - pred / )
            # Setting this ratio to lower than 1e-6 doesn't seem to help, so this seems like a good compromise
            self._proj_min_reduction_ratio = 0.01 * min_reduction_ratio

        self._ftol, self._gtol = self._checkTolerance(ftol, gtol)
        self._assert_tolerances = assert_tolerances

        self._diag_hessian_fn = diag_hessian_fn

        #if (diag_mu_scaling_t is not None) and (diag_precond_t is not None):
        #    raise ValueError("Cannot enable both Marquardt-Fletcher scaling and CG preconditioning. "
        #                     + "This has not been tested.")

        self._diag_mu_scaling_t = diag_mu_scaling_t
        #self._diag_mu_thres_low = diag_mu_thres_low

        self._diag_precond_t = diag_precond_t

        # Currently supporting two types of random estimation:
        # "martens" : See Chapter 4, Algorithm 6 in Martens, J. (2016). Second-Order Optimization for Neural Networks.
        # U. of Toronto Thesis, 179.
        # "bekas": Bekas, C., Kokiopoulou, E., & Saad, Y. (2007).
        # An estimator for the diagonal of a matrix. Applied Numerical Mathematics, 57(11–12), 1214–1229.
        # https://doi.org/10.1016/j.apnum.2007.01.003
        if stochastic_diag_estimator_type is not None:
            if diag_mu_scaling_t is not None or diag_precond_t is not None:
                raise ValueError("Cannot use stochastic estimation on top of actual supplied tensors")
            if stochastic_diag_estimator_type not in ['martens', 'bekas']:
                raise ValueError('"martens" and "bekas" are the only supported options.')
            if stochastic_diag_estimator_type == 'martens' and self._diag_hessian_fn is None:
                raise ValueError('Martens-type stochastic estimation of the GGN diagonal requires the diagonal of the' +
                                 ' inner hessian matrix as input.')
        self._stochastic_diag_estimator_type = stochastic_diag_estimator_type
        self._stochastic_diag_estimator_iters = stochastic_diag_estimator_iters


        with tf.variable_scope(name):
            self._mu = tf.Variable(mu, dtype=self._dtype, name="lambda", trainable=False)
            self._update_var = tf.Variable(tf.zeros_like(self._input_var, dtype=self._dtype), name="delta",
                                               trainable=False)
            self._dummy_var = tf.Variable(tf.ones_like(self._preds_t),name="dummy", dtype=self._dtype, trainable=False)

            self._loss_before_update = tf.Variable(0., name="loss_before_update", dtype=self._dtype,
                                                   trainable=False)

            self._iteration = tf.Variable(0, name="iteration", dtype=tf.int32, trainable=False)

            self._total_cg_iterations = tf.Variable(0, name="total_cg_iterations",
                                                        dtype=tf.int32,
                                                        trainable=False)

            self._projected_gradient_iterations = tf.Variable(0, name="projected_grad_iterations",
                                                                  dtype=tf.int32,
                                                                  trainable=False)
            self._total_proj_ls_iterations = tf.Variable(0, name="total_projection_line_search_iterations",
                                                             dtype=tf.int32,
                                                             trainable=False)

            self._projected_gradient_linesearch = AdaptiveLineSearch(name='proj_ls_linesearch', dtype=self._dtype)
                                                                     #suff_decr=self._min_reduction_ratio)

            # This stores the maximum encountered values of the diagonal of the GN matrix.
            # This is based on the minpack implementation of the LM problem
            # For reference, see Section 2.2 here:
            # https://arxiv.org/pdf/1201.5885.pdf
            if self._diag_mu_scaling_t is not None or (stochastic_diag_estimator_type is not None):
                self._diag_mu_max_values_t = tf.Variable(tf.zeros_like(self._input_var,
                                                                       dtype=self._dtype),
                                                         name="diag_mu_max_values", trainable=False)



        self._minimize_output_op = self._setupMinimizeOp()

        self._variables = [self._mu, self._update_var, self._dummy_var,
                           self._loss_before_update, self._iteration, self._total_cg_iterations,
                           self._projected_gradient_iterations, self._total_proj_ls_iterations]
        reset_ops = [v.assign(v.initial_value) for v in self._variables]
        self._reset_op = tf.group([*reset_ops, self._projected_gradient_linesearch.reset])

    @property
    def reset(self):
        return self._reset_op

    def _checkTolerance(self, ftol, gtol):
        """This is adapted almost exactly from the corresponding scipy function"""
        def check(tol, name):
            if (tol is None) or (tol < self._machine_eps):
                tol = self._machine_eps
                print(f"If {name} tolerance is set to None or to below the machine epsilon ({self._machine_eps:.2e}), "
                      + "it is reset to the machine epsilon by default.")
            return tol

        ftol = check(ftol, "ftol")
        gtol = check(gtol, "gtol")
        return ftol, gtol

    def _setupHessianVectorProduct(self,
                                   jvp_fn: Callable[[tf.Tensor], tf.Tensor],
                                   x: tf.Tensor,
                                   v_constant: tf.Tensor) -> tf.Tensor:
        predictions_this = self._predictions_fn(v_constant)
        if self._diag_hessian_fn is None:
            loss_this = self._loss_fn(predictions_this)
            hjvp = _hessian_vector_product(ys=[loss_this],
                                           xs=[predictions_this],
                                           v=[jvp_fn(x)])
        else:
            hjvp = self._diag_hessian_fn(predictions_this) * jvp_fn(x)

        jhjvp = tf.gradients(predictions_this, v_constant, hjvp)[0]
        return jhjvp
        
    def _setupSecondOrder(self) -> None:
        with tf.name_scope(self._name + '_gngvp'):
            self.vjp_fn = lambda x: tf.gradients(self._preds_t, self._input_var, x,
                                                 stop_gradients=[x], name="vjp")[0]
            jvp_fn = lambda x: tf.gradients(self.vjp_fn(self._dummy_var), self._dummy_var, x, name='jvpz')[0]
            self.jvp_fn = jvp_fn
            
            self._jhjvp_fn = lambda x, v_constant: self._setupHessianVectorProduct(jvp_fn, x, v_constant)

            self._loss_grads_t = tf.gradients(self._loss_t, self._preds_t)[0]
            self._grads_t = self.vjp_fn(self._loss_grads_t)

            #self._loss_grads_fn = lambda p: tf.gradients(self._loss_fn(p), p)[0]
            #self._grads_fn = lambda p: self.vjp_fn(self._loss_grads_fn(p))

    @staticmethod
    def _applyConstraint(input_var, lma_update):
        return input_var.constraint(input_var + lma_update)


    def _applyProjectedGradient(self, lmstate):#, lm_update, beta = 0.5, sigma=1e-5):
        """
        Reference:
        Journal of Computational and Applied Mathematics 172 (2004) 375–397
        """

        projected_var = self._applyConstraint(self._input_var, lmstate.dx)
        projected_loss_new = self._loss_fn(self._predictions_fn(projected_var))
        projected_loss_change = self._loss_before_update - projected_loss_new
        projection_reduction_ratio = projected_loss_change / tf.abs(self._loss_before_update)
        #with tf.control_dependencies([tf.print(projection_reduction_ratio)]):
        fconv = tf.abs(self._loss_before_update) <= (self._machine_eps)

        #dx = projected_var - self._input_var
        #f0 = self._loss_before_update
        #df0 = tf.reduce_sum(dx * self._grads_t)
        #test_condition = (projected_loss_new < (f0 + 1e-4 * df0))

        no_projection_condition = ((projection_reduction_ratio > self._proj_min_reduction_ratio) | fconv)

        def _loss_and_update_fn(x, y):
            update = self._applyConstraint(x, y)
            loss = self._loss_fn(self._predictions_fn(update))
            return loss, update

        def _linesearch():
            dx = projected_var - self._input_var
            lhs = tf.reduce_sum(dx * self._grads_t)
            rhs = 1e-8 * tf.linalg.norm(dx) ** 2.1
            #with tf.control_dependencies([tf.print('lhs', lhs, '-rhs', -rhs,
            #                                       'alpha', self._projected_gradient_linesearch._alpha)]):
            descent_dir = tf.cond(lhs <= -rhs, lambda: dx, lambda: -self._grads_t)
            #descent_dir = -self._grads_t

            linesearch_state = self._projected_gradient_linesearch.search(objective_and_update=_loss_and_update_fn,
                                                                          x0=self._input_var,
                                                                          descent_dir=descent_dir,  # -self._grads_t,
                                                                          gradient=self._grads_t,
                                                                          f0=self._loss_before_update)
            counter_ops = [self._total_proj_ls_iterations.assign_add(linesearch_state.step_count),
                          self._projected_gradient_iterations.assign_add(1)]
            with tf.control_dependencies(counter_ops):#, tf.print(self._loss_before_update - linesearch_state.newf)]):
                output = tf.identity(linesearch_state.newx)
            return output

        #print_op = tf.print(  # 'loss_before_LM', self._loss_t,
            #                    'loss_after_lm', loss_new,
            #                    'reduction_ration', reduction_ratio,
            # 'loss_after_lm', lmstate.loss,
            # 'loss_after_constraint', projected_loss_new,
        #    'loss_change', lmstate.actual_reduction,
        #    'projection_reduction_ratio', projection_reduction_ratio,
        #    'projection_condition', no_projection_condition,
            #'test_condition', test_condition,
            #'df0', df0,
        #    'df1', projected_loss_change)
        # 'test_ratio', projected_loss_new / (f0 + 1e-4 * df0 / tf.linalg.norm(dx)))
        #                    'test_rhs', test_rhs, 'dist', dist)

        #with tf.control_dependencies([print_op]):
        input_var_update = tf.cond(no_projection_condition,
                                       lambda: projected_var,
                                       _linesearch)
        dx_new = tf.cond(no_projection_condition,
                         lambda: lmstate.dx,
                         lambda: input_var_update - self._input_var)
                         #lambda: tf.zeros_like(lmstate.dx))
        return input_var_update, dx_new

    def _setupStochasticDiagEstimation(self):

        if self._stochastic_diag_estimator_type == 'martens':
            rands = tf.random.uniform(shape=[self._stochastic_diag_estimator_iters,
                                             *self._preds_t.shape.as_list()],
                                      minval=0, maxval=2, dtype=tf.int32)
            rands = tf.cast(rands, dtype=tf.float32) * 2 - 1

            hessian_t = self._diag_hessian_fn(self._preds_t)
            stochastic_diag_fn = lambda v: self.vjp_fn(hessian_t ** 0.5 * v) ** 2

            #stochastic_diags = tf.map_fn(stochastic_diag_fn, rands)
            stochastic_diags = []
            rands_unstacked = tf.unstack(rands, axis=0)
            for r in rands_unstacked:
                sd = stochastic_diag_fn(r)
                stochastic_diags.append(sd)

            mean_estimation = tf.reduce_mean(stochastic_diags, axis=0)
            self._diag_mu_scaling_t = mean_estimation
            self._diag_precond_t = mean_estimation
        elif self._stochastic_diag_estimator_type == 'bekas':
            rands = tf.random.uniform(shape=[self._stochastic_diag_estimator_iters,
                                             *self._input_var.shape.as_list()],
                                      minval=0, maxval=2, dtype=tf.int32)
            rands = tf.cast(rands, dtype=tf.float32) * 2 - 1
            stochastic_diag_fn = lambda v: v * self._jhjvp_fn(v, self._input_var)
            #stochastic_diags = tf.map_fn(stochastic_diag_fn, rands)
            stochastic_diags = []
            rands_unstacked = tf.unstack(rands, axis=0)
            for r in rands_unstacked:
                sd = stochastic_diag_fn(r)
                stochastic_diags.append(sd)
            mean_estimation = tf.reduce_mean(stochastic_diags, axis=0)
            self._diag_mu_scaling_t = mean_estimation
            self._diag_precond_t = mean_estimation


    def minimize(self):
        return self._minimize_output_op

    def _setupMinimizeOp(self) -> tf.Operation:
        tf.logging.warning("The ftol, gtol, and xtol conditions are adapted from "
                           + "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html."
                           + "This is a test version, and there is no guarantee that these work as intended.")
        with tf.name_scope(self._name + '_minimize_step'):

            # Set up the second order calculations to define matrix-free linear ops.
            self._setupSecondOrder()

            # Set up stochastic GGN diagonal calculation
            if self._stochastic_diag_estimator_type is not None:
                self._setupStochasticDiagEstimation()

            print(self._loss_before_update, self._loss_t)
            store_loss_op = tf.assign(self._loss_before_update, self._loss_t,
                                      name='store_loss_op')

            #if self._diag_mu_scaling_t is not None:
            #    jhjvp_fn_l_h = lambda l, h, v_constant: self._jhjvp_fn(h, v_constant) + l * h #* self._diag_mu_scaling_t
            #else:
            jhjvp_fn_l_h = lambda l, h, v_constant: self._jhjvp_fn(h, v_constant) + l * h
            linear_b = -self._grads_t

            grad_norm = tf.linalg.norm(self._grads_t)
            this_forcing_eta = tf.minimum(grad_norm ** 0.5, self._min_cg_tol)

            grad_norm_regularization = 1.0
            if self._grad_norm_regularization_power != 0:
                grad_norm_regularization = tf.linalg.norm(self._grads_t) ** self._grad_norm_regularization_power


            if self._diag_mu_scaling_t is not None:
                diag_mu_scaling_this_iter = tf.where(self._diag_mu_scaling_t > self._diag_mu_max_values_t,
                                           self._diag_mu_scaling_t,
                                           self._diag_mu_max_values_t)
                diag_mu_scaling_this_iter = tf.where(tf.equal(diag_mu_scaling_this_iter, 0),
                                           tf.ones_like(diag_mu_scaling_this_iter),
                                           diag_mu_scaling_this_iter)

            def _damping_linesearch_step(state: LMState):
                if self._diag_mu_scaling_t is not None:
                    damping = state.mu_new * grad_norm_regularization * diag_mu_scaling_this_iter
                else:
                    damping = state.mu_new * grad_norm_regularization

                linear_ax = MatrixFreeLinearOp(lambda h: jhjvp_fn_l_h(damping, h, state.v_const),
                                               tf.TensorShape((self._input_var.shape.dims[0],
                                                               self._input_var.shape.dims[0])),
                                               self._dtype)

                preconditioner = None # Default
                if self._diag_precond_t is not None:

                    precond_t = 1 / (self._diag_precond_t + damping)
                    preconditioner = MatrixFreeLinearOp(lambda x: x * precond_t,
                                                         tf.TensorShape((self._input_var.shape.dims[0],
                                                                         self._input_var.shape.dims[0])),
                                                        self._dtype)

                cg_solve = conjugate_gradient(operator=linear_ax,
                                              rhs=linear_b,
                                              x=state.dx,
                                              tol=this_forcing_eta,
                                              max_iter=self._max_cg_iter,
                                              preconditioner=preconditioner)

                cg_update = tf.identity(cg_solve.x, name='cg_solved')
                optimized_var = self._input_var + cg_update

                pred_reduction = -0.5 * tf.tensordot(cg_update, damping * cg_update + linear_b, 1)

                loss_new = self._loss_fn(self._predictions_fn(optimized_var))
                actual_reduction = loss_new - self._loss_before_update

                ratio = tf.cond(tf.math.not_equal(pred_reduction, tf.constant(0., dtype=self._dtype)),
                                lambda: actual_reduction / pred_reduction,
                                lambda: tf.constant(0., dtype=self._dtype))
                f1 = lambda: tf.minimum(state.mu_new * self._mu_expansion, self._mu_thres_high)
                f2 = lambda: tf.maximum(state.mu_new * self._mu_contraction, self._mu_thres_low)
                f3 = lambda: state.mu_new


                # Using updates from "On a New Updating Rule of the Levenberg–Marquardt Parameter"
                case1 = ratio < self._min_reduction_ratio
                case2 = (ratio >= self._min_reduction_ratio) & (ratio < self._update_cond_thres_low)
                case3 = (ratio > self._update_cond_thres_high)
                #case2 = ((ratio > self._min_reduction_ratio)
                #         & (grad_norm < (self._update_cond_thres_low / state.mu_new)))

                #case3 = ((ratio > self._min_reduction_ratio)
                #         & (grad_norm > (self._update_cond_thres_high / state.mu_new)))

                #with tf.control_dependencies([tf.print('ratio', ratio, 'damping', damping)]):
                mu_new = tf.case({case1: f1, case2: f1, case3: f2}, default=f3, exclusive=True)

                #with tf.control_dependencies([tf.print('ratio', ratio,
                #                                       'loss_old', self._loss_before_update,
                #                                       'loss_new', loss_new)]):
                #update_factor = tf.case({tf.less(ratio, self._update_cond_thres_low):f1,
                #                         tf.greater(ratio, self._update_cond_thres_high):f2},
                #                       default=f3, exclusive=True)

                #mu_new = tf.clip_by_value(state.mu_new * update_factor,
                #                          self._mu_thres_low,# / grad_norm_regularization,
                #                          self._mu_thres_high)# / grad_norm_regularization)

                state_new = state._replace(mu_old=state.mu_new,
                                           mu_new=mu_new,
                                           loss=loss_new,
                                           dx=cg_update,
                                           ratio=ratio,
                                           actual_reduction=actual_reduction,
                                           pred_reduction=pred_reduction,
                                           cgi=state.cgi + cg_solve.i,
                                           i=state.i + 1)
                converged = self._checkConvergenceAndTolerances(state_new)
                state_new = state_new._replace(converged=converged)
                return [state_new]

            def _damping_linesearch_cond(state: LMState):
                #print_op = tf.print(state.i, state.ratio, state.converged, state.pred_reduction, state.actual_reduction)
                #with tf.control_dependencies([print_op]):
                stop_cond = ~((state.converged > 0)
                              | (state.ratio >= self._min_reduction_ratio))
                return stop_cond

            # Check for gradient convergence and tolerance
            grad_inf_norm = tf.norm(self._grads_t, ord=np.inf)
            grad_converged = tf.cond(grad_inf_norm < self._gtol, lambda: 1, lambda: 0)

            x0 = tf.zeros_like(self._update_var)
            if self._warm_start:
                x0 = self._update_var
            lmstate0 = LMState(mu_old=self._mu,
                               mu_new=self._mu,
                               dx=x0,
                               loss=tf.constant(0., dtype=self._dtype),
                               actual_reduction=tf.constant(0., dtype=self._dtype),
                               pred_reduction=tf.constant(1., dtype=self._dtype),
                               ratio=tf.constant(0., dtype=self._dtype),
                               cgi=tf.constant(0, dtype='int32'),
                               v_const=self._input_var,
                               converged=grad_converged,
                               i=tf.constant(0, dtype='int32'))

            while_loop_op = lambda: tf.while_loop(_damping_linesearch_cond,
                                                  _damping_linesearch_step,
                                                  [lmstate0],
                                                  back_prop=False)
            with tf.control_dependencies([store_loss_op]):
                lmstate = tf.cond(grad_converged > 0, lambda: [lmstate0], while_loop_op)

            # Update x only if the reduction ratio changed sufficiently
            dx_new = tf.cond(lmstate.ratio >= self._min_reduction_ratio,
                             lambda: lmstate.dx,
                             lambda: tf.zeros_like(self._update_var))

            assert_ops = []
            if self._assert_tolerances:
                message_str = "Meets one of the convergence criterions or strict tolerance criterions. " +\
                              "Check returned code for detailed information."
                assert_op = tf.assert_greater(lmstate.converged, 0, summarize=1,
                                              message=message_str)
                assert_ops.append(assert_op)

            no_update_condition = ((grad_converged > 0)
                                   | (lmstate.ratio < self._min_reduction_ratio))

            with tf.control_dependencies(assert_ops):
                if self._input_var.constraint is not None:
                    updated_var, dx_new = tf.cond(no_update_condition,
                                          lambda: (self._input_var, dx_new),
                                          lambda: self._applyProjectedGradient(lmstate))
                else:
                    updated_var = tf.cond(no_update_condition,
                                          lambda: self._input_var,
                                          lambda: self._input_var + dx_new)
            update_ops = [tf.assign(self._mu, lmstate.mu_new),
                          tf.assign(self._update_var, dx_new),
                          tf.assign(self._input_var, updated_var)]
            if self._diag_mu_scaling_t is not None:
                update_ops.append(tf.assign(self._diag_mu_max_values_t, diag_mu_scaling_this_iter))
            with tf.control_dependencies([*update_ops]):
                counter_ops = [self._total_cg_iterations.assign_add(lmstate.cgi, name='cg_counter_op'),
                                self._iteration.assign_add(lmstate.i, name='counter_op')]
            with tf.control_dependencies(counter_ops):
                output_op = tf.identity(lmstate.converged)

        return output_op

    def _checkConvergenceAndTolerances(self, lmstate):

        fconv = ((tf.abs(lmstate.actual_reduction) <= self._ftol)
                 & (tf.abs(lmstate.pred_reduction) <= self._ftol)
                 & (lmstate.ratio <= 1)
                 & (lmstate.i > 0))
        dtol = (((lmstate.mu_old == self._mu_thres_high)
                 | (lmstate.mu_old == self._mu_thres_low))
                & (lmstate.i > 0))
        itol = lmstate.i >= self._max_linesearch_iters

        convergence_condition = tf.case({fconv: lambda: 2,
                                         dtol: lambda: 3,
                                         itol: lambda: 4},
                                        default=lambda: 0,
                                        exclusive=False)
        return convergence_condition


class ScaledLMA(LMA):
    "This is just for convenience"
    def __init__(self,
                 diag_mu_scaling_t: tf.Tensor,
                 *args: int,
                 mu: float = 1.0,
                 grad_norm_regularization_power: float = 0.0,
                 **kwargs: int):
        super().__init__(*args,
                         diag_mu_scaling_t=diag_mu_scaling_t,
                         mu=mu,
                         grad_norm_regularization_power=grad_norm_regularization_power,
                         **kwargs)




class PCGLMA(LMA):
    """This does not work."""
    def __init__(self, *args,
                 diag_precond_t: tf.Tensor = None,
                 **kwargs: int):
        super().__init__(*args, **kwargs)
        self._diag_precond_t = diag_precond_t

    def minimize(self) -> tf.Operation:
        tf.logging.warning("The ftol, gtol, and xtol conditions are adapted from "
                           + "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html."
                           + "This is a test version, and there is no guarantee that these work as intended.")
        with tf.name_scope(self._name + '_minimize_step'):

            store_loss_op = tf.assign(self._loss_before_update, self._loss_t,
                                      name='store_loss_op')

            jhjvp_fn_l_h = lambda l, h, v_constant: self._jhjvp_fn(h, v_constant) + l * h
            linear_b = -self._grads_t

            grad_norm = tf.linalg.norm(self._grads_t)
            this_forcing_eta = tf.minimum(grad_norm ** 0.5, self._min_cg_tol)

            grad_norm_regularization = tf.linalg.norm(self._grads_t) ** self._grad_norm_regularization_power


            def _damping_linesearch_step(state: LMState):
                damping = state.mu_new * grad_norm_regularization
                linear_ax = MatrixFreeLinearOp(lambda h: jhjvp_fn_l_h(damping, h, state.v_const),
                                               tf.TensorShape((self._input_var.shape.dims[0],
                                                               self._input_var.shape.dims[0])))

                precond_t = 1/(self._diag_precond_t + damping)
                precond_min = tf.reduce_min(precond_t)
                precond_t = tf.clip_by_value(precond_t, clip_value_min=precond_min,
                                             clip_value_max=precond_min * 10**7)
                precondition_op = MatrixFreeLinearOp(lambda x: x * precond_t,
                                                     tf.TensorShape((self._input_var.shape.dims[0],
                                                                     self._input_var.shape.dims[0])))

                cg_solve = conjugate_gradient(operator=linear_ax,
                                              rhs=linear_b,
                                              x=None,
                                              tol=this_forcing_eta,
                                              max_iter=self._max_cg_iter,
                                              preconditioner=precondition_op)
                cg_update = tf.identity(cg_solve.x, name='cg_solved')
                optimized_var = self._input_var + cg_update
                pred_reduction = -0.5 * tf.tensordot(cg_update, damping * cg_update + linear_b, 1)

                loss_new = self._loss_fn(self._predictions_fn(optimized_var))
                actual_reduction = loss_new - self._loss_before_update
                ratio = tf.cond(tf.math.not_equal(pred_reduction, 0.),
                                lambda: actual_reduction / pred_reduction,
                                lambda: 0.)
                f1 = lambda: tf.minimum(state.mu_new * self._mu_expansion, self._mu_thres_high)
                f2 = lambda: tf.maximum(state.mu_new * self._mu_contraction, self._mu_thres_low)
                f3 = lambda: state.mu_new

                # Using updates from "On a New Updating Rule of the Levenberg–Marquardt Parameter"
                case1 = ratio < self._min_reduction_ratio
                case2 = (ratio >= self._min_reduction_ratio) & (ratio < self._update_cond_thres_low)
                case3 = (ratio > self._update_cond_thres_high)
                # case2 = ((ratio > self._min_reduction_ratio)
                #         & (grad_norm < (self._update_cond_thres_low / state.mu_new)))

                # case3 = ((ratio > self._min_reduction_ratio)
                #         & (grad_norm > (self._update_cond_thres_high / state.mu_new)))

                # with tf.control_dependencies([tf.print('ratio', ratio, case1, case2, case3, state.mu_new)]):
                mu_new = tf.case({case1: f1, case2: f1, case3: f2}, default=f3, exclusive=True)

                # with tf.control_dependencies([tf.print('ratio', ratio,
                #                                       'loss_old', self._loss_before_update,
                #                                       'loss_new', loss_new)]):
                # update_factor = tf.case({tf.less(ratio, self._update_cond_thres_low):f1,
                #                         tf.greater(ratio, self._update_cond_thres_high):f2},
                #                       default=f3, exclusive=True)

                # mu_new = tf.clip_by_value(state.mu_new * update_factor,
                #                          self._mu_thres_low,# / grad_norm_regularization,
                #                          self._mu_thres_high)# / grad_norm_regularization)

                state_new = state._replace(mu_old=state.mu_new,
                                           mu_new=mu_new,
                                           loss=loss_new,
                                           dx=cg_update,
                                           ratio=ratio,
                                           actual_reduction=actual_reduction,
                                           pred_reduction=pred_reduction,
                                           cgi=state.cgi + cg_solve.i,
                                           i=state.i + 1)
                converged = self._checkConvergenceAndTolerances(state_new)
                state_new = state_new._replace(converged=converged)
                return [state_new]

            def _damping_linesearch_cond(state: LMState):
                # print_op = tf.print(state.i, state.ratio, state.converged, state.pred_reduction, state.actual_reduction)
                # with tf.control_dependencies([print_op]):
                stop_cond = ~((state.converged > 0)
                              | (state.ratio >= self._min_reduction_ratio))
                return stop_cond

            # Check for gradient convergence and tolerance
            grad_inf_norm = tf.norm(self._grads_t, ord=np.inf)
            grad_converged = tf.cond(grad_inf_norm < self._gtol, lambda: 1, lambda: 0)

            lmstate0 = LMState(mu_old=self._mu,
                               mu_new=self._mu,
                               dx=tf.zeros_like(self._update_var),
                               loss=tf.constant(0., dtype=self._dtype),
                               actual_reduction=tf.constant(0., dtype=self._dtype),
                               pred_reduction=tf.constant(1., dtype=self._dtype),
                               ratio=tf.constant(0., dtype=self._dtype),
                               cgi=tf.constant(0, dtype='int32'),
                               v_const=self._input_var,
                               converged=grad_converged,
                               i=tf.constant(0, dtype='int32'))

            while_loop_op = lambda: tf.while_loop(_damping_linesearch_cond,
                                                  _damping_linesearch_step,
                                                  [lmstate0],
                                                  back_prop=False)
            with tf.control_dependencies([store_loss_op]):
                lmstate = tf.cond(grad_converged > 0, lambda: [lmstate0], while_loop_op)

            # Update x only if the reduction ratio changed sufficiently
            dx_new = tf.cond(lmstate.ratio >= self._min_reduction_ratio,
                             lambda: lmstate.dx,
                             lambda: tf.zeros_like(self._update_var))

            assert_ops = []
            if self._assert_tolerances:
                message_str = "Meets one of the convergence criterions or strict tolerance criterions. " + \
                              "Check returned code for detailed information."
                assert_op = tf.assert_greater(lmstate.converged, 0, summarize=1,
                                              message=message_str)
                assert_ops.append(assert_op)

            no_update_condition = ((grad_converged > 0)
                                   | (lmstate.ratio < self._min_reduction_ratio))
            with tf.control_dependencies(assert_ops):
                if self._input_var.constraint is not None:
                    updated_var = tf.cond(no_update_condition,
                                          lambda: self._input_var,
                                          lambda: self._applyProjectedGradient(lmstate))
                else:
                    updated_var = tf.cond(no_update_condition,
                                          lambda: self._input_var,
                                          lambda: self._input_var + dx_new)
            update_ops = [tf.assign(self._mu, lmstate.mu_new),
                          tf.assign(self._update_var, dx_new),
                          tf.assign(self._input_var, updated_var)]
            with tf.control_dependencies([*update_ops]):
                counter_ops = [self._total_cg_iterations.assign_add(lmstate.cgi, name='cg_counter_op'),
                               self._iteration.assign_add(lmstate.i, name='counter_op')]
            with tf.control_dependencies(counter_ops):
                output_op = tf.identity(lmstate.converged)

        return output_op


