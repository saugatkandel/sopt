from typing import Callable, NamedTuple, List, Tuple, Union
import tensorflow as tf
import numpy as np
from sopt.optimizers.tensorflow2.utils import MatrixFreeLinearOp, conjugate_gradient, AdaptiveLineSearch
import sopt.optimizers.tensorflow2.utils.autodiff_helper as adh
import logging
logger = tf.get_logger()
logger.setLevel(logging.WARNING)

__all__ = ['LMA']

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
    #v_const: tf.Tensor


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
                 diag_hessian_fn: Callable[[tf.Tensor], tf.Tensor] = None,
                 diag_mu_scaling_t: tf.Tensor = None,  # Use  Marquardt-Fletcher scaling
                 diag_precond_t: tf.Tensor = None,  # Use preconditioning for CG steps
                 min_cg_tol: float = None,  # Force CG iterations to have at least this tolerance.s
                 warm_start: bool = False,
                 assert_tolerances: bool = False) -> None:

        self._name = name
        if len(input_var.shape) > 1:
            raise ValueError("The optimizer currently only supports a one-dimensional variable array. "
                             + "Reshaping into multidimensional arrays should can be wrapped into predictions_fn.")
        self._input_var = input_var
        self._dtype = input_var.dtype.base_dtype.name
        self._machine_eps = np.finfo(self._dtype).eps

        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn

        # Multiplicating factor to update the damping factor at the end of each cycle
        self._mu_contraction = mu_contraction
        self._mu_expansion = mu_expansion
        self._update_cond_thres_low = update_cond_thres_low
        self._update_cond_thres_high = update_cond_thres_high
        self._mu_thres_low = mu_thres_low
        self._mu_thres_high = mu_thres_high
        self._grad_norm_regularization_power = grad_norm_regularization_power

        self._max_linesearch_iters = max_mu_linesearch_iters
        if max_mu_linesearch_iters is None:
            mu_range = self._mu_thres_high / self._mu_thres_low
            self._max_linesearch_iters = np.ceil(-np.log(mu_range) / np.log(mu_contraction)).astype('int32')

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

        self._ftol, self._gtol = self._check_tolerance(ftol, gtol)
        self._assert_tolerances = assert_tolerances

        self._diag_hessian_fn = diag_hessian_fn

        # if (diag_mu_scaling_t is not None) and (diag_precond_t is not None):
        #    raise ValueError("Cannot enable both Marquardt-Fletcher scaling and CG preconditioning. "
        #                     + "This has not been tested.")

        self._diag_mu_scaling_t = diag_mu_scaling_t
        # self._diag_mu_thres_low = diag_mu_thres_low

        self._diag_precond_t = diag_precond_t

        self._mu = tf.Variable(mu, dtype=self._dtype, trainable=False)
        self._update_var = tf.Variable(tf.zeros_like(self._input_var), trainable=False)

        self._loss_old = tf.Variable(np.inf, dtype=self._dtype, trainable=False)
        self._loss_new = tf.Variable(np.inf, dtype=self._dtype, trainable=False)

        self._iteration = tf.Variable(0, dtype=tf.int32, trainable=False)

        self._total_cg_iterations = tf.Variable(0, dtype=tf.int32, trainable=False)

        self._projected_gradient_iterations = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._total_proj_ls_iterations = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._projected_gradient_linesearch = AdaptiveLineSearch(name='proj_ls_linesearch', dtype=self._dtype)

        # This stores the maximum encountered values of the diagonal of the GN matrix.
        # This is based on the minpack implementation of the LM problem
        # For reference, see Section 2.2 here:
        # https://arxiv.org/pdf/1201.5885.pdf
        if self._diag_mu_scaling_t is not None:
            self._diag_mu_max_values_t = tf.Variable(tf.zeros_like(self._diag_mu_scaling_t), trainable=False)
        # convergence status is 0 for not converged
        self._convergence_status = tf.Variable(0, dtype='int32', trainable=False)

        self._variables = [self._mu, self._update_var, self._loss_old, self._loss_new,
                           self._iteration, self._total_cg_iterations, self._projected_gradient_iterations,
                           self._total_proj_ls_iterations, self._convergence_status]


        logger.warning("The ftol, gtol, and xtol conditions are adapted from "
                       + "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html."
                       + "This is a test version, and there is no guarantee that these work as intended.")

    def reset(self):
        for v in self._variables:
            v.assign(v.initial_value)
        return self._projected_gradient_linesearch.reset()

    def _check_tolerance(self, ftol, gtol):
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

    def _hvp(self, prediction: tf.Tensor,
             vector: tf.Tensor,
             return_loss_grad: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
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

    @staticmethod
    def _applyConstraint(input_var, lma_update):
        return input_var.constraint(input_var + lma_update)

    def _applyProjectedGradient(self, lmstate, objective_grad, loss_old):  # , lm_update, beta = 0.5, sigma=1e-5):
        """
        Reference:
        Journal of Computational and Applied Mathematics 172 (2004) 375–397
        """

        projected_var = self._applyConstraint(self._input_var, lmstate.dx)
        projected_loss_new = self._loss_fn(self._predictions_fn(projected_var))
        projected_loss_change = loss_old - projected_loss_new
        projection_reduction_ratio = projected_loss_change / tf.abs(loss_old)

        fconv = tf.abs(loss_old) <= (self._machine_eps)
        no_projection_condition = ((projection_reduction_ratio > self._proj_min_reduction_ratio) | fconv)

        def _loss_and_update_fn(x, y):
            update = self._applyConstraint(x, y)
            loss = self._loss_fn(self._predictions_fn(update))
            return loss, update

        def _linesearch():
            linesearch_state = self._projected_gradient_linesearch.search(objective_and_update=_loss_and_update_fn,
                                                                          x0=self._input_var,
                                                                          descent_dir=-objective_grad,
                                                                          gradient=objective_grad,
                                                                          f0=self._loss_old)
            self._total_proj_ls_iterations.assign_add(linesearch_state.step_count),
            self._projected_gradient_iterations.assign_add(1)
            return linesearch_state

        if no_projection_condition:
            input_var_update = projected_var
            dx_new = lmstate.dx
            loss_new = projected_loss_new
        else:
            lsstate = _linesearch()
            input_var_update = lsstate.newx
            dx_new = input_var_update - self._input_var
            loss_new = lsstate.newf
        return input_var_update, dx_new, loss_new

    def _jthjvp_fn(self, vector):
        with tf.GradientTape() as gt:
            prediction, jvp = adh.jvp_forward(self._predictions_fn, self._input_var, vector)
            hjvp = self._hvp(prediction, jvp, return_loss_grad=False)

        jthjvp = gt.gradient(prediction, self._input_var, output_gradients=hjvp)
        return jthjvp

    def _getStepGvpFn(self):
        with tf.GradientTape(persistent=True) as gt:
            with tf.GradientTape(persistent=True) as gt2:
                prediction = self._predictions_fn(self._input_var)
                loss_old = self._loss_fn(prediction)
            dummy = tf.ones_like(prediction)
            gt.watch(dummy)
            vjp_fn = lambda v: gt2.gradient(prediction, self._input_var, output_gradients=v)
            inner_vjp = vjp_fn(dummy)

        jvp_fn_this = lambda v: gt.gradient(inner_vjp, dummy, output_gradients=v)
        hjvp_fn = lambda v: self._hvp(prediction, jvp_fn_this(v), return_loss_grad=False)
        jthjvp_fn = lambda v: vjp_fn(hjvp_fn(v))
        objective_grad = gt.gradient(loss_old, self._input_var)
        return loss_old, objective_grad, jthjvp_fn

    def _deprecated_gvp_fn(self, v):
        with tf.GradientTape(persistent=True) as gt:
            gt.watch(self._input_var)
            with tf.GradientTape(persistent=True) as gt2:
                prediction = self._predictions_fn(self._input_var)
            dummy = tf.ones_like(prediction)
            gt.watch(dummy)
            inner_vjp = gt2.gradient(prediction, self._input_var, output_gradients=dummy)
        jvp = gt.gradient(inner_vjp, dummy, output_gradients=v)
        hjvp = self._hvp(prediction, jvp, return_loss_grad=False)
        jthjvp = gt2.gradient(prediction, self._input_var, output_gradients=hjvp)
        return jthjvp

    @tf.function
    def minimize(self) -> tf.Tensor:
        with tf.name_scope(self._name + '_minimize_step'):

            # This works - trying to find an efficient substitute-----------------------------------------------------
            #with tf.GradientTape() as gt:
            #    loss_old = self._loss_fn(self._predictions_fn(self._input_var))
            #
            #objective_grad = gt.gradient(loss_old, self._input_var)
            #
            #jthjvp_fn_l_v = lambda l, v: self._jthjvp_fn(v) + l * v
            #---------------------------------------------------------------------------------------------------------

            loss_old, objective_grad, gvp_fn = self._getStepGvpFn()
            gvp_fn_l_v = lambda l, v: gvp_fn(v) + l * v

            linear_b = -objective_grad
            grad_norm = tf.linalg.norm(objective_grad)
            this_forcing_eta = tf.minimum(grad_norm ** 0.5, self._min_cg_tol)

            grad_norm_regularization = 1.0
            if self._grad_norm_regularization_power != 0:
                grad_norm_regularization = grad_norm ** self._grad_norm_regularization_power

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

                linear_ax = MatrixFreeLinearOp(lambda v: gvp_fn_l_v(damping, v),
                                               tf.TensorShape((self._input_var.shape.dims[0],
                                                               self._input_var.shape.dims[0])),
                                               dtype=self._dtype)

                preconditioner = None  # Default
                if self._diag_precond_t is not None:
                    precond_t = 1 / (self._diag_precond_t + damping)
                    preconditioner = MatrixFreeLinearOp(lambda x: x * precond_t,
                                                        tf.TensorShape((self._input_var.shape.dims[0],
                                                                        self._input_var.shape.dims[0])),
                                                        dtype=self._dtype)

                cg_x0 = tf.zeros_like(self._update_var)
                if self._warm_start:
                    cg_x0 = self._update_var
                cg_solve = conjugate_gradient(operator=linear_ax,
                                              rhs=linear_b,
                                              x=cg_x0,
                                              tol=this_forcing_eta,
                                              max_iter=self._max_cg_iter,
                                              preconditioner=preconditioner)


                optimized_var = self._input_var + cg_solve.x
                pred_reduction = -0.5 * tf.reduce_sum(cg_solve.x * (damping * cg_solve.x + linear_b))

                loss_new = self._loss_fn(self._predictions_fn(optimized_var))
                actual_reduction = loss_new - loss_old
                if pred_reduction != 0:
                    ratio = actual_reduction / pred_reduction
                else:
                    ratio = tf.constant(0., dtype=self._dtype)

                # Using updates from "On a New Updating Rule of the Levenberg–Marquardt Parameter"
                if ratio < self._min_reduction_ratio:
                    mu_new = tf.minimum(state.mu_new * self._mu_expansion, self._mu_thres_high)
                elif ratio < self._update_cond_thres_high:
                    mu_new = tf.maximum(state.mu_new * self._mu_contraction, self._mu_thres_low)
                else:
                    mu_new = state.mu_new

                state_new = state._replace(mu_old=state.mu_new,
                                           mu_new=mu_new,
                                           loss=loss_new,
                                           dx=cg_solve.x,
                                           ratio=ratio,
                                           actual_reduction=actual_reduction,
                                           pred_reduction=pred_reduction,
                                           cgi=state.cgi + cg_solve.i,
                                           i=state.i + 1)
                converged = self._checkConvergenceAndTolerances(state_new)
                state_new = state_new._replace(converged=converged)
                return [state_new]

            def _damping_linesearch_cond(state: LMState):
                stop_cond = ~((state.converged > 0)
                              | (state.ratio >= self._min_reduction_ratio))
                return stop_cond

            # Check for gradient convergence and tolerance
            grad_inf_norm = tf.norm(objective_grad, ord=np.inf)
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
                               #v_const=self._input_var,
                               converged=grad_converged,
                               i=tf.constant(0, dtype='int32'))
            if grad_converged > 0:
                lmstate = lmstate0
            else:
                [
                    lmstate
                ] = tf.nest.map_structure(tf.stop_gradient,
                                          tf.while_loop(_damping_linesearch_cond,
                                                        _damping_linesearch_step,
                                                        [lmstate0]))

            # Update x only if the reduction ratio changed sufficiently
            dx_new = tf.cond(lmstate.ratio >= self._min_reduction_ratio,
                             lambda: lmstate.dx,
                             lambda: tf.zeros_like(self._update_var))

            if self._assert_tolerances:
                message_str = "Meets one of the convergence criterions or strict tolerance criterions. " + \
                              "Check returned code for detailed information."
                tf.assert_greater(lmstate.converged, 0, summarize=1,
                                  message=message_str)

            no_update_condition = ((grad_converged > 0)
                                   | (lmstate.ratio < self._min_reduction_ratio))

            updated_var = self._input_var
            loss_new = lmstate.loss
            if self._input_var.constraint is not None:
                if not no_update_condition:
                    updated_var, dx_new, loss_new = self._applyProjectedGradient(lmstate, objective_grad, loss_old)
            else:
                if not no_update_condition:
                    updated_var = self._input_var + dx_new

            self._mu.assign(lmstate.mu_new)
            self._update_var.assign(dx_new)
            self._input_var.assign(updated_var)
            self._loss_old.assign(loss_old)
            self._loss_new.assign(loss_new)

            if self._diag_mu_scaling_t is not None:
                self._diag_mu_max_values_t.assign(diag_mu_scaling_this_iter)

            self._total_cg_iterations.assign_add(lmstate.cgi, name='cg_counter_op')
            self._iteration.assign_add(lmstate.i, name='counter_op')
            self._convergence_status.assign(lmstate.converged)
        return loss_new

    def _checkConvergenceAndTolerances(self, lmstate: LMState) -> int:

        fconv = ((tf.abs(lmstate.actual_reduction) <= self._ftol)
                 & (tf.abs(lmstate.pred_reduction) <= self._ftol)
                 & (lmstate.ratio <= 1)
                 & (lmstate.i > 0))
        dtol = (((lmstate.mu_old == self._mu_thres_high)
                 | (lmstate.mu_old == self._mu_thres_low))
                & (lmstate.i > 0))
        itol = lmstate.i >= self._max_linesearch_iters

        if fconv:
            convergence_condition = 2
        elif dtol:
            convergence_condition = 3
        elif itol:
            convergence_condition = 4
        else:
            convergence_condition = 0
        return convergence_condition
