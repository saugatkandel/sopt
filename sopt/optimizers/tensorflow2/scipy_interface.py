from typing import Callable, NamedTuple, List, Tuple, Union
import tensorflow as tf
import numpy as np
import scipy.optimize as spopt
import sopt.optimizers.tensorflow2.utils.autodiff_helper as adh
import logging
logger = tf.get_logger()
logger.setLevel(logging.WARNING)

__all__ = ['ScipyOptimizerInterface']

class ScipyOptimizerInterface(object):
    """Notes :
    - distinguishing between predictions_fn and loss_fn is necessary for Gauss-Newton methods.
    - "predictions_fn" should output the predicted data, which should be a 1-d vector.
    - "loss_fn" should take the 1-d predicted data as input and return a scalar.
    - "method" is the chosen scipy optimization method
    - "use_full_hessian" distinguishes between gauss-newton methods (when False) and full newton methods (when True).
    - "diag_hessian_fn" should output the diagonal elements of the diagonal approximation to the full Hessian (for the
     newton methods) or the hessian of the loss function (for the generalized Gauss-Newton methods).
     - "optimizer_args" are the extra inputs to the scipy.optimize.minimize function.
     - "optimizer_method_options" is the input for the optimizer-specific "options" parameter.
     - Some scipy optimizers (e.g. L-BFGS-B) seem to expect float64 arrays. Since we want to work with float32 arrays
      with TensorFlow, we have to do a lot of back and forth conversion.
      - for the second-order methods, we supply a function that calculated the hessian-vector product
       ("hessp" parameter) instead of the full hessian matrix.
     """
    supported_methods = ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG', 'trust-ncg']
    def __init__(self,
                 input_var: tf.Variable,
                 predictions_fn: Callable[[tf.Tensor], tf.Tensor],
                 loss_fn: Callable[[tf.Tensor], tf.Tensor],
                 method:str,
                 name: str = '',
                 use_full_hessian: bool=False,
                 diag_hessian_fn: Callable[[tf.Tensor], tf.Tensor] = None,
                 max_outer_iterations: int = 100,
                 optimizer_args: dict=None,
                 optimizer_method_options: dict = None) -> None:

        logger.warning("This is a test version, and there is no guarantee that these work as intended.")
        self._name = name
        if len(input_var.shape) > 1:
            raise ValueError("The optimizer currently only supports a one-dimensional variable array. "
                             + "Reshaping into multidimensional arrays should can be wrapped into predictions_fn.")

        self._method = method
        if method not in self.supported_methods:
            logger.warning("Supplied method has not been tested. Use at your own risk.")
        self._input_var = input_var
        self._predictions_fn = predictions_fn
        self._loss_fn = loss_fn
        self._diag_hessian_fn = diag_hessian_fn

        self._full_objective_fn = lambda x: self._loss_fn(self._predictions_fn(x))
        self._use_full_hessian = use_full_hessian

        self._optimizer_args = {'fun': self._scipy_fun,
                                'x0': self._input_var.numpy(),
                                'method': self._method,
                                'jac': self._scipy_grad,
                                'callback': self._scipy_update_callback}
        if use_full_hessian:
            self._optimizer_args['hessp'] = self._scipy_full_hvp
        else:
            self._optimizer_args['hessp'] = self._scipy_gvp

        if optimizer_args is not None:
            self._optimizer_args.update(optimizer_args)

        self._optimizer_args['options'] = {'maxiter': max_outer_iterations}
        if optimizer_method_options is not None:
            self._optimizer_args['options'].update(optimizer_method_options)

        self._optimizer_results = None
        self._loss_per_iter = []
        self._iterations = 0

    def _scipy_fun(self, x: np.ndarray):
        x_in = tf.constant(x, dtype='float32')
        return self._full_objective_fn(x_in).numpy().astype('float64')

    def _scipy_grad(self, x: np.ndarray):
        x_in = tf.constant(x, dtype='float32')
        with tf.GradientTape() as gt:
            gt.watch(x_in)
            loss = self._full_objective_fn(x_in)
        return gt.gradient(loss, x_in).numpy().astype('float64')


    def _scipy_full_hvp(self, x: np.ndarray, v: np.ndarray):
        x_in = tf.constant(x, dtype='float32')
        v_in = tf.constant(v, dtype='float32')

        loss, hvp = self._hvp(x_in, v_in, self._full_objective_fn)
        return hvp.numpy().astype('float64')

    def _scipy_gvp(self, x: np.ndarray, v:np.ndarray):

        v_in = tf.constant(v, dtype='float32')
        x_in = tf.constant(x, dtype='float32')
        loss, gvp = self._gvp_fn(x_in, v_in)
        return gvp.numpy().astype('float64')

    def _scipy_update_callback(self, x: np.ndarray):
        self._input_var.assign(x.astype('float32'))
        self._loss_per_iter.append(self._scipy_fun(x))
        self._iterations += 1

    def _hvp(self, x: tf.Tensor,
             v: tf.Tensor,
             objective_fn_this: Callable,
             return_loss: bool = True) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        if self._diag_hessian_fn is not None:
            diag_hessian = self._diag_hessian_fn(x)
            hvp = diag_hessian * v
            if return_loss:
                loss = self._full_objective_fn(x)
        else:
            loss, _, hvp = adh.hvp_forward_backward(objective_fn_this, x, v)
        if return_loss:
            return loss, hvp
        return hvp

    def _gvp_fn(self, x_in, v):
        with tf.GradientTape(persistent=True) as gt:
            gt.watch(x_in)
            with tf.GradientTape(persistent=True) as gt2:
                gt2.watch(x_in)
                prediction = self._predictions_fn(x_in)
                loss_old = self._loss_fn(prediction)
            dummy = tf.ones_like(prediction)
            gt.watch(dummy)
            inner_vjp = gt2.gradient(prediction, x_in, output_gradients=dummy)
        jvp = gt.gradient(inner_vjp, dummy, output_gradients=v)
        hjvp = self._hvp(prediction, jvp, self._loss_fn, False)
        jthjvp = gt2.gradient(prediction, x_in, output_gradients=hjvp)
        return loss_old, jthjvp

    def minimize(self, this_step_optimizer_args:dict = None):
        optimizer_args = self._optimizer_args.copy()
        if this_step_optimizer_args is not None:
            optimizer_args.update(this_step_optimizer_args)
        out = spopt.minimize(**optimizer_args)
        self._optimizer_results = out
        return out
