#Author - Saugat Kandel
# coding: utf-8


from autograd import numpy as np
import autograd as ag
from scipy.sparse.linalg import LinearOperator
import scipy
from typing import Callable, List



class LMA(object):
    """The Levenberg-Marquardt algorithm
    """
    def __init__(self, 
                 input_variable: np.ndarray,
                 loss_input_fn: Callable[[np.ndarray], np.ndarray],
                 loss_fn: Callable[[np.ndarray], float],
                 damping_factor: float = 1.0, 
                 damping_update_factor: float = 2/3,
                 update_cond_threshold_low: float = 0.25, # following the least squares book
                 update_cond_threshold_high: float = 0.75,
                 damping_threshold_low = 1e-7,
                 damping_threshold_high = 1e7,
                 max_cg_iter: int = 100,
                 cg_tol: float = 1e-5,
                 squared_loss: bool = True) -> None:
        """
        Paramters:
        input_var: 
        1-d numpy array. Flatten before passing, if necessary.
        
        loss_input_fn: 
        Function that takes in input_var as the only input parameter.
        This should output a 1-d array, that is then passed to loss_fn for the 
        actual loss calculation.  
        Separating the loss calculation into a two step process this way 
        simplifies the second order calculations.
        
        loss_fn:
        Function that takes in the output of loss_input_fn and 
        calculates the singular loss value.
        """
        
        self._input_var = input_variable
        self._loss_input_fn = loss_input_fn
        self._loss_fn = loss_fn
        
        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_factor = damping_factor
        self._damping_update_factor = damping_update_factor
        self._update_cond_threshold_low = update_cond_threshold_low
        self._update_cond_threshold_high =  update_cond_threshold_high
        self._damping_threshold_low = damping_threshold_low
        self._damping_threshold_high = damping_threshold_high
        self._max_cg_iter = max_cg_iter
        self._cg_tol = cg_tol
        self._squared_loss = squared_loss
        
        # variable used for the updates
        self._update_var = np.zeros_like(self._input_var)
        
        self._vjp = ag.make_vjp(self._loss_input_fn)
        self._jvp = ag.differential_operators.make_jvp_reversemode(self._loss_input_fn)
        
        self._grad = ag.grad(self._loss_fn)
        
        if self._squared_loss:
            self._hjvp = self._jvp
        else:
            self._hjvp = ag.differential_operators.make_hvp(self._loss_fn)

    def _matrix_vector_operators(self) -> List[float]:   
        vjp_fun_this, predictions_array = self._vjp(self._input_var) 
        jvp_fun_this = self._jvp(self._input_var)
        
        loss_before_update = self._loss_fn(predictions_array)
        
        if self._squared_loss: 
            hjvp_fun_this = lambda x: x#jvp_fun_this
            jloss = self._grad(predictions_array)
        else:
            hjvp_fun_this, jloss = self._hjvp(predictions_array)
        
        return vjp_fun_this, hjvp_fun_this, jloss, loss_before_update
    
    def minimize(self):
        
        [
            vjp_fun_this, 
            hjvp_fun_this,
            jloss, 
            loss_before_update 
        ] = self._matrix_vector_operators()
        
        linear_b = vjp_fun_this(jloss)
        
        while True:
            linear_ax = lambda h: vjp_fun_this(hjvp_fun_this(h)) + self._damping_factor * h
            
            # I am planning on trying out both the scipy linear solver 
            # and my own conjugate gradient solver.
            # For the initial guess, I am following Marten's recipe
            # i.e., using the solution from the previous run.
            
            A = LinearOperator((linear_b.size, linear_b.size), matvec=linear_ax)
            opt_out = scipy.sparse.linalg.cg(A, linear_b, tol=self._cg_tol, 
                                             x0=self._update_var,
                                             maxiter=self._max_cg_iter)
            if opt_out[1] < 0:
                raise ValueError("Linear system not correctly solved")
            
            update_this = opt_out[0]
            x_new = self._input_var + update_this
            
            loss_new = self._loss_fn(self._predictions_fn(x_new))
            loss_change = loss_new - loss_before_update
            expected_quadratic_change = np.dot(update_this, self._damping_factor * update_this + linear_b)
            reduction_ratio = loss_change / expected_quadratic_change
            
            if reduction_ratio > self._update_cond_threshold_high:
                self._damping_factor *= self._damping_update_factor
            elif reduction_ratio < self._update_cond_threshold_low:
                self._damping_factor *= 1 / self._damping_update_factor 
            
            if reduction_ratio > 0:
                self._update_var = update_this
                self._input_var = x_new
                break      
        return self._input_var
        
        

