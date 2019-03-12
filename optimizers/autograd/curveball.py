#Author - Saugat Kandel
# coding: utf-8


from autograd import numpy as np
import autograd as ag
from typing import Callable, List



## IN CONSTRUCTION
class Curveball(object):
    """Adapted from:
    https://github.com/jotaf98/curveball
    """
    def __init__(self, 
                 input_variable: np.ndarray,
                 loss_input_fn: Callable[[np.ndarray], np.ndarray],
                 loss_fn: Callable[[np.ndarray], float],
                 damping_factor: float = 1.0, 
                 damping_update_factor: float = 0.999,
                 damping_update_frequency: int = 5,
                 alpha_init: float = 1.0,
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
        
        from 
        The alpha should generally just be 1.0 and doesn't change. 
        The beta and rho values are updated at each cycle, so there is no intial value."""
        
        self._input_var = input_variable
        self._loss_input_fn = loss_input_fn
        self._loss_fn = loss_fn
        
        # Multiplicating factor to update the damping factor at the end of each cycle
        self._damping_factor = damping_factor
        self._damping_update_factor = damping_update_factor
        self._damping_update_frequency = damping_update_frequency
        self._squared_loss = squared_loss
        self._alpha = alpha_init
        
        self._z = np.zeros_like(self._input_var)
        self._iteration = 0
        
        self._vjp = ag.make_vjp(self._loss_input_fn)
        self._jvp = ag.differential_operators.make_jvp_reversemode(self._loss_input_fn)
        
        self._grad = ag.grad(self._loss_fn)
        
        if self._squared_loss:
            self._hjvp = self._jvp
        else:
            self._hjvp = ag.differential_operators.make_hvp(self._loss_fn)
        
        self._iteration = 0


    def _matrix_vector_updates(self) -> List[float]:   
        vjp_fun_this, loss_input_array = self._vjp(self._input_var) 
        jvp_fun_this = self._jvp(self._input_var)
        
        loss_before_update = self._loss_fn(loss_input_array)
        
        if self._squared_loss: 
            hjvp_fun_this = lambda x: x#jvp_fun_this
            jloss = self._grad(loss_input_array)
        else:
            hjvp_fun_this, jloss = self._hjvp(loss_input_array)
        
        jvpz = jvp_fun_this(self._z)
        hjvpz = hjvp_fun_this(jvpz)
        
        jhjvpz = vjp_fun_this(hjvpz + jloss)
        
        deltaz = jhjvpz + self._damping_factor * self._z  
        jvpdz = jvp_fun_this(deltaz)
        
        if self._squared_loss:
            hjvpdz = jvpdz
        else:
            hjvpdz = hjvp_fun_this(jvpdz)
            
        a11 = np.sum(hjvpdz * jvpdz)
        a12 = np.sum(jvpz * hjvpdz)
        a22 = np.sum(jvpz * hjvpz)
        
        b1 = np.sum(jloss * jvpdz)
        b2 = np.sum(jloss * jvpz)
        
        a11 = a11 + np.sum(deltaz * deltaz) * self._damping_factor
        a12 = a12 + np.sum(deltaz * self._z) * self._damping_factor
        a22 = a22 + np.sum(self._z * self._z) * self._damping_factor
        
        A = np.array([[a11, a12],[a12, a22]])
        b = np.array([b1, b2])
        
        m_b = np.linalg.pinv(A) @ b
        
        beta = m_b[0]
        rho = -m_b[1]
        
        M = -0.5 * m_b @ b
        
        self._z = rho * self._z - beta * deltaz
        self._input_var = self._input_var + self._alpha * self._z
        return loss_before_update, M
    
    def _damping_update(self, 
                        loss_before_update: float,
                        expected_quadratic_change: float,
                        update_threshold_low: float = 0.5, 
                        update_threshold_high: float = 1.5,
                        damping_threshold_low: float = 1e-7,
                        damping_threshold_high: float = 1e7) -> None:
        
        loss_after_update = self._loss_fn(self._loss_input_fn(self._input_var))
        actual_loss_change = loss_after_update - loss_before_update
        gamma = actual_loss_change / expected_quadratic_change
        
        if gamma < update_threshold_low:
            self._damping_factor = self._damping_factor / self._damping_update_factor
        elif gamma > update_threshold_high:
            self._damping_factor = self._damping_factor * self._damping_update_factor
        
        self._damping_factor = np.clip(self._damping_factor, 
                                       a_min=damping_threshold_low, 
                                       a_max=damping_threshold_high)
    
    def minimize(self) -> np.ndarray:
        loss_before_update, M = self._matrix_vector_updates()
        
        self._iteration += 1
        if self._iteration % self._damping_update_frequency == 0:
            self._damping_update(loss_before_update, M)
        
        return self._input_var

