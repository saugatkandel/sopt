#Author - Saugat Kandel
# coding: utf-8


import tensorflow as tf
from tensorflow.linalg import LinearOperator
from typing import Callable, Optional, NamedTuple



class MatrixFreeLinearOp(LinearOperator):
    def __init__(self, 
                 operator: Callable[[tf.Tensor], tf.Tensor],
                 shape: tf.TensorShape) -> None:
        self._operator = operator
        self._op_shape = shape
        super().__init__(dtype=tf.float32)
    
    def _matvec(self,
                x: tf.Tensor,
                adjoint: Optional[bool] = False) -> tf.Tensor:
        return self._operator(x)
    
    def _shape(self) -> tf.TensorShape:
        return self._op_shape
        



# This is adapted from 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/solvers/python/ops/linear_equations.py
def conjugate_gradient(operator: LinearOperator,
                       rhs: tf.Tensor,
                       x: Optional[tf.Tensor] = None,
                       preconditioner: Optional[LinearOperator] = None,
                       tol: float = 1e-4,
                       max_iter: int = 20,
                       name: str = "conjugate_gradient") -> NamedTuple:
    r"""Conjugate gradient solver.
    Solves a linear system of equations `A*x = rhs` for selfadjoint, positive
    definite matrix `A` and right-hand side vector `rhs`, using an iterative,
    matrix-free algorithm where the action of the matrix A is represented by
    `operator`. The iteration terminates when either the number of iterations
    exceeds `max_iter` or when the residual norm has been reduced to `tol`
    times its initial value, i.e. \\(||rhs - A x_k|| <= tol ||rhs||\\).
    Args:
    operator: An object representing a linear operator with attributes:
      - shape: Either a list of integers or a 1-D `Tensor` of type `int32` of
        length 2. `shape[0]` is the dimension on the domain of the operator,
        `shape[1]` is the dimension of the co-domain of the operator. On other
        words, if operator represents an N x N matrix A, `shape` must contain
        `[N, N]`.
      - dtype: The datatype of input to and output from `apply`.
      - apply: Callable object taking a vector `x` as input and returning a
        vector with the result of applying the operator to `x`, i.e. if
       `operator` represents matrix `A`, `apply` should return `A * x`.
    rhs: A rank-1 `Tensor` of shape `[N]` containing the right-hand size vector.
    preconditioner: An object representing a linear operator, see `operator`
      for detail. The preconditioner should approximate the inverse of `A`.
      An efficient preconditioner could dramatically improve the rate of
      convergence. If `preconditioner` represents matrix `M`(`M` approximates
      `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
      `A^{-1}x`. For this to be useful, the cost of applying `M` should be
      much lower than computing `A^{-1}` directly.
    x: A rank-1 `Tensor` of shape `[N]` containing the initial guess for the
      solution.
    tol: A float scalar convergence tolerance.
    max_iter: An integer giving the maximum number of iterations.
    name: A name scope for the operation.
    Returns:
    output: A namedtuple representing the final state with fields:
      - i: A scalar `int32` `Tensor`. Number of iterations executed.
      - x: A rank-1 `Tensor` of shape `[N]` containing the computed solution.
      - r: A rank-1 `Tensor` of shape `[M]` containing the residual vector.
      - p: A rank-1 `Tensor` of shape `[N]`. `A`-conjugate basis vector.
      - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
        `preconditioner=None`.
    """
    # ephemeral class holding CG state.
    class CGState(NamedTuple):
        i: tf.Tensor
        x: tf.Tensor
        r: tf.Tensor
        p: tf.Tensor
        gamma: tf.Tensor
    
    def stopping_criterion(state):
        with tf.name_scope('cg_cond'):
            output = tf.linalg.norm(state.r) > tol
        return output

    def cg_step(state):  # pylint: disable=missing-docstring
        with tf.name_scope('cg_body'):
            z = operator.matvec(state.p)
            alpha = state.gamma / tf.tensordot(state.p, z, 1)
            x = state.x + alpha * state.p
            r = state.r - alpha * z
            if preconditioner is None:
                gamma = tf.tensordot(r,r,1)
                beta = gamma / state.gamma
                p = r + beta * state.p
            else:
                q = preconditioner.matvec(r)
                gamma = tf.tensordot(r,q,1)
                beta = gamma / state.gamma
                p = q + beta * state.p
            output = CGState(i=state.i + 1, x=x, r=r, p=p, gamma=gamma)
        return output

    with tf.name_scope(name):
        if x is None:
            x = tf.zeros_like(rhs)
            r0 = rhs
        else:
            r0 = rhs - operator.matvec(x)
        if preconditioner is None:
            p0 = r0
        else:
            p0 = preconditioner.matvec(r0)
        gamma0 = tf.tensordot(r0, p0, 1)
        tol *= tf.linalg.norm(r0)
        state = CGState(i=tf.constant(0, dtype=tf.int32), x=x, r=r0, p=p0, gamma=gamma0)
        state = tf.while_loop(stopping_criterion, cg_step, 
                              [state], maximum_iterations=max_iter,
                              back_prop=False,
                              name='cg_while')
        return CGState(
            state.i,
            x=tf.squeeze(state.x),
            r=tf.squeeze(state.r),
            p=tf.squeeze(state.p),
            gamma=state.gamma)



def linear_cg_solve(linear_op, b, x_init, tol=1e-7, maxiter=None):
    """
    This is functionally identical to the conjugate_gradient method,
    except without the option to use the preconditioner.
    
    Source:
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



# Might be buggy
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

