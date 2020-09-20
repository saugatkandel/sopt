from tensorflow.linalg import LinearOperator
import tensorflow as tf
from typing import Callable, Optional, NamedTuple

__all__ = ['MatrixFreeLinearOp', 'conjugate_gradient']


class MatrixFreeLinearOp(LinearOperator):
    def __init__(self,
                 operator: Callable[[tf.Tensor], tf.Tensor],
                 shape: tf.TensorShape) -> None:
        self._operator = operator
        self._op_shape = shape
        super().__init__(dtype=tf.float32, is_self_adjoint=True, is_positive_definite=True)

    def _matvec(self,
                x: tf.Tensor,
                adjoint: Optional[bool] = False) -> tf.Tensor:
        return self._operator(x)

    def _shape(self) -> tf.TensorShape:
        return self._op_shape

    def _shape_tensor(self) -> None:
        pass

    def _matmul(self) -> None:
        pass


# ephemeral class holding CG state.
class CGState(NamedTuple):
    i: tf.Tensor
    x: tf.Tensor
    r: tf.Tensor
    p: tf.Tensor
    gamma: tf.Tensor
    r_check: tf.Tensor


# This is adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/solvers/python/ops/linear_equations.py
@tf.function
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

    Notes
    ------
    We expect that the r vectors obtained should be perpendicular to each other at every step. If they are not,
    then we have an issue with numerical stability. This seems to happen in the edge case when I am trying to calculate
    the LM directions for both the probe and the object in the blind ptychography case---I have no idea why.

    As a hack around this edge case, I am setting a stopping condition that depends on the direction of r in consecutive
    iterations.

    References
    ----------
    [1] Section 10.24 from https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/chapter10.pdf

    """

    def stopping_criterion(state: CGState) -> tf.Tensor:
        with tf.name_scope('cg_cond'):
            output = tf.linalg.norm(state.r) > tol
        return output

    def cg_step(state: CGState) -> CGState:  # pylint: disable=missing-docstring
        with tf.name_scope('cg_body'):
            z = operator.matvec(state.p)
            alpha = state.gamma / tf.tensordot(state.p, z, 1)
            x = state.x + alpha * state.p
            r = state.r - alpha * z
            r_check = tf.abs(tf.tensordot(state.r, r, 1)) / (tf.linalg.norm(state.r) * tf.linalg.norm(r))
            if preconditioner is None:
                gamma = tf.tensordot(r, r, 1)
                beta = gamma / state.gamma
                p = r + beta * state.p
            else:
                q = preconditioner.matvec(r)
                gamma = tf.tensordot(r, q, 1)
                beta = gamma / state.gamma
                p = q + beta * state.p

            output = CGState(i=state.i + 1, x=x,
                             # tf.debugging.check_numerics(x, message='Invalid x in CG iterations.'),
                             r=r,  # tf.debugging.check_numerics(r, message='Invalid r in CG iterations.'),
                             p=p,  # tf.debugging.check_numerics(p, message='Invalid p in CG iterations.'),
                             gamma=gamma,
                             r_check=r_check)
        return [output]

    with tf.name_scope(name):
        with tf.name_scope('cg_init'):
            if x is None:
                x = tf.zeros_like(rhs)
                # r0 = tf.debugging.check_numerics(rhs, 'input rhs invalid')
                r0 = rhs
            else:
                r0 = rhs - operator.matvec(x)
            if preconditioner is None:
                p0 = r0
            else:
                p0 = preconditioner.matvec(r0)
            # gamma0 = tf.reduce_sum(r0 * p0)#
            gamma0 = tf.tensordot(r0, p0, 1)
            tol *= tf.linalg.norm(rhs)  # r0)
            r_check0 = 0.
        state = CGState(i=tf.constant(0, dtype=tf.int32), x=x, r=r0,
                        p=p0, gamma=gamma0, r_check=r_check0)
        [state] = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond=stopping_criterion, body=cg_step,
                                                                        loop_vars=[state], maximum_iterations=max_iter,
                                                                        name='cg_while'))
        return CGState(
            state.i,
            x=tf.squeeze(state.x),
            r=tf.squeeze(state.r),
            p=tf.squeeze(state.p),
            gamma=state.gamma,
            r_check=state.r_check)