#Author - Saugat Kandel
# coding: utf-8


import six
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.profiler.internal import flops_registry



# This script adapts the flops counter from the tensorflow profiler to ensure that
# it correctly estimates the flops reequired for complex-valued ops.
# 
# NOTE:
# I am only adapting the flops counter for a subset of the ops implemented in Tensorflow,
# i.e. only the ops I actually use with complex numbers.
#
# Adapted from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/internal/flops_registry.py
# and
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py



class RegisterStatistics(object):
    """A decorator for registering the statistics function for an op type.
    This decorator can be defined for an op type so that it gives a
    report on the resources used by an instance of an operator, in the
    form of an OpStats object.
    Well-known types of statistics include these so far:
    - flops: When running a graph, the bulk of the computation happens doing
    numerical calculations like matrix multiplications. This type allows a node
    to return how many floating-point operations it takes to complete. The
    total number of FLOPs for a graph is a good guide to its expected latency.
    You can add your own statistics just by picking a new type string, registering
    functions for the ops you care about, and then calling get_stats_for_node_def.
    If a statistic for an op is registered multiple times, a KeyError will be
    raised.
    Since the statistics is counted on a per-op basis. It is not suitable for
    model parameters (capacity), which is expected to be counted only once, even
    if it is shared by multiple ops. (e.g. RNN)
    For example, you can define a new metric called doohickey for a Foo operation
    by placing this in your code:
    ```python
    @ops.RegisterStatistics("Foo", "doohickey")
    def _calc_foo_bojangles(unused_graph, unused_node_def):
    return ops.OpStats("doohickey", 20)
    ```
    Then in client code you can retrieve the value by making this call:
    ```python
    doohickey = ops.get_stats_for_node_def(graph, node_def, "doohickey")
    ```
    If the NodeDef is for an op with a registered doohickey function, you'll get
    back the calculated amount in doohickey.value, or None if it's not defined.
    """

    def __init__(self, op_type: str,
                 statistic_type: str):
        """Saves the `op_type` as the `Operation` type."""
        if not isinstance(op_type, six.string_types):
            raise TypeError("op_type must be a string.")
        if "," in op_type:
            raise TypeError("op_type must not contain a comma.")
        self._op_type = op_type
        if not isinstance(statistic_type, six.string_types):
            raise TypeError("statistic_type must be a string.")
        if "," in statistic_type:
            raise TypeError("statistic_type must not contain a comma.")
        self._statistic_type = statistic_type

    def __call__(self, f):
        """Registers "f" as the statistics function for "op_type".
        
        If the "op_type" already exists in the registry, 
        then replace the flops counter for that "op_type"."""
        op_str = self._op_type + "," + self._statistic_type
        if op_str in ops._stats_registry._registry:
            del ops._stats_registry._registry[op_str]
        ops._stats_registry.register(f, op_str)
        return f



@RegisterStatistics("Square", "flops")
def _square_flops(graph, node):
    """Compute flops for Square operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 6
    else:
        ops_per_element = 1
    return flops_registry._unary_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("Reciprocal", "flops")
def _reciprocal_flops(graph, node):
    """Compute flops for Reciprocal operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 6
    else:
        ops_per_element = 1
    return flops_registry._unary_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("Neg", "flops")
def _neg_flops(graph, node):
    """Compute flops for Neg operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 2
    else:
        ops_per_element = 1
    return flops_registry._unary_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("AssignSub", "flops")
def _assign_sub_flops(graph, node):
    """Compute flops for AssignSub operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 2
    else:
        ops_per_element = 1
    return flops_registry._unary_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("AssignAdd", "flops")
def _assign_add_flops(graph, node):
    """Compute flops for AssignAdd operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 2
    else:
        ops_per_element = 1
    return flops_registry._unary_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("Conj", "flops")
def _conj_flops(graph, node):
    """Compute flops for Conj operation."""
    return flops_registry._unary_op_flops(graph, node)



@RegisterStatistics("Abs", "flops")
def _abs_flops(graph, node):
    """Compute flops for Abs operation."""
    # mul, sqrt
    return flops_registry._unary_op_flops(graph, node, ops_per_element=2)



@RegisterStatistics("ComplexAbs", "flops")
def _complex_abs_flops(graph, node):
    """Compute flops for Abs operation."""
    # conj, mul, sqrt
    return flops_registry._unary_op_flops(graph, node, ops_per_element=8)



################################################################################
# Binary operations
################################################################################



@RegisterStatistics("Add", "flops")
def _add_flops(graph, node):
    """Compute flops for Add operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 2
    else:
        ops_per_element = 1
    return flops_registry._binary_per_element_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("Sub", "flops")
def _sub_flops(graph, node):
    """Compute flops for Sub operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 2
    else:
        ops_per_element = 1
    return flops_registry._binary_per_element_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("Mul", "flops")
def _mul_flops(graph, node):
    """Compute flops for Mul operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 6
    else:
        ops_per_element = 1
    return flops_registry._binary_per_element_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("RealDiv", "flops")
def _real_div_flops(graph, node):
    """Compute flops for RealDiv operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 6
    else:
        ops_per_element = 1
    return flops_registry._binary_per_element_op_flops(graph, node, ops_per_element=ops_per_element)



@RegisterStatistics("Pow", "flops")
def _pow_flops(graph, node):
    """Compute flops for Pow operation."""
    if node.attr['T'].type == tf.complex64:
        ops_per_element = 6
    else:
        ops_per_element = 1
    return flops_registry._binary_per_element_op_flops(graph, node, ops_per_element=ops_per_element)



################################################################################
# Reduction ops
################################################################################



@RegisterStatistics("Mean", "flops")
def _mean_flops(graph, node):
    """Compute flops for Mean operation."""
    # reduction - sum, finalization - divide
    if node.attr['T'].type == tf.complex64:
        reduce_flops = 2
        finalize_flops = 2
    else:
        reduce_flops = 1
        finalize_flops = 1
    return flops_registry._reduction_op_flops(graph, node, reduce_flops=reduce_flops, finalize_flops=finalize_flops)



@RegisterStatistics("Sum", "flops")
def _sum_flops(graph, node):
    """Compute flops for Sum operation."""
    # reduction - sum, no finalization
    if node.attr['T'].type == tf.complex64:
        reduce_flops = 2
        finalize_flops = 0
    else:
        reduce_flops = 1
        finalize_flops = 0
    return flops_registry._reduction_op_flops(graph, node, reduce_flops=reduce_flops, finalize_flops=finalize_flops)



@RegisterStatistics("Prod", "flops")
def _prod_flops(graph, node):
    """Compute flops for Prod operation."""
    # reduction - sum, no finalization
    if node.attr['T'].type == tf.complex64:
        reduce_flops = 6
        finalize_flops = 0
    else:
        reduce_flops = 1
        finalize_flops = 0
    return flops_registry._reduction_op_flops(graph, node, reduce_flops=reduce_flops, finalize_flops=finalize_flops)



@RegisterStatistics("BiasAddGrad", "flops")
def _bias_add_grad_flops(graph, node):
    """Compute flops for BiasAddGrad operation."""
    # Implementation of BiasAddGrad, essentially it's a reduce sum and reshaping:
    # So computing flops same way as for "Sum"
    if node.attr['T'].type == tf.complex64:
        reduce_flops = 2
        finalize_flops = 0
    else:
        reduce_flops = 1
        finalize_flops = 0
    return flops_registry._reduction_op_flops(graph, node, reduce_flops=reduce_flops, finalize_flops=finalize_flops)



@RegisterStatistics("AddN", "flops")
def _add_n_flops(graph, node):
    """Compute flops for AddN operation."""
    if not node.input:
        return _zero_flops(graph, node)
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    if node.attr['T'].type == tf.complex64:
        flops_per_element = 2
    else:
        flops_per_element = 1
    return ops.OpStats("flops", in_shape.num_elements() * flops_per_element * (len(node.input) - 1))



@RegisterStatistics("FFT2D", "flops")
def _fft_2d_flops(graph, node):
    """Compute flops for fft2d operation.
    
    The radix-2 Cooley-Tukey algorithm asymptotically requires 5 N log2(N) floating-point operations.
    I am using this value as the flops estimate.
    
    Source:
    http://www.fftw.org/speed/method.html
    """
    if not node.input:
        return _zero_flops(graph, node)
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    n = in_shape.num_elements()
    num_ops = np.int_(np.ceil(5 * n * np.log2(n)))
    return ops.OpStats("flops", num_ops)



@RegisterStatistics("IFFT2D", "flops")
def _ifft_2d_flops(graph, node):
    """Compute flops for ifft2d operation.
    
    Using same value as in fft2d"""
    if not node.input:
        return _zero_flops(graph, node)
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    n = in_shape.num_elements()
    num_ops = np.int_(np.ceil(5 * n * np.log2(n)))
    return ops.OpStats("flops", num_ops)

