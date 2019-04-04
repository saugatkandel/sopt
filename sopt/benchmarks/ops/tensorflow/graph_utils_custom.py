#Author - Saugat Kandel
# coding: utf-8


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
import benchmarks.ops.tensorflow.flops_registry_custom
import tensorflow as tf
from typing import List



def get_flops_for_node_list(g: tf.Graph,
                            nodes_list: List[tf.NodeDef]) -> int:
    total_flops = 0
    for node in nodes_list:
        try:
            stats = ops.get_stats_for_node_def(g, node, 'flops')
        except ValueError:
            stats = None
        if stats and stats.value:
            total_flops += int(stats.value)
    return total_flops
        



def get_flops_for_sub_graph(g: tf.Graph,
                            sub_graph_def: tf.GraphDef) -> int:
    warning = """
    WARNING: 
    For gradient calculations, I don't think the number generated here 
    reflects the true cost of the grad calculation.
    This gives a lower number than what I think should be the actual cost.
    """
    print(warning)
    return get_flops_for_node_list(g, sub_graph_def.node)

