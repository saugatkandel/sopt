#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.framework import graph_util
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from optimizers.tensorflow.curveball import Curveball
from optimizers.tensorflow.lma import LMA



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph



def tf_y_pred(z, tf_affine_transform):
    return tf.reshape(tf_affine_transform @ tf.reshape(z, [3, -1]), [-1])
def tf_loss(y_pred, tf_y_true):
    return 0.5 * tf.reduce_sum((tf_y_true - y_pred)**2)



z_true = np.random.randn(3,100).astype('float32')

random_mat = np.random.randn(3,3)
random_symmetric_mat = random_mat + random_mat.T
evals, evecs = np.linalg.eig(random_symmetric_mat)
affine_transform = evecs

y_true = affine_transform @ z_true
y_true_flat = y_true.flatten()

z_guess = np.random.randn(300).astype('float32')



# Reference:
# https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model

# See more documentation at 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md



# Forward model
tf.reset_default_graph()
var = tf.get_variable('var', dtype=tf.float32, initializer=z_guess)

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform')

preds = tf_y_pred(var, tf_affine_transform)
loss_tensor = tf_loss(preds, tf_y_true)    

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_fwd = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_fwd.total_float_ops)



# Forward model + gradients
tf.reset_default_graph()
var = tf.get_variable('var', dtype=tf.float32, initializer=z_guess)

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform')

preds = tf_y_pred(var, tf_affine_transform)
loss_tensor = tf_loss(preds, tf_y_true)  

gradients = tf.gradients([loss_tensor], [var])

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_fwd_grad = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_fwd_grad.total_float_ops)



# gauss-newton vector product
tf.reset_default_graph()
var = tf.get_variable('var', dtype=tf.float32, initializer=z_guess)
z = tf.get_variable('z', dtype=tf.float32, initializer=tf.zeros_like(z_guess, dtype='float32'))
dummy_var = tf.get_variable('dummy', dtype=tf.float32, initializer=tf.zeros_like(y_true_flat, dtype='float32'))

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform')

preds = tf_y_pred(var, tf_affine_transform)
loss_tensor = tf_loss(preds, tf_y_true)  

#jloss = tf.gradients(loss_tensor, preds)

vjp_dummy = tf.gradients(preds, var,dummy_var)[0]
jvpz = tf.gradients(vjp_dummy, dummy_var, z)[0]

gvpz = tf.gradients(preds, var, jvpz)[0]

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_gvp = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_gvp.total_float_ops)



# generalized gauss-newton vector product (with hessian-vector product)
tf.reset_default_graph()
var = tf.get_variable('var', dtype=tf.float32, initializer=z_guess)
z = tf.get_variable('z', dtype=tf.float32, initializer=tf.zeros_like(z_guess, dtype='float32'))
dummy_var = tf.get_variable('dummy', dtype=tf.float32, initializer=tf.zeros_like(y_true_flat, dtype='float32'))

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform')

preds = tf_y_pred(var, tf_affine_transform)
loss_tensor = tf_loss(preds, tf_y_true)  

#jloss = tf.gradients(loss_tensor, preds)

vjp_dummy = tf.gradients(preds, var,dummy_var)[0]
jvpz = tf.gradients(vjp_dummy, dummy_var, z)[0]

hjvpz = _hessian_vector_product([loss_tensor], [preds], [jvpz])

gvpz = tf.gradients(preds, var, hjvpz)[0]

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_gvp = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_gvp.total_float_ops)



# curveball (without hessian-vector-product)
tf.reset_default_graph()
var = tf.get_variable('var', dtype=tf.float32, initializer=z_guess)

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform') + 1.1231

preds_fn = lambda x: tf_y_pred(x, tf_affine_transform)
loss_fn = lambda x: tf_loss(x, tf_y_true)

optimizer = Curveball(var, predictions_fn=preds_fn, loss_fn=loss_fn, squared_loss=True, name='opt')
minimize_op = optimizer.minimize()

session = tf.Session()

session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_gvp = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_gvp.total_float_ops)
print("Since the number of ops required for the matrix inversion step is not fixed, "
      +"this number changes depending on the matrix values (by < 100)")



# LMA (without hessian-vector-product)
tf.reset_default_graph()
var = tf.get_variable('var', dtype=tf.float32, initializer=z_guess)

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform')

preds_fn = lambda x: tf_y_pred(x, tf_affine_transform)
loss_fn = lambda x: tf_loss(x, tf_y_true)

optimizer = LMA(var, predictions_fn=preds_fn, loss_fn=loss_fn, squared_loss=True, 
                name='opt', max_cg_iter=10)
minimize_op = optimizer.minimize()

session = tf.Session()

session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_gvp = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_gvp.total_float_ops)
print("Note that this number includes all the conjugate gradient iterations required for the completion of this step. "
      + "This usually increases closer to the minimum")

