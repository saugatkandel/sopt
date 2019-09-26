#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sopt.examples.utils import getSampleObj
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from tensorflow.python.framework import graph_util
import sopt.benchmarks.ops.tensorflow.flops_registry_custom
from sopt.benchmarks.ops.tensorflow.graph_utils_custom import get_flops_for_sub_graph



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



def get_var_and_support(var, support, var_shape, support_shape):
    tf_var_reshaped = tf.reshape(var, [2, var_shape, var_shape])
    tf_var_cmplx = tf.complex(tf_var_reshaped[0], tf_var_reshaped[1])
    tf_var_padded = tf.pad(tf_var_cmplx, [[0,support_shape - var_shape], [0, support_shape - var_shape]])
    tf_var_and_support = tf_var_padded + support
    return tf_var_and_support



def predictions_fn(var, support, var_shape, support_shape):
    tf_var_and_support = get_var_and_support(var, support, var_shape, support_shape)
    tf_fft_mod = tf.abs(tf.fft2d(tf_var_and_support)) / support_shape
    return tf.reshape(tf_fft_mod, [-1])



def loss_fn(predictions, tf_diffraction_mod):
    return 0.5 * tf.reduce_sum((predictions - tf_diffraction_mod)**2)



image = getSampleObj(256, phase_range=np.pi)
ft_mod = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image), norm='ortho')))

var_shape = 100
sample_support = image.copy()
sample_support[:var_shape, :var_shape] = 0

plt.figure(figsize=[8,2])
figs = { 'Mod':np.abs(image), 'Phase':np.angle(image), 
        'sup_mod':np.abs(sample_support), 'sup_phase': np.angle(sample_support)}
for i, (key, val) in enumerate(figs.items()):
    plt.subplot(1,4,i+1)
    plt.pcolormesh(val, cmap='gray')
    plt.colorbar()
    plt.title(key)
    plt.axis('off')
plt.tight_layout()
plt.show()



init_weights = np.ones(2 * var_shape * var_shape)
support_shape = sample_support.shape[0]
var_shape, support_shape



# Forward model only
tf.reset_default_graph()
tf_var = tf.get_variable('var', dtype=tf.float32, shape=[2 * var_shape**2], initializer=tf.ones_initializer)
tf_support = tf.constant(sample_support, dtype=tf.complex64, name='support')
tf_diffraction_mod = tf.constant(np.fft.fftshift(ft_mod).flatten(), dtype=tf.float32, name='diffractions')

predictions_fn_this = lambda x: predictions_fn(x, tf_support, var_shape, support_shape)
loss_fn_this = lambda x: loss_fn(x, tf_diffraction_mod)

loss_tensor = loss_fn_this(predictions_fn_this(tf_var))
var_and_support_tensor = get_var_and_support(tf_var, tf_support, var_shape, support_shape)

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_fwd = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_fwd.total_float_ops)



# Forward model + gradients
tf.reset_default_graph()
tf_var = tf.get_variable('var', dtype=tf.float32, shape=[2 * var_shape**2], initializer=tf.ones_initializer)
tf_support = tf.constant(sample_support, dtype=tf.complex64, name='support')
tf_diffraction_mod = tf.constant(np.fft.fftshift(ft_mod).flatten(), dtype=tf.float32, name='diffractions')

predictions_fn_this = lambda x: predictions_fn(x, tf_support, var_shape, support_shape)
loss_fn_this = lambda x: loss_fn(x, tf_diffraction_mod)

loss_tensor = tf.identity(loss_fn_this(predictions_fn_this(tf_var)), name='loss_tensor')
var_and_support_tensor = get_var_and_support(tf_var, tf_support, var_shape, support_shape)

gradients = tf.gradients([loss_tensor], [tf_var])
gradients_tensor = tf.identity(gradients[0], name='gradients_tensor')

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_fwd_grad = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_fwd_grad.total_float_ops)



g = tf.get_default_graph()
graph_def = g.as_graph_def()
sub_graph_fwd = graph_util.extract_sub_graph(graph_def, ['loss_tensor'])
sub_graph_grad = graph_util.extract_sub_graph(graph_def, ['gradients_tensor'])



total_fwd_flops = get_flops_for_sub_graph(g, sub_graph_fwd)
total_grad_flops = get_flops_for_sub_graph(g, sub_graph_grad)
total_fwd_flops, total_grad_flops



# gauss-newton vector product
tf.reset_default_graph()
tf_var = tf.get_variable('var', dtype=tf.float32, shape=[2 * var_shape**2], initializer=tf.ones_initializer)

tf_support = tf.constant(sample_support, dtype=tf.complex64, name='support')
tf_diffraction_mod = tf.constant(np.fft.fftshift(ft_mod).flatten(), dtype=tf.float32, name='diffractions')

z = tf.get_variable('z',  initializer=tf.zeros_like(tf_var))
dummy_var = tf.get_variable('dummy', initializer=tf.zeros_like(tf_diffraction_mod))

predictions_fn_this = lambda x: predictions_fn(x, tf_support, var_shape, support_shape)
loss_fn_this = lambda x: loss_fn(x, tf_diffraction_mod)

predictions_tensor = predictions_fn_this(tf_var)
loss_tensor = tf.identity(loss_fn_this(predictions_tensor), name='loss_tensor')
var_and_support_tensor = get_var_and_support(tf_var, tf_support, var_shape, support_shape)

#jloss = tf.gradients(loss_tensor, predictions_tensor)

vjp_dummy = tf.gradients(predictions_tensor, tf_var, dummy_var)[0]
jvpz = tf.gradients(vjp_dummy, dummy_var, z)[0]

gvpz = tf.gradients(predictions_tensor, tf_var, jvpz)[0]
gvpz_tensor = tf.identity(gvpz, 'gvp_tensor')

session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_gvp = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_gvp.total_float_ops)



g = tf.get_default_graph()
graph_def = g.as_graph_def()
sub_graph_fwd = graph_util.extract_sub_graph(graph_def, ['loss_tensor'])
sub_graph_gvp = graph_util.extract_sub_graph(graph_def, ['gvp_tensor'])



total_fwd_flops = get_flops_for_sub_graph(g, sub_graph_fwd)
total_gvp_flops = get_flops_for_sub_graph(g, sub_graph_gvp)
total_fwd_flops, total_gvp_flops



# generalized gauss-newton vector product (with hessian-vector product)
tf.reset_default_graph()
tf_var = tf.get_variable('var', dtype=tf.float32, shape=[2 * var_shape**2], initializer=tf.ones_initializer)

tf_support = tf.constant(sample_support, dtype=tf.complex64, name='support')
tf_diffraction_mod = tf.constant(np.fft.fftshift(ft_mod).flatten(), dtype=tf.float32, name='diffractions')

z = tf.get_variable('z',  initializer=tf.zeros_like(tf_var))
dummy_var = tf.get_variable('dummy', initializer=tf.zeros_like(tf_diffraction_mod))

predictions_fn_this = lambda x: predictions_fn(x, tf_support, var_shape, support_shape)
loss_fn_this = lambda x: loss_fn(x, tf_diffraction_mod)

predictions_tensor = predictions_fn_this(tf_var)
loss_tensor = tf.identity(loss_fn_this(predictions_tensor), name='loss_tensor')
var_and_support_tensor = get_var_and_support(tf_var, tf_support, var_shape, support_shape)

#jloss = tf.gradients(loss_tensor, predictions_tensor)

vjp_dummy = tf.gradients(predictions_tensor, tf_var, dummy_var)[0]
jvpz = tf.gradients(vjp_dummy, dummy_var, z)[0]

hjvpz = _hessian_vector_product([loss_tensor], [predictions_tensor], [jvpz])

gvpz = tf.gradients(predictions_tensor, tf_var, hjvpz)[0]
gvpz_tensor = tf.identity(gvpz, name='gvp_tensor')
session = tf.Session()
session.run(tf.global_variables_initializer())

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_gvp = tf.profiler.profile(run_meta=run_meta, cmd='graph', options=opts) 
print(flops_gvp.total_float_ops)



g = tf.get_default_graph()
graph_def = g.as_graph_def()
sub_graph_fwd = graph_util.extract_sub_graph(graph_def, ['loss_tensor'])
sub_graph_gvp = graph_util.extract_sub_graph(graph_def, ['gvp_tensor'])



total_fwd_flops = get_flops_for_sub_graph(g, sub_graph_fwd)
total_gvp_flops = get_flops_for_sub_graph(g, sub_graph_gvp)
total_fwd_flops, total_gvp_flops

