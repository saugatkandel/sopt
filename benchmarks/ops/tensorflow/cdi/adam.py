#Author - Saugat Kandel
# coding: utf-8


# Calculates the number of flops required to minimize a simple CDI problem using the Adam optimizer.
# This serves as a comparison to the Curveball and LMA methods
# However, we should note that we have not optimized the hyperparameters either here or for the
# curveball and lma cases. So this is not a rigorous comparison.



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from examples.utils import getSampleObj
import benchmarks.ops.tensorflow.flops_registry_custom
from tensorflow.python.framework import graph_util
from benchmarks.ops.tensorflow.graph_utils_custom import get_flops_for_sub_graph, get_flops_for_node_list



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



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



tf.reset_default_graph()
with tf.name_scope("initialize"):
    tf_var = tf.get_variable('var', dtype=tf.float32, shape=[2 * var_shape**2], initializer=tf.ones_initializer)
    tf_support = tf.constant(sample_support, dtype=tf.complex64, name='support')
    tf_diffraction_mod = tf.constant(np.fft.fftshift(ft_mod).flatten(), dtype=tf.float32, name='diffractions')

def get_var_and_support(var):
    tf_var_reshaped = tf.reshape(var, [2, var_shape, var_shape])
    tf_var_cmplx = tf.complex(tf_var_reshaped[0], tf_var_reshaped[1])
    tf_var_padded = tf.pad(tf_var_cmplx, [[0,support_shape - var_shape], [0, support_shape - var_shape]])
    tf_var_and_support = tf_var_padded + tf_support
    return tf_var_and_support

def predictions_fn(var):
    with tf.name_scope("predictions"):
        tf_var_and_support = get_var_and_support(var)
        fft_step = tf.fft2d(tf_var_and_support, name='fft_step')
        tf_fft_mod = tf.abs(fft_step) / support_shape
    return tf.reshape(tf_fft_mod, [-1])

def loss_fn(predictions):
    with tf.name_scope("loss"):
        loss = 0.5 * tf.reduce_sum((predictions - tf_diffraction_mod)**2)
    return loss


predictions_tensor = predictions_fn(tf_var)
loss_fn_tensor = loss_fn(predictions_tensor)
loss_fn_tensor_identity = tf.identity(loss_fn_tensor, name='loss_tensor')

adam_opt = tf.train.AdamOptimizer(1e-1)
min_op = adam_opt.minimize(loss_fn_tensor, var_list=[tf_var], name='adam')

session = tf.Session()
session.run(tf.global_variables_initializer())



g = tf.get_default_graph()
graph_def = g.as_graph_def()
g.finalize()



run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops_adam_and_fwd = tf.profiler.profile(run_meta=run_meta, cmd='scope', options=opts) 
flops_adam_and_fwd = flops_adam_and_fwd.total_float_ops
# In this case, since we supply the loss tensor to adam, the value here is the *correct* value
print(flops_adam_and_fwd)



session.run(loss_fn_tensor)



get_ipython().run_cell_magic('time', '', '# Count number of flops required to reach loss < 1e-2\ncount_outer = 0\nwhile True:\n    count_outer += 1\n    _ = session.run(min_op)\n    lossval = session.run(loss_fn_tensor)\n    if count_outer % 10 == 0: print(lossval)\n    if lossval < 1e-2: break\nprint(count_outer)')



total_flops = flops_adam_and_fwd * count_outer



total_flops





