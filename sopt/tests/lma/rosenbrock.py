#Author - Saugat Kandel
# coding: utf-8


# This class tests whether tensorflow and autograd both calculate the hessian-vector-products identically.

# The rosenbrock function is not a least squares optimization problem. 
# Additionally, I have formulated the loss function in such a way that to find a minimum, we need to 
# calculate the hessian-vector product.



from autograd import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sopt.optimizers.autograd.lma import LMA as LMAag
from sopt.optimizers.tensorflow.lma import LMA as LMAtf



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



def rosenbrock(x):
    x_reshaped = np.reshape(x, [2, -1])
    return np.sum(100*(x_reshaped[1] - x_reshaped[0]**2)**2 + (1 - x_reshaped[0])**2)



x_fn = lambda z: z
z_init = np.zeros(10)



# Autograd
lma_ag = LMAag(z_init, x_fn, rosenbrock, squared_loss=False, cg_tol=1e-5, max_cg_iter=20)



ag_losses = []
for i in range(10):
    out = lma_ag.minimize()
    lossval = rosenbrock(x_fn(out))
    ag_losses.append(lossval)



# Tensorflow
tf.reset_default_graph()
tf_var = tf.Variable(z_init, dtype='float32')
tf_x_fn = lambda x: tf.identity(x)
tf_x_fn_tensor = tf_x_fn(tf_var)

def tf_rosenbrock(x):
    x_reshaped = tf.reshape(x, [2, -1])
    return tf.reduce_sum(100*(x_reshaped[1] - x_reshaped[0]**2)**2 + (1 - x_reshaped[0])**2)

tf_rosenbrock_tensor = tf_rosenbrock(tf_x_fn_tensor)

lma_tf = LMAtf(tf_var, tf_x_fn, tf_rosenbrock, name='ros', squared_loss=False, cg_tol=1e-5, max_cg_iter=200)
minimizer = lma_tf.minimize()

session = tf.Session()
session.run(tf.global_variables_initializer())



tf_losses = []
for i in range(10):
    session.run(minimizer)
    lossval = session.run(tf_rosenbrock_tensor)
    tf_losses.append(lossval)



plt.plot(ag_losses, color='blue', ls=':', linewidth=5.0, alpha=0.8, label='ag')
plt.plot(tf_losses, color='red', ls='--', linewidth=5.0, alpha=0.4, label='tf')
plt.yscale('log')
plt.legend(loc='best')
plt.show()



# Solution is all ones
session.run(lma_tf._input_var)





