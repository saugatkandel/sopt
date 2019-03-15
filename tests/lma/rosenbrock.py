#Author - Saugat Kandel
# coding: utf-8


# This class only tests whether tensorflow and autograd both calculate the hessian-vector-products identically.



from autograd import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from optimizers.autograd.lma import LMA
#from optimizers.tensorflow.curveball import Curveball as Cat



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



def rosenbrock(x):
    x_reshaped = np.reshape(x, [2, -1])
    return np.sum(100*(x_reshaped[1] - x_reshaped[0]**2)**2 + (1 - x_reshaped[0])**2)



x_fn = lambda z: z
z_init = np.zeros(10)



# Autograd
lma_ag = LMA(z_init, x_fn, rosenbrock, squared_loss=False, cg_tol=1e-8, max_cg_iter=50, damping_update_factor=0.9)



ag_losses = []
for i in range(100):
    out = lma_ag.minimize()
    lossval = rosenbrock(x_fn(out))
    ag_losses.append(lossval)
    if i % 10 == 0: print(lossval)



out



# Tensorflow
tf.reset_default_graph()
tf_var = tf.Variable(z_init, dtype='float32')
tf_x_fn = lambda x: tf.identity(x)
tf_x_fn_tensor = tf_x_fn(tf_var)

def tf_rosenbrock(x):
    x_reshaped = tf.reshape(x, [2, -1])
    return tf.reduce_sum(100*(x_reshaped[1] - x_reshaped[0]**2)**2 + (1 - x_reshaped[0])**2)

tf_rosenbrock_tensor = tf_rosenbrock(tf_x_fn_tensor)

cball_tf = Cat(tf_var, tf_x_fn, tf_rosenbrock, name='ros', squared_loss=False)
minimizer = cball_tf.minimize()

session = tf.Session()
session.run(tf.global_variables_initializer())



tf_losses = []
for i in range(30):
    session.run(minimizer)
    lossval = session.run(tf_rosenbrock_tensor)
    #session.run(damping_update, feed_dict={placeholder:lossval})
    tf_losses.append(lossval)



# The discrepancy here is because curveball requires a matrix inversion step
# the matrix becomes singular fairly often
# I tried calculating the pseudo inverse myself, but all the approach I tried for this 
# in tensorflow game solutions less stable and accurate than the numpy counterpart.
plt.plot(ag_losses, color='blue', ls=':', linewidth=5.0, alpha=0.8, label='ag')
plt.plot(tf_losses, color='red', ls='--', linewidth=5.0, alpha=0.4, label='tf')
plt.yscale('log')
plt.legend(loc='best')
plt.show()



# Solution is all ones
session.run(tf_var)





