#Author - Saugat Kandel
# coding: utf-8


# This class tests whether the tensorflow and autograd versions give identical outputs for a simple least squared loss function



from autograd import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from optimizers.autograd.curveball import Curveball as Cag
from optimizers.tensorflow.curveball import Curveball as Cat



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



z_true = np.random.randn(3,100).astype('float32')

random_mat = np.random.randn(3,3)
random_symmetric_mat = random_mat + random_mat.T
evals, evecs = np.linalg.eig(random_symmetric_mat)
affine_transform = evecs

y_true = affine_transform @ z_true
y_true_flat = y_true.flatten()



def y_pred(z_flat):
    z_arr = np.reshape(z_flat, (3, -1))
    return (affine_transform @ z_arr).flatten()

def loss_fn(y):
    return 0.5 * np.sum((y - y_true_flat)**2)



z_guess = np.random.randn(300).astype('float32')



# Testing the autograd implementations



ca1 = Cag(z_guess, y_pred, loss_fn, squared_loss=True)
ca2 = Cag(z_guess, y_pred, loss_fn, squared_loss=False)



ag_losses1 = []
ag_losses2 = []
for i in range(100):
    out1 = ca1.minimize()
    lossval = loss_fn(y_pred(out1))
    out2 = ca2.minimize()
    ag_losses1.append(loss_fn(y_pred(out1)))
    ag_losses2.append(loss_fn(y_pred(out2)))



# Tensorflow test



tf.reset_default_graph()
var1 = tf.get_variable('var1', dtype=tf.float32, initializer=z_guess)
var2 = tf.get_variable('var2', dtype=tf.float32, initializer=z_guess)

tf_y_true = tf.convert_to_tensor(y_true_flat, dtype='float32', name='y_true')
tf_affine_transform = tf.convert_to_tensor(affine_transform, dtype='float32', name='affine_transform')

def tf_y_pred(z):
    return tf.reshape(tf_affine_transform @ tf.reshape(z, [3, -1]), [-1])
def tf_loss(y_pred):
    return 0.5 * tf.reduce_sum((tf_y_true - y_pred)**2)

preds1 = tf_y_pred(var1)
preds2 = tf_y_pred(var2)
loss_tensor1 = tf_loss(preds1)
loss_tensor2 = tf_loss(preds2)

ct1 = Cat(var1, preds1, loss_tensor1, name='opt1', squared_loss=False)
ct2 = Cat(var2, preds2, loss_tensor2, name='opt2', squared_loss=False)

ct1_min = ct1.minimize()
ct1_placeholder = ct1.loss_after_update_placeholder
ct1_damping_update = ct1.damping_update()

ct2_min = ct2.minimize()
ct2_placeholder = ct2.loss_after_update_placeholder
ct2_damping_update = ct2.damping_update()


session = tf.Session()
session.run(tf.global_variables_initializer())



tf_losses1 = []
tf_losses2 = []
for i in range(100):
    session.run([ct1_min, ct2_min])
    lossval1, lossval2 = session.run([loss_tensor1, loss_tensor2])
    session.run([ct1_damping_update, ct2_damping_update],
                feed_dict={ct1_placeholder: lossval1,
                           ct2_placeholder: lossval2})
    tf_losses1.append(lossval1)
    tf_losses2.append(lossval2)
    



plt.plot(ag_losses1, color='blue', ls=':', linewidth=5.0, alpha=0.8, label='ag_sq_true')
plt.plot(ag_losses2, color='green', ls='--', linewidth=5.0, alpha=0.4, label='ag_sq_false')
plt.plot(tf_losses1, color='red', ls=':', linewidth=5.0, alpha=0.8, label='tf_sq_true')
plt.plot(tf_losses2, color='orange', ls='--', linewidth=5.0, alpha=0.4, label='tf_sq_false')
plt.yscale('log')
plt.legend(loc='best')
plt.show()





