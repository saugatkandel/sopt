# Second Order Optimization

Tensorflow and autograd reverse-mode optimization routines that use a damped Gauss-Newton matrix. The methods included are:

1) Levenberg-Marquardt
2) Curveball

The code here is primarily based on the references: 

1) https://j-towns.github.io/2017/06/12/A-new-trick.html ("trick" for reverse-mode jacobian-vector product calculations)
2) https://arxiv.org/abs/1805.08095 (theory for the Curveball method)
3) https://github.com/jotaf98/curveball (a mixed-mode (forward + reverse) implementation of Curveball)

Basics:

We can write an optimization problem with *m* parameters and *n* data points as a composition of the "model"

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f:\mathbb{R}^m\rightarrow\mathbb{R}^n" title="\Large Model" />

and the "loss":

<img src="https://latex.codecogs.com/svg.latex?\Large&space;L:\mathbb{R}^n\rightarrow\mathbb{R}." title="\Large Loss" />

The generalized Gauss-Newton matrix then takes the form  

<img src="https://latex.codecogs.com/svg.latex?\Large&space;G=J^T_f\cdot{H}_L\cdot{J}_f" title="\Large Gauss-Newton" />

with *J* is the Jacobian for the function *f*, and *H* the hessian for *L*. 

Todo:

The Tensorflow version of the algorithms is ahead of the Autograd versions. At some point, I need to update the Autograd 
version so that the algorithms match up.. 

Notes:

1. For least-squares loss functions, the hessian of the loss function *L* is simply an identity matrix. In this case, we do not need to calculate the hessian-vector products, which saves some computational effort. The included code assumes by default  that the loss function is a least-squared problem. This can be switched off by setting the parameter **_squared_loss_ = False** when initializing the optimizer.

2. When the input variable is not 1-D, it is difficult to keep track of the correct shapes for the various matrix-vector products. While this is certainly doable, I am avoiding this extra work for now by assuming that the input variable and predicted and measured data are all 1-D. It is simpler to just reshape the variable as necessary for any other calculation/analysis.

3. For now, the optimizers support only a single variable, not a list of variables. If I want to use a list of variables, I would either create separate optimizers for each and use an alternating minimization algorithm, or use a single giant concatenated variable that contains the desired variable.

4. For now, the tensorflow-based optimizers do not inherit from tf.train.Optimizer. I will look into this in the future.

5. The optimizers require callable functions, instead of just the tensors, to calculate the predicted data and the loss value. This is primarily to accomodate the LMA algorithm where we need to ensure that we can calculate the loss value without affecting the second order matrix-vector products.
