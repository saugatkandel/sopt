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

Note:

For least-squares loss functions, the hessian of the loss function *L* is simply an identity matrix. In this case, we do not need to calculate the hessian-vector products, which saves some computational effort. The included code assumes by default  that the loss function is a least-squared problem. This can be switched off by setting the parameter **_squared_loss_ = False** when initializing the optimizer.


