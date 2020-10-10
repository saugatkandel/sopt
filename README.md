# AD-based Second Order Optimization

Tensorflow reverse-mode optimization routines that use a damped Generalized Gauss-Newton matrix. The methods included are:

1) Levenberg-Marquardt with projected gradient for convex constraints
2) Curveball
3) Nonlinear conjugate gradient (PR)
4) Backtracking and adaptive linesearch methods
5) An interface to the scipy optimizer class for gradient based or full Newton-type algorithms.


**Basics**:

We can write an optimization problem with *m* parameters and *n* data points as a composition of the "model"

<img src="https://latex.codecogs.com/svg.latex?\small&space;f:\mathbb{R}^m\rightarrow\mathbb{R}^n" title="\Large Model" />

and the "loss":

<img src="https://latex.codecogs.com/svg.latex?\small&space;L:\mathbb{R}^n\rightarrow\mathbb{R}." title="\Large Loss" />

The generalized Gauss-Newton matrix then takes the form  

<img src="https://latex.codecogs.com/svg.latex?\small&space;G=J^T_f\cdot{H}_L\cdot{J}_f" title="\Large Gauss-Newton" />

with *J* is the Jacobian for the function *f*, and *H* the hessian for *L*. 

**Notes**:

1. We can either manually supply the hessian of the loss function (if the hessian is a diagonal matrix), 
or have the optimizers calculate the hessian-vector products internally. 
By default, the hvps are calculated internally. For least-squares loss functions, 
the hessian of the loss function *L* is simply an identity matrix. 
In this case, we can simply supply the parameter `_diag_hessian_fn_ = lambda x: 1.0` to save some (miniscule)
computational effort.

2. When the input variable is not 1-D, it is difficult to keep track of the correct shapes for the various matrix-vector products. While this is certainly doable, I am avoiding this extra work for now by assuming that the input variable and predicted and measured data are all 1-D. It is simpler to just reshape the variable as necessary for any other calculation/analysis.

3. For now, the optimizers support only a single variable, not a list of variables. 
If I want to use a list of variables, I would either create separate optimizers for each and use
 an alternating minimization algorithm, or use a single giant concatenated variable that contains the desired variable.

5. The optimizers require callable functions, instead of just the tensors, 
to calculate the predicted data and the loss value. 

6. To see example usages, check the *tests* module.

**Warning: deprecated**:
1) the Autograd code.
2) the Tensorflow 1.x codes
3) the *examples* and the *benchmarks* modules.

***Todo***:
1) Consistent naming for `loss`, and `objective`. 


***References***:

1) https://j-towns.github.io/2017/06/12/A-new-trick.html ("trick" for reverse-mode jacobian-vector product calculations)
2) https://arxiv.org/abs/1805.08095 (theory for the Curveball method)
3) https://github.com/jotaf98/curveball (a mixed-mode (forward + reverse) implementation of Curveball)
4) https://arxiv.org/pdf/1201.5885.pdf (for the LM diagonal scaling)
5) Martens, J. (2016). Second-Order Optimization for Neural Networks. U. of Toronto Thesis, 179. (For preconditioning)
6) the scipy lm code (for the LM xtol and gtol conditions)
7) the manopt package (for the linesearches)
8) Numerial optimization by Nocedal and Wright. (the LM, CG, and linesearch codes all rely on this)
