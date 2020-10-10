# The problem is that a lot of these higher order derivatives don't work properly with tf.function.
# I often run into unexpected errors.
# It is probably worth trying out pytorch and jax for this.
import tensorflow as tf


# @tf.function
def jvp_forward(predict_fn, var, vector):
    with tf.autodiff.ForwardAccumulator(primals=var, tangents=vector) as acc:
        prediction = predict_fn(var)
    jvp = acc.jvp(prediction)
    return prediction, jvp


# @tf.function
def jvp_double_backward(predict_fn, var, vectors):
    "Attempting using persistent storage to efficiently calculate jvp for multiple vectors"
    with tf.GradientTape(persistent=True) as gt:
        with tf.GradientTape() as gt2:
            prediction = predict_fn(var)
        dummy = tf.ones_like(prediction)
        gt.watch(dummy)
        inner_vjp = gt2.gradient(prediction, var, output_gradients=dummy)
    jvps = [gt.gradient(inner_vjp, dummy, output_gradients=vector)
            for vector in vectors]
    return prediction, jvps


# @tf.function
def hvp_forward_backward(objective_fn, objective_input, vector):
    "I think this should be the preferred version"
    with tf.autodiff.ForwardAccumulator(primals=objective_input, tangents=vector) as acc:
        with tf.GradientTape() as gt:
            gt.watch(objective_input)
            loss = objective_fn(objective_input)
        backward = gt.gradient(loss, objective_input)
    hvp = acc.jvp(backward)
    return loss, backward, hvp


# @tf.function
def hvp_direct(objective_fn, objective_input, vectors):
    "Attempting using persistent storage to efficient calculate hvp for multiple vectors"
    with tf.GradientTape(persistent=True) as gt:
        gt.watch(objective_input)
        with tf.GradientTape() as gt2:
            gt2.watch(objective_input)
            loss = objective_fn(objective_input)
        grad = gt.gradient(loss, objective_input)
        dot_prods = [tf.reduce_sum(grad * v) for v in vectors]
    hvps = [gt.gradient(dp, objective_input) for dp in dot_prods]
    return loss, grad, hvps


# @tf.function
def hvp_backward_forward(objective_fn, objective_input, vectors):
    "Attempting using persistent storage to efficient calculate hvp for multiple vectors"
    with tf.GradientTape(persistent=True) as gt:
        gt.watch(objective_input)
        jvps = []
        for i, vector in enumerate(vectors):
            with tf.autodiff.ForwardAccumulator(primals=objective_input, tangents=vector) as acc:
                loss = objective_fn(objective_input)
            jvp = acc.jvp(loss)
            jvps.append(jvp)
    hvps = [gt.gradient(jvp, objective_input) for jvp in jvps]
    return loss, hvps

# @tf.function
# def _gvp_double_backward(var, vectors, diag_hessian=None):
#    with tf.GradientTape() as gt:
#        predicted, jvps = _jvp_double_backward(var, vectors)
#        if diag_hessian is not None:
#            hjvps = [diag_hessian * jvp for jvp in jvps]
#        else:
#            hjvps = _hvp_backward_forward(loss_fn, predicted, jvps)
#    return gt.gradient(predicted, var, output_gradients=hjvp)

# @tf.function
# def _gvp_forward(var, vector):
#    with tf.GradientTape() as gt:
#        predicted, jvp = _jvp_forward(var, vector)
#        hjvp = _hvp(loss_fn, predicted, jvp)
#    return gt.gradient(predict, var, output_gradients=hjvp)