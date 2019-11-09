# Optimization

[TOC]

## Gradient Checking

做微小扰动，根据导数定义直接求
$$
\frac{\partial}{\partial\theta_j}J(\Theta)=\frac{J(\Theta^{(j+)})-J(\Theta^{(j-)})}{2\epsilon}
$$

## Gradient Descent

```python
# Vanilla Gradient Descent
while True:
    weights_grad = evaluate_gradient(loss_func, data, weights)
    weights += -step_size * weights_grad # perform parameter update
```

## Stochastic Gradient Descent

- $x_{t+1}=x_t-\alpha\triangledown f(x_t)$

Approximate sum using a minibatch of examples, 32/64/128 common.

```python
# Vanilla Minibatch Gradient Descent
while True:
    data_batch = sample_training_data(data,256) # sample 256 examples
    weights_grad = evaluate_gradient(loss_func, data_batch, weights)
    weights += -step_size * weights_grad # perform parameter update
```

- [ ] Very slow progress along shallow dimension, jitter along steep direction
- [ ] Zero gradient, gradient descent gets stuck (Saddle points much more common in high dimension)
- [ ] Our gradients come from minibatches so they can be noisy!

## SGD + Momentum

- $v_{t+1}=\rho v_t+\triangledown f(x_t)$
- $x_{t+1}=x_t-\alpha v_{t+1}$

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x += learning_rate * vx
```

Typically $\rho$=0.9 or 0.99

## Nesterov Momentum

计算将要到达的点的梯度，而不是当前点

Change of variables $\tilde{x_t}=x_t+\rho v_t$

- $v_{t+1}=\rho v_t-\alpha\triangledown f(\tilde{x_t})$
- $\tilde{x_{t+1}}=\tilde{x_t}+v_{t+1}+\rho(v_{t+1}-v_t)​$

```python
dx = compute_gradient(x)
old_v = v
v = rho * v - learmning_rate * dx
x += -rho * old_v + (1 + rho) * v
```

## Adagrad

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

## RMSProp

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

## Adam

综合了Momentum和Adagrad/RMSProp

```python
first_moment = 0
second_moment = 0
for t in range(num_iterations):
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx # Momentum
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx # AdaGrad / RMSProp
    first_unbias = first_moment / (1 - beta1 ** t) # 
    second_unbias = second_moment / (1 - beta2 ** t) # Bias correction
    x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```

Bias correction for the fact that first and second moment estimates start at zero

- Adam with beta1 = 0.9, beta2 = 0.999, and learning_rate = 1e-3 or 5e-4 is a great starting point for many models!