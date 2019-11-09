# Anomaly Detection

## Algorithm

假设所有$n$个变量互相独立且服从高斯分布，则

1. 对于每个单一变量：
   $$
   p(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
   $$

2. 对于整体而言：
   $$
   p(x)=\prod_{j=1}^np(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n\frac{1}{\sqrt{2\pi\sigma_j^2}}\exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})
   $$
   Anomaly if $p(x)<\epsilon$

特性：高斯分布均沿着坐标轴，无法处理某些变量相关，其中一个异常的情况，需要手动添加feature

## Multivariate Gaussian distribution

1. Fit model $p(x)$ by setting
   $$
   \mu=\frac{1}{m}\sum_{i=1}^m x^{(i)}
   $$

   $$
   \Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu)(x^{(i)}-\mu)^T
   $$

2. Given a new example $x$, compute
   $$
   p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
   $$
   Flag a anomaly if $p(x)<\epsilon$

一元高斯分布是多元的特例。仅当$m\gg n$的时候使用（计算$\Sigma^{-1}$需要$O(n^3)$）

