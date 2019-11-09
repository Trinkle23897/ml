# Recommender System

[TOC]

## Problem Formulation

- $r(i,j)=1$	if user $j$ has rated movie $i$, 0 otherwise
- $y^{(i,j)}=$ rating by user $j$ on movie $i$ if defined
- $\theta^{(j)}=$ parameter vector for user $j$  
- $x^{(i)}=$ feature vector for movie $i$
- For user $j$, movie $i$, predict rating: $(\theta^{(j)})^T(x^{(i)})$ 要尽可能接近原来的$y^{(i,j)}$——回归问题

## Collaborative Filtering

1. Initialize $x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)}$ to small random values.	

2. 将$y$去均值，因为对于没有任何评分的用户而言，系统给出的得分始终是0，去均值之后给出的0等价于原始平均分

3. Given $x^{(1)},\cdots,x^{(n_m)}$, estimate $\theta^{(1)},\cdots,\theta^{(n_u)}$: 
   $$
   \min_{\theta^{(1)},\cdots,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta_k^{(j)})^2
   $$
   给电影参数求用户偏好参数

4. Given $\theta^{(1)},\cdots,\theta^{(n_u)}$, estimate $x^{(1)},\cdots,x^{(n_m)}$: 
   $$
   \min_{x^{(1)},\cdots,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2
   $$
   给用户偏好参数求电影参数

5. 综合1、2，可以同时更新$x$和$\theta$
   $$
   J(x,\theta)=\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta_k^{(j)})^2
   $$


$$
\frac{\partial J}{\partial x_k^{(i)}}=\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})\theta_k^{(j)}+\lambda x_k^{(i)}
$$

$$
\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)}+\lambda \theta_k^{(j)}
$$

- Vectorization

```matlab
% machine-learning-ex8/ex8/cofiCostFunc.m
J=sum(sum((X*Theta'-Y).^2.*R))/2+lambda/2*(sum(sum(Theta.^2))+sum(sum(X.^2)));
X_grad=(X*Theta'-Y).*R*Theta+lambda*X;
Theta_grad=((X*Theta'-Y).*R)'*X+lambda*Theta;
```

