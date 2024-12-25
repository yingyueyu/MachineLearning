# 将svm原问题转换成对偶问题

## 原问题转换成普适定义

原问题:

$$
\min f(W) = \min \frac{1}{2}\|W\|^2 + C \overset{N}{\underset{i=1}\sum}\xi_i
$$

限制条件:

1. $Y_i[W^T\phi(X_i) + b] \geq 1 - \xi_i$
2. $\xi_i \geq 0$

注意: 原问题一定是凸函数，一定有唯一最优解

**为了满足原函数的普适定义**，我们需要满足:

3. $g_i(w) \leq 0$
4. $h_i(w) = 0$

我们将条件 1 转换成条件 3

$$
Y_i[W^T\phi(X_i) + b] \geq 1 - \xi_i \\
Y_i[W^T\phi(X_i) + b] + \xi_i \geq 1 \\
Y_i[W^T\phi(X_i) + b] + \xi_i - 1 \geq 0 \\
-(Y_i[W^T\phi(X_i) + b] + \xi_i - 1) \leq 0 \\
-Y_i[W^T\phi(X_i) + b] - \xi_i + 1 \leq 0 \\
-Y_iW^T\phi(X_i) - Y_ib - \xi_i + 1 \leq 0 \\
1 - \xi_i -Y_iW^T\phi(X_i) - Y_ib \leq 0
$$

因为原限制条件 2 的存在， 我们将符号反过来 $\xi_i \leq 0$，则

$$
1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib \leq 0
$$

又因为 $\xi_i \leq 0$ 被改成了小于 0，所以原问题 $f(W) = \frac{1}{2}\|W\|^2 - C \overset{N}{\underset{i=1}\sum}\xi_i$

==**所以整体来讲原问题被转换为:**==

- 最小化: $\frac{1}{2}\|W\|^2 - C \overset{N}{\underset{i=1}\sum}\xi_i$
- 限制条件:
  - $1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib \leq 0$
  - $\xi_i \leq 0$

==**注意点有两个:**==
1. 上面的普适定义中没有 $h(w)$
2. 上面的两个限制条件，对应的是普适定义中的一个 $g(w)$

这两点对后面带入对偶问题很有用

## 对偶问题

定义 L 函数

$$
L(w, \alpha, \beta) = f(w) + \overset{K}{\underset{i=1}\sum} \alpha_i g_i(w) + \overset{M}{\underset{i=1}\sum} \beta_i h_i(w)
$$

定义对偶问题

- 最大化: $\theta(\alpha, \beta)=\underset{所有w}\inf\{L(w, \alpha, \beta)\}$
- 限制条件: $\alpha_i \geq 0$ $(i=1 \sim K)$

带入之前推导出来的原问题到 $\theta$ 中:

最大化:

$$
\theta(\alpha, \beta) = \underset{所有w,\xi, b}{inf}\{\frac{1}{2}\|W\|^2 - C \overset{N}{\underset{i=1}\sum}\xi_i + \overset{N}{\underset{i=1}\sum} \alpha_i [1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib] + \overset{N}{\underset{i=1}\sum}\beta_i \xi_i\}
$$

限制条件:

- $\alpha_i \geq 0$
- $\beta_i \geq 0$

==解释==:

1. 因为原问题的普适定义中的 $g(w)$ 对应两个限制条件，所以有上文的
  $$
  \overset{N}{\underset{i=1}\sum} \alpha_i [1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib] + \overset{N}{\underset{i=1}\sum}\beta_i \xi_i
  $$
2. 因为解释 1 中将 $g(w)$ 拆成了两个式子带入，所以原来的 $\overset{K}{\underset{i=1}\sum} \alpha_i g_i(w)$ 的系数 $\alpha_i$，对应解释 1 中的 $\alpha_i$ $\beta_i$

## 求对偶问题的最优解

为了让上述的 $\theta(\alpha, \beta)$ 最大，则需要让 $所有w,\xi, b$ 最小

因为 $\theta$ 是凸函数，这里可以通过求导的方式求得 $所有w,\xi, b$ 的最小值，则有以下结论

- $\frac{\delta L}{\delta w} = 0 \Rightarrow W - \overset{N}{\underset{i=1}\sum}\alpha_i Y_i \phi(X_i) = 0 \Rightarrow W = \overset{N}{\underset{i=1}\sum}\alpha_i Y_i \phi(X_i)$
- $\frac{\delta L}{\delta \xi_i} = 0 \Rightarrow -C + \alpha_i + \beta_i = 0 \Rightarrow C = \alpha_i + \beta_i$
- $\frac{\delta L}{\delta b} = 0 \Rightarrow \overset{N}{\underset{i=1}\sum}\alpha_i Y_i = 0$

带入上述条件，化简以下函数

$$
\theta(\alpha, \beta) = \underset{所有w,\xi, b}{inf}\{\frac{1}{2}\|W\|^2 - C \overset{N}{\underset{i=1}\sum}\xi_i + \overset{N}{\underset{i=1}\sum} \alpha_i [1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib] + \overset{N}{\underset{i=1}\sum}\beta_i \xi_i\}
$$

结果只剩下三项:

1. $\overset{N}{\underset{i=1}\sum}\alpha_i$
2. $\frac{1}{2}\|W\|^2$
   $$
   \frac{1}{2}\|W\|^2 = \frac{1}{2} W^TW \\
   = \frac{1}{2} (\overset{N}{\underset{i=1}\sum}\alpha_i Y_i \phi(X_i))^T (\overset{N}{\underset{j=1}\sum}\alpha_j Y_j \phi(X_j)) \\
   = \frac{1}{2}\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_j\phi(X_i)^T\phi(X_j) \\
   = \frac{1}{2}\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_jK(X_i,X_j)
   $$
3. $-\overset{N}{\underset{i=1}\sum}\alpha_iY_iW^T\phi(X_i)$
   $$
   -\overset{N}{\underset{i=1}\sum}\alpha_iY_iW^T\phi(X_i) = -\overset{N}{\underset{i=1}\sum}\alpha_iY_i(\overset{N}{\underset{j=1}\sum}\alpha_j Y_j \phi(X_j))^T\phi(X_i) \\
   = -\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_j\phi(X_ j)^T\phi(X_i) \\
   = -\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_jK(X_j,X_i) \\
   = -\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_jK(X_i,X_j)
   $$

将上述3项加起来，则:

$$
\theta(\alpha, \beta) = \overset{N}{\underset{i=1}\sum}\alpha_i - \frac{1}{2}\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_jK(X_i,X_j)
$$

再看限制条件，因为

- $C = \alpha_i + \beta_i$
- $\alpha_i \geq 0$
- $\beta_i \geq 0$

所以合并这三个条件则：

$$
0 \leq \alpha_i \leq C
$$

## 对偶问题化简结果

最大化:

$$
\theta(\alpha) = \overset{N}{\underset{i=1}\sum}\alpha_i - \frac{1}{2}\overset{N}{\underset{i=1}\sum}\overset{N}{\underset{j=1}\sum}\alpha_i\alpha_jY_iY_jK(X_i,X_j)
$$

限制条件

- $0 \leq \alpha_i \leq C$
- $\overset{N}{\underset{i=1}\sum}\alpha_i Y_i = 0$

## 对原始分类问题进行转换

通过上述对偶问题，我们可以优化出 $\alpha$，但是原问题中，我们需要求 $W \xi b$，所以我们还要进行转换

此处SVM发明者告诉我们，并不一定要知道 $W \xi$ 我们就能实现分类任务

### 梳理测试流程

这里我们先不考虑优化问题，我们已检测样本的测试流程来看以下内容，假设有样本 $X$ 判断其属于哪一类，则有:

$$
若 \ W^T\phi(X) + b \geq 0 \ 则 \ y = +1 \\
若 \ W^T\phi(X) + b \lt 0 \ 则 \ y = -1
$$

### 求 $W^T\phi(X)$

根据之前的结论 $\frac{\delta L}{\delta w} = 0 \Rightarrow W = \overset{N}{\underset{i=1}\sum}\alpha_i Y_i \phi(X_i)$，带入测试函数中，有:

$$
W^T\phi(X) = (\overset{N}{\underset{i=1}\sum}\alpha_i Y_i \phi(X_i))^T\phi(X) \\
= \overset{N}{\underset{i=1}\sum}\alpha_iY_i\phi(X_i)^T\phi(X) \\
= \overset{N}{\underset{i=1}\sum}\alpha_iY_iK(X_i, X)
$$

### 求 $b$

接下来还需要推导 $b$，这里利用之前的 KKT 条件

- $\forall{i} = 1 \sim K$
- 或者 $\alpha_i^* = 0$
- 或者 $g_i^*(w^*) = 0$

我们需要理解以下两个对应关系

- 在[对偶问题](#对偶问题)的==解释==中，我们知道普适定义中的 $\alpha$ 其实应该对应我们的 $\alpha_i$ 和 $\beta_i$
- 在[原问题转换成普适定义](#原问题转换成普适定义)的==注意点有两个==中，我们知道普适定义中的 $g(w)$ 对应
  - $1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib \leq 0$
  - $\xi_i \leq 0$

==**再结合 KKT 条件，我们有以下结论**==

1. 要么 $\beta_i = 0$；要么 $\xi_i = 0$
2. 要么 $\alpha_i = 0$；要么 $1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib = 0$

此时我们任取一个 $\alpha_i$ 就能算出一个 $b$，过程如下

任取一个 $0 \lt \alpha_i \lt C \Rightarrow \beta_i = C - \alpha_i \gt 0$

由KKT条件1推出，此时 $\beta \neq 0 \Rightarrow \xi_i = 0$

由KKT条件2推出，此时 $\alpha_i \neq 0 \Rightarrow 1 + \xi_i -Y_iW^T\phi(X_i) - Y_ib = 0$

因为 $\xi_i = 0$，所以

$$
1 -Y_iW^T\phi(X_i) - Y_ib = 0
$$

$$
b = \frac{1-Y_iW^T\phi(X_i)}{Y_i} \\
= \frac{1-Y_i(\overset{N}{\underset{j=1}\sum}\alpha_j Y_j \phi(X_j))^T\phi(X_i)}{Y_i} \\
= \frac{1-\overset{N}{\underset{j=1}\sum}\alpha_jY_iY_j\phi(X_j)^T\phi(X_i)}{Y_i} \\
= \frac{1-\overset{N}{\underset{j=1}\sum}\alpha_jY_iY_jK(X_i,X_j)}{Y_i}
$$

至此为止，我们由对偶问题来求最优解的 $\alpha$，再利用 KKT 条件，解 $b$ 则能完整表示推理函数 $W^T\phi(X) + b$

接下来总结一下整个 SVM 的训练和测试过程