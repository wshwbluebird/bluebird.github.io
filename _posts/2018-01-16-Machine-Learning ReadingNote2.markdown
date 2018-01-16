---
layout:     post
title:      "读书笔记 Machine Learning #2"
subtitle:   "线性模型_1"
date:       2018-01-16
author:     "bluebird"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Machine Learning
    - Reading Note
---

> “新手上路. ”

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

##1. 基本形式

对于由d个属性描述的示例，$$x = (x_1;x_2;…;x_d)$$，其中$$x_i$$是x在第i个属性上的取值，线性模型(linear model)通过学得一个属性的线性组合来进行预测的函数，即$$f(x)=w_1x_1+w_2x_2+...+w_dx_d+b$$一般用向量的形式表示为$$f(x)=w^Tx+b $$

## 2. 线性回归

给定数据集$$D = \{(x_1,y_1),(x_2,y_2),…,(x_m,y_m)\}$$， 其中$$x_i = (x_{i1},x_{i2},…,x_{id}), y_i\in R.$$ 线性回归视图学得一个线性模型尽可能准确的预测实值输出的标记

####属性数量为1的情况

模型试图学得$$f(x_i) = wx_i+b, 使得 f(x_i)\approx y_i$$。确定w和b的过程就是学习器学习的过程，主要依靠的是衡量f(x)和y之间的差别，其中均方误差是一个常用的度量，我们可以让模型的均方误差最小化，即

$$(w^* ,b^*) = \arg\min_{(w,b)} \sum_{i=1}^m(y_i-wx_i-b)^2$$

也就是使用最小二乘法进行回归求解

求解w和b使$$E_{(w,b)} = \sum_{i=1}^m(y_i-wx_i-b)^2$$的过程称为线性回归的最小二乘**参数估计**(parameter estimation),我们可以通过对w和b求偏导的方式确定目标函数的最值以及w和b的取值。首先求导得到

$$\frac{\partial E_{(w,b)}}{\partial w} = 2(w\sum_{i=1}^mx_i^2 - \sum_{i=1}^m(y_i-b)x_i)$$

$$\frac{\partial E_{(w,b)}}{\partial b} = 2(mb - \sum_{i=1}^m(y_i-wx_i))$$

令上面两个式子等于0，可以得到w和b的最优解

$$w = \frac{\sum_{i=1}^my_i(x_i-\overline x)}{\sum_{i=1}^mx_i^2-\frac{1}{m}(\sum_{i=1}^mx_i)^2}$$

$$b = \frac{1}{m}\sum_{i=1}^m(y_i-wx_i)$$

其中$$\overline x=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i)$$



#### 属性数量不为1的情况

更一般的情况是数据集D，样本有d个属性描述，此时我们试图学得$$f(x_i) = w^Tx_i+b, 使得 f(x_i)\approx y_i$$，这称为**多元线性回归**(multivariate linear regression)

针对这种情况，通常考虑把w和b吸入向量并把数据集化成$$m\times (d+1)$$大小的矩阵的形式进行最小二乘估计，具体包括

1. $$\hat w = (w;b)$$
2. $$X= \left [\begin{matrix} x_{11}&x_{12}&…&x_{1d}&1\\x_{21}&x_{22}&…&x_{2d}&1\\…&…&…&…&…\\x_{m1}&x_{m2}&…&x_{md}&1\end{matrix}\right ]$$
3. $$y=(y_1;y_2;…;y_m)$$

我们可以到到优化模型为

$$\hat w^* = \arg\min_{\hat w}(y-X\hat w)^T(y-X\hat w)$$

令$$E_{\hat w} = (y-X\hat w)^T(y-X\hat w)$$，对$$\hat w$$进行求导得到

$$\frac{\partial E_{\hat w}}{\partial \hat w}=2X^T(X\hat w-y)$$ 令此式子得0即可求得$$\hat w$$最优解（但是由于矩阵，情况会变得复杂，我们此时假设$$X^TX$$为满秩矩阵或正定矩阵），则我们可以求解出

$$\hat w^* = (X^TX)^{-1}X^Ty$$其中$$ (X^TX)^{-1}$$是$$ (X^TX)$$的逆矩阵

当$$X^TX$$不是满秩矩阵的时候，这意味着会求出多个$$\hat w^*$$都能使均方误差最小化，这时候将又学习算法的归纳偏好决定（后面章节）

#### 扩展

|  情况  |              方程              |           描述           |
| :--: | :--------------------------: | :--------------------: |
|  线性  |        $$y = w^Tx+b$$        |     让$$w^Tx+b$$逼近y     |
|  对数  |      $$\ln y = w^Tx+b$$      |   让$$e^{w^Tx+b}$$逼近y   |
|  泛化  | $$ g(y) = w^Tx+b$$，g()是单调可微的 | 让$$g^{-1}(w^Tx+b)$$逼近y |



## 3. 对数几率回归

对数几率回归用来解决二分类问题，在线性回归模型中，预测值$$z=w^Tx+b$$ 是实值，二在分类问题中需要将其转换成0/1值，最理想的函数是**单位阶跃函数**(unit-step function)

$$\begin{equation} f(x)= \begin{cases} 0& \text{z<0;} \\0.5& \text{z=0;}\\1& \text{z>0;}\end{cases}\end{equation}$$

但是该函数不是连续函数，不能使用上面说的$$ g(y) = w^Tx+b$$，使其转化为泛化的线性回归问题，所以使用**对数几率函数**(logistic function该术语来自周志华老师的书)

$$y=\frac{1}{1+e^{-z}}$$

对数几率函数是一个sigmoid函数，可以将z值转化为一个接近0或1的值

![ROC](https://github.com/wshwbluebird/wshwbluebird.github.io/raw/master/ML_img/sigmoid.jpeg)

使用对数几率函数可以替代$$ g(y) = w^Tx+b$$中的g，将问题转化为线性回归问题进行求解，可以得到

$$\ln \frac{y}{1-y} = w^Tx+b$$

书中对这个概念进行了进一步的解释：y是x作为正例的可能性，1-y是x作为反例的可能性，两者的比值成为**几率**(odds)反应了x作为正例的相对可能性，取对数之后可到到**对数几率**(logit)，$$\ln \frac{y}{1-y}$$ 该函数是任意阶可导的凸函数有很好的数学性质，有很多优化算法都可以进行求解。

#####求解

将y视为类的后验概率估计$$p(y=1|x)$$，则可以得到$$\ln \frac{p(y=1|x)}{p(y=0|x)} = w^Tx+b$$

根据之前定义和概率运算可以得到

$$p(y=1|x) = \frac{e^{w^T+b}}{1+e^{w^T+b}}$$

$$p(y=0|x) = \frac{1}{1+e^{w^T+b}}$$

用极大似然法构造对数似然方程

$$\mathfrak l(w,b) = \sum_{i=1}^m\ln p(y_i|x_i;w,b)$$ 

每个样本属于其真实标记的概率越大越好，令 $$\beta = (w;b)$$ ， $$\hat x = (x;1)$$  将$$w^Tx+b$$简写成$$\beta^T\hat x$$，将简化后的式子带入消元，得到新的似然方曾为

$$\mathfrak l(w,b) = \sum_{i=1}^m(-y_i\beta^T\hat x_i+\ln (1+e^{\beta^T\hat x_i}))$$

此式为关于$\beta$的高阶可导连续函数，根据凸优化理论可使用梯度下降法或牛顿法来求解，即

$$\beta^*=\arg\min_\beta \mathfrak l(\beta)$$

由于此方程存在二阶导数，故可以使用牛顿法进行迭代求解，可得更新公式

$$\beta^{t+1} = \beta^t-(H)^{-1}J$$

其中

$$H = \frac{\partial^2\mathfrak l(\beta)}{\partial\beta\partial\beta} =\sum_{i=1}^m\hat x_i\hat x_i^Tp1(\hat x_i;\beta)(1-p_1(\hat x_i;\beta))$$

$$J = \frac{\partial\mathfrak l(\beta)}{\partial\beta} =  -\sum_{i=1}^m\hat x_i(y_i-p_1(\hat x_i;\beta))$$



####实现代码

编程实现对率回归，并给出西瓜数据集3.0a上的结果

在这里我使用python进行编程，代码如下

```python
# encoding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt


def read(filename):
    """
    读取文件
    :param filename: 文件名
    :return:   文件中的数据和标签
    """
    x = []
    y = []
    r = []
    with open(filename, 'r') as f:
        p = f.readlines()
        for line in p:
            mp = line.split(' ')
            x.append(mp[1])
            y.append(mp[2])
            r.append(mp[3])

    return [x, y, r]


def p1(x, beta):
    """
    获得当前参数为正例的概率
    :param x:       样例x
    :param beta:    当前参数
    :return:        概率
    """
    up = 0.0
    for i in range(len(x)):
        up = up + x[i] * beta[i]
    up = math.exp(up)
    return up / (1.0 + up)


def getJ(xarray, beta, y):
    """
    获得似然函数的雅克比矩阵
    :param xarray:  输入进来的序列
    :param beta:    当前的参数
    :param y:       标签
    :return:        雅克比矩阵
    """
    ans = np.array([0.0, 0.0, 0.0]).astype('float64')
    for i in range(0, len(xarray[0])):
        x = xarray[:, i]
        ans += x * (y[i] - p1(x, beta))
    return -ans


def getH(xarray, beta, y):
    """
    获得似然函数的海森矩阵
    :param xarray: 输入进来的序列
    :param beta:   当前的参数
    :param y:      标签
    :return:       海森矩阵
    """
    zero = [0, 0, 0]
    ans = np.array([zero, zero, zero]).astype('float32')
    for i in range(0, len(xarray[0])):
        x = xarray[:, i]
        xx = np.dot(np.array([x]).T, np.array([x]))
        ans += xx * p1(x, beta) * (1 - p1(x, beta))

    return ans


def main():
    """
    主函数
    :return:
    """

    # 读取数据并整合为np array 格式(迭代用)
    [x1, x2, r] = read("melon3_0a.txt")
    x = np.array([np.array(x1).astype('float32'), np.array(x2).astype('float32'), np.ones(len(x1)).astype('float32')]);
    beta = np.array([0.0, 0.0, 0.0]).astype('float32')
    y = np.array(r).astype('float32')

    # 真核数据将数据整合为list格式 (作图用)
    x1 = x[0, :]
    x1 = x1.tolist()
    x2 = x[1, :]
    x2 = x2.tolist()
    r = y.tolist()

    # 画出图像标签
    f1 = plt.figure(1)
    st = f1.add_subplot(111)
    st.set_title('LR')
    plt.xlabel('density')
    plt.ylabel('sugar content')

    # 画出散点图
    for i in range(len(x1)):
        if r[i] == 1:
            st.scatter(x1[i], x2[i], c='r', marker='o')
        else:
            st.scatter(x1[i], x2[i], c='b', marker='x')

    # 学习参数 迭代次数为50
    for i in range(1, 50):
        de = np.array(np.dot(np.mat(getH(x, beta, y)).I, getJ(x, beta, y).T).tolist()[0])
        beta = beta - de

    print(beta)
    # 根据学习出来的参数 画出边界曲线
    beta = beta.tolist()
    ply = -(0.1 * beta[0] + beta[2]) / beta[1]
    pry = -(0.9 * beta[0] + beta[2]) / beta[1]
    plt.plot([0.1, 0.9], [ply, pry])
    plt.show()

if __name__ == "__main__":
    main()

```



西瓜数据集3.0a的内容如下

```
1 0.697 0.460 1
2 0.774 0.376 1
3 0.634 0.264 1
4 0.608 0.318 1
5 0.556 0.215 1
6 0.403 0.237 1
7 0.481 0.149 1
8 0.437 0.211 1
9 0.666 0.091 0
10 0.243 0.267 0
11 0.245 0.057 0
12 0.343 0.099 0
13 0.639 0.161 0
14 0.657 0.198 0
15 0.360 0.370 0
16 0.593 0.042 0
17 0.719 0.103 0

```

西瓜数据集放在和该代码相同的路径下即可运行，参照网上另外一份基于matlab 的实现，我也参照此方法进行画图，即西瓜类别的分界线，如下

![ROC](https://github.com/wshwbluebird/wshwbluebird.github.io/raw/master/ML_img/LR.png)



## 参考资料

blog.csdn.net/icefire_tyh/article/details/52068844

周志华 《机器学习》 清华大学出版社