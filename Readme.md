[toc]
# SVM

Simple implementation of a Support Vector Machine using SMO

## 原理解释：
支持向量机(Support Vector Machines, SVM): 是一种监督学习算法。

* 支持向量(Support Vector)就是离分隔超平面最近的那些点。
* 机(Machine)就是表示一种算法，而不是表示机器。

### SVM 工作原理

1. 寻找最大分类间距
2. 转而通过拉格朗日函数求解对偶问题
3. 利用SMO算法优化求解对偶问题

#### 怎么寻找最大间隔

> 点到超平面的距离

* 分隔超平面`函数间距`:  $$y(x)=w^Tx+b$$
* 分类的结果:  $$f(x)=sign(w^Tx+b)$$  (sign表示>0为1，<0为-1，=0为0) 
* 点到超平面的`几何间距`: $$d(x)=(w^Tx+b)/||w||$$  （||w||表示w矩阵的二范数=> $$\sqrt{w^T*w}$$, 点到超平面的距离也是类似的）

> 拉格朗日乘子法

* 类别标签用-1、1，是为了后期方便 $$label*(w^Tx+b)$$ 的标识和距离计算；如果 $$label*(w^Tx+b)>0$$ 表示预测正确，否则预测错误。

* 现在目标很明确，就是要找到`w`和`b`，因此我们必须要找到最小间隔的数据点，也就是前面所说的`支持向量`。

  * 也就说，让最小的距离取最大.(最小的距离: 就是最小间隔的数据点；最大: 就是最大间距，为了找出最优超平面--最终就是支持向量)
  * 目标函数: $$arg: max_{w, b} \left( min[label*(w^Tx+b)]*\frac{1}{||w||} \right) $$
    1. 如果 $$label*(w^Tx+b)>0$$ 表示预测正确，也称`函数间隔`，$$||w||$$ 可以理解为归一化，也称`几何间隔`。
    2. 令 $$label*(w^Tx+b)>=1$$， 因为0～1之间，得到的点是存在误判的可能性，所以要保障 $$min[label*(w^Tx+b)]=1$$，才能更好降低噪音数据影响。
    3. 所以本质上是求 $$arg: max_{w, b}  \frac{1}{||w||} $$；也就说，我们约束(前提)条件是: $$label*(w^Tx+b)=1$$

* 新的目标函数求解:  $$arg: max_{w, b}  \frac{1}{||w||} $$

  * => 就是求: $$arg: min_{w, b} ||w|| $$ (求矩阵会比较麻烦，如果x只是 $$\frac{1}{2}*x^2$$ 的偏导数，那么。。同样是求最小值)
  * => 就是求: $$arg: min_{w, b} (\frac{1}{2}*||w||^2)$$ (二次函数求导，求极值，平方也方便计算)
  * 本质上就是求线性不等式的二次优化问题(求分隔超平面，等价于求解相应的凸二次规划问题)

* 通过拉格朗日乘子法，求二次优化问题

  得到公式:  $$max_{\alpha} \left( \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i, j=1}^{m} label_i \ast label_j \ast \alpha_i \ast \alpha_j \ast <x_i, x_j> \right) $$

* 约束条件:  $$a>=0$$ 并且 $$\sum_{i=1}^{m} a_i \ast label_i=0$$

> 松弛变量(slack variable)

* 我们知道几乎所有的数据都不那么干净, 通过引入松弛变量来 `允许数据点可以处于分隔面错误的一侧`。
* 约束条件:  $$C>=a>=0$$ 并且 $$\sum_{i=1}^{m} a_i \ast label_i=0$$

### SMO 高效优化算法

* SVM有很多种实现，最流行的一种实现是:  `序列最小优化(Sequential Minimal Optimization, SMO)算法`。
* 下面还会介绍一种称为 `核函数(kernel)` 的方式将SVM扩展到更多数据集上。

* SMO用途: 用于训练 SVM
* SMO目标: 求出一系列 alpha 和 b,一旦求出 alpha，就很容易计算出权重向量 w 并得到分隔超平面。
* SMO思想: 是将大优化问题分解为多个小优化问题来求解的。
* SMO原理: 每次循环选择两个 alpha 进行优化处理，一旦找出一对合适的 alpha，那么就增大一个同时减少一个。
  * 这里指的合适必须要符合一定的条件
    1. 这两个 alpha 必须要在间隔边界之外
    2. 这两个 alpha 还没有进行过区间化处理或者不在边界上。
  * 之所以要同时改变2个 alpha；原因是我们有一个约束条件:  $$\sum_{i=1}^{m} a_i \ast label_i=0$$；如果只是修改一个 alpha，很可能导致约束条件失效。

> SMO 伪代码大致如下: 

```
创建一个 alpha 向量并将其初始化为0向量
当迭代次数小于最大迭代次数时(外循环)
    对数据集中的每个数据向量(内循环): 
        如果该数据向量可以被优化
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
```

## 开发环境：Dev envs & dependencies
* Window10
* Python==3.8
* cuda==9.2
* Pytorch==1.5.1
* numpy == 1.18.5 

## 代码文档：Documentation for codes

+ #### Code Files

  + iris-virginica.txt

    数据文件

  + SVM.py 

    支持向量机SMO优化算法模型文件

  + Train.py

    数据处理，模型训练文件

  +  Readme.md

    说明文件

+ #### How to run

  ```
  python3 Train.py
  ```

  默认参数：

  max_iter=10000：默认最大迭代次数

  kernel_type='linear'：默认使用线性核函数

  C=1.0：默认正则化参数

  epsilon=0.001：默认收敛参数

+ #### 数据处理

  将数据处理成(data, label)的格式，并且进行shuffle打乱顺序

  取出75%的数据（112条）作为训练集，25%（38条）作为测试集

  数据格式如下

  ```
  ([4.8,3.4,1.9,0.2], 1)
  ([6.3,3.4,5.6,2.4],-1)
  ```

  

+ #### 决策函数

  $f(x) = sign(W\cdot X^T + b)$ 

+ #### 核函数

  $K( x, x′) =<φ( x) ⋅φ( x′) >$

+ #### Results

  ![image-20210322182027753](C:\Users\Jackson\AppData\Roaming\Typora\typora-user-images\image-20210322182027753.png)

