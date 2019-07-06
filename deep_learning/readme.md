## 一些深度学习的基础知识

1. #### [MLP](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)(Multilayer Perceptrons:多层感知机)

   ##### Multi-Layer Neural Network

* 单个神经元的处理过程：每个输入乘上各自的weight,再加上bias，再带入激活函数中

<img src="../assets/single_neuron.png" width="500">

​		公式如下( $f$ 为激活函数)：
$$
h_{W,b}(x)=f(W^Tx)=f(\sum^3_{i=1}W_ix_i+b)
$$

* 常见激活函数

  sigmoid function(S 型函数)：

  范围[0,1]
  $$
  f(z)=\frac{1}{1+exp(-z)}\\
  f'(x)=f(z)(1-f(z))
  $$
  tanh function：

  范围[-1,1]
  $$
  \qquad\qquad f(z)=tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}\\
  f'(z)=1-(f(z))^2
  $$
  rectified linear function(修正线性函数)：

  范围[0,$+\infty$]
  $$
  f(z)=max(0,x)
  $$

<img src="../assets/activation_functions.png" width=500>

##### Neural Network model

   * 一些基本概念

     蓝色的为输入层，中间黄色的为隐藏层，“+1”为bias units，该神经网络为有3个输入神经元，3个隐藏神经元，一个输出神经元

     使用$L_l$来表示在第几层。其中使用$W^{(l)}_{ij}$ 来表示在第$l$ 层的第j个神经元和在$l+1$层的第i个神经元之间的连接

     $b^{(l)}_i$ 为第$l+1$ 层 i 神经元的的bias

     <img src="../assets/Network331.png" width=500>

* 具体计算公式

  使用$a_i^{(l)}$ 来表示每个神经元的输出,如此为**前向传播(forward propagation)**
  $$
  a_1^{(2)}=f(W_{11}^{(1)}x_1+W_{12}^{(1)}x_2+W_{13}^{(1)}x_3+b_1^{(1)})\\
  a^{(2)}_2=f(W^{(1)}_{21}x_1+W^{(1)}_{22}x_2+W^{(1)}_{23}x_3+b_2^{(1)})\\
  a_3^{(3)}=f(W_{31}^{(1)}x_1+W_{32}^{(1)}x_2+W_{33}^{(1)}x_3+b_3^{(1)})\\
  \quad \;h_{W,b}=a_1^{(3)}=f(W^{(2)}_{11}a^{(2)}_1+W^{(2)}_{12}a^{(2)}_2+W^{(2)}_{13}a^{(2)}_3)
  $$
  简化公示的表示：
  $$
  \qquad\qquad\qquad\; z^{(l)}_i=\sum_{j=1}^nW^{(l-1)}_{ij}x_j+b_i^{(l-1)}\\
  a^{(l)}=f(z_i^{(l)})
  $$
  继续简化:
  $$
  \qquad\; z^{(l+1)}=W^{(l)}a^{(l)}+b^{(l)}\\
  a^{(l+1)}=f(z^{(l+1)})
  $$

  ##### Backpropagation Algorithm(反向传播算法)

  ​	反向传播更新的是W，不是x

  * cost function(损失函数):
    $$
    J(W,b;x,y)=\frac{1}{2}||h_{W,b}(x)-y||^2
    $$

  * 再计算出每个Output节点的损失函数和，具体操作为将其相加

  * 更新节点$W_{ij}^{(l)}$ 和 $b_i^{(l)}$ (输出层):
    $$
    W^{(l)}_{ij}=W^{(l)}_{ij}−α\frac{∂}{∂W^{(l)}_{ij}}J(W,b)\\
    b^{(l)}_i=b^{(l)}_i−α\frac{∂}{∂b^{(l)}_i}J(W,b)
    $$

  

  * 如果是更新在隐藏层的神经元，则要将神经元的所有输出一起进行反向传播运算
  * 具体的例子请看[这里](<https://www.cnblogs.com/charlotte77/p/5629865.html>)

2. #### Convolutional Neural Networks (CNNs / ConvNets)

   * 卷积层，池化层，全连接层
   * 每层接受的输入为一个3D的形状
   * 卷积层和全连接层有参数（传递），激活函数和池化层无参数

   * 感受野(receptive field)：卷积核大小

   * 计算输出：$(W - F + 2P)/S +1$

     其中W 为输入大小，F为卷积核大小，P为zero-padding，S为卷积核步长

   * 参数共享？？