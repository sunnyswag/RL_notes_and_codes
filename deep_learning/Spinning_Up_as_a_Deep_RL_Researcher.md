## 一些深度学习的基础知识

1. #### [MLP](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)(多层神经网络)

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
     

     比较图如下：

     <img src="../assets/activation_functions.png" width=500>

     