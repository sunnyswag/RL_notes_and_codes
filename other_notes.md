### model-based和model-free的区别

​	用一句话总结*“If, after learning, the agent can make predictions about what the next state and reward will be before it takes each action, it's a model-based RL algorithm.If ti can't, then it's a model-free algorithm”*

​	举个例子，如果agent站在门外而此时门关着的没法打开，model-free的方法会去尝试撞门、打开门、或者去往其他方向。而model-based的方法会直接选择去往其他方向并获得reward。

​	因此，model-based的方法可称为模仿学习，个人认为并不能当作监督学习，因为监督学习中会有label，而model-based的方法虽然有提供数据但并没有label。