## 实现RL基本算法

**REINFORCE**<br>
**REINFORCE with baseline**<br>
**Q-learning**<br>
**DQN**<br>
**DoubleDQN**<br>
**DuelingDQN**<br>
**A2C**<br>
**DDPG**<br>
**PPO**<br>
**TD3**<br>
**SAC**<br>

## TODO

* 测试类
* replaybuffer类
* plot类


## 算法复现的步骤:
1. 查看算法伪代码理解算法思路
2. 以伪代码为基础，结合网上博客的分析，得出自己的理解
3. 找两到三份书写正确，书写规范的代码供参考
4. 编码&&debug
    1. code:记得附上ref链接
    2. debug:检查是否出现各种矩阵shape不正确的错误
5. 完善步骤2的理解
6. 查看论文

## Tag 1:
**应该是这样子的: <br>
在DPG之前很难(或者无法)做到连续动作空间的决策，因为为了exploration，使用的都是随机策略<br>。
而最早使用随机策略进行连续动作空间决策的是A3C， why?
**
**记得附上参考连接，体现素质**

## Tag 2:
如何调参debug:
1. 首先检查一遍代码的结构与实现是否有误
2. 检查超参设置是否合理
3. 检查reward,loss(actor&&critic),episode_step是否正常
reward是否达到预期<br>
loss是否有波动，波动范围如何<br>
在step较多的episode，reward是增加还是减少
4. 检查action。连续动作的action是否一直处在low和high action处。如此就需要更改defaut的网络初始化权重<br>
貌似nn.Sequential和手动定义时nn的权重的初始化时不一样的<br>
5. 权重初始化方式(和seed有关,defaut seed(1)):
默认权重初始化方式:
        ```
        # in_dim神经元数目
        # 如nn.Linner(400, 300),stdv = 1./sqrt(400)=0.05
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        ```
手动权重初始化:<br>
方式一(手动定义nn):
        ```
        init_w = 3e-3
        # 随机生成-init_w 和init_w之间的数
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        ```
方式二(使用nn.Sequential定义nn):
        ```
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.1)
        # 类中:
        self.apply(init_weights)
        ```

## Tag 3:
**对输入nn的state做归一化**<br>
**如果看到Loss function非常平稳无偏差，那一定有问题**
**书写update计算loss时，调试去看维度是否出现了问题，是否出现了不必要的广播导致了错误的出现, **
**该squeeze(),unsqueeze()的地方一个都不能少

## Tag 4:
**Actor_Critic计算TD_target的两种方式:<br>
1. 计算 $G_t$,直接 TD_target = $[G_t], t=1,2,3,\dots$(此时 $G_t$ 为TD_target的无偏估计)
    1. G_t 初始化为0
    2. G_t 初始化为next_state的value
    3. G_t 全为next_state的value
2. 计算下一个状态的value, TD_target = reward + gamma * model(next_state)
3. 具体选择哪个时看哪个的方差较小，再决定选择哪个
**
### 不同方法的效果(A2C)
**1.1:**
<img src="../assets/TD_target_first_way_initial_with_gt_target=0.png"></img>
**1.2:**
<img src="../assets/TD_target_first_way_initial_with_gt_target=next_value.png"></img>
**1.3:**
<img src="../assets/TD_target_first_way_with_all_gt_target=next_value.png"></img>
**2.:(无法收敛)**
<img src="../assets/TD_target_second_way.png"></img>
### 总结:
这只是一个公式在实现上的差别，具体的可以打印出各自的方差再做出选择