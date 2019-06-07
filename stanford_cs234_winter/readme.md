## Lecture 1 - Introduction

* 课堂笔记

  1. 强化学习算法组成：

     model: 对agent的action做出反应，返回reward和state等参数

     policy:agent从状态到动作的映射函数。Deterministic和Stochastic

     value function:在某个状态的未来奖励或者在特定策略下的动作

  2. agent类型

     model-based: 有model，may or may not have policy and /or value function

     model-free : 无model, 有value function and / or policy funcion

  3. exploration和exploitation

     exploration(探测): 尝试新的东西，让agent在未来做出更好的决策

     exploitation(开发): 根据过去的经历，选择reward最大的action

---

## Lecture 2 - Given a Model of the World

* 课堂笔记

  1. Markov Reward Process(MRP)

     S (有限的)状态 $ s\in S$

     P 状态链，$ P(s_{t+1}=s`|s_t=s,a_t=a) ​$

     R 奖励函数，$R(s_t=s)=\mathbb{E}[r_t|s_t=s]$

      $\gamma \;$折扣值 $γ \in [0,\;1]$

  2. Markov Decision Process(MDP)

     S (有限的)状态 $ s\in S$

     A (有限的)动作集 $ a \in A$

     P 状态链，$ P(s_{t+1}=s`|s_t=s,a_t=a) $

     R 奖励函数，某个动作的奖励$ R(s_t = s,a_t =a)=\mathbb{E}[r_t|s_t=s,a_t=a] $

     $γ $ 折扣值 $γ \in [0,\;1]$

     MDP可以表示为一个元组$(S, A, P, R, γ)​$

     **将MRP和MRP理解为两种不同的表现形式**

  3. MDP Value Function

     定义$G_t​$
     $$
     G_t=r_t+γr_{t+1}+γ^2r_{t+2}+γ^3r_{t+3}+\cdots
     $$

     用$G_t$来定义$V(s)$
     $$
     V(s)=\mathbb{E}[G_t|s_t=s]=\mathbb{E}[r_t+γr_{t+1}+γ^2r_{t+2}+γ^3r_{t+3}+\cdots|s_t=s]
     $$
     考虑 $γ​$ = 0 ，$γ​$ = 1的情况

     对于有限状态的MRP， 我们可以用矩阵来表示$V(s)$

  4. MDP Policies

     Plolicy:

  $$
  \pi(a|s)=P(a_t=a|t_s=s)
  $$

  ​	$MRP(S,R^\pi,P^\pi,\gamma)$，其中：

  ​	reward:
  $$
  R^\pi(s)=\sum\limits_{a\in A}\pi(a|s)P(s'|s,a)
  $$
  ​	状态转移模型:
  $$
  P^\pi(s'|s) =\sum\limits_{a\in A}\pi(a|s)P(s'|s,a)
  $$

  5. State-Action Value Q   

     定义：

  $$
  Q^\pi(s,\ a) = R(s,\ a)\;+\;γ\sum\limits_{s'\in S}P(s'|s,\;a)V^\pi(s')
  $$

  6. Policy的迭代更新

     对于每一个 $s \in S$ 和每个$a \in A$，计算Q值：
     $$
     Q^{\pi_i}(s,\ a) = R(s,\ a)\;+\;γ\sum\limits_{s'\in S}P(s'|s,\;a)V^{\pi_i}(s')
     $$
     对于所有的$ s \in S $ ，计算更新$\pi_{i+1}$
     $$
     \pi_{i+1}(s) = argm{ax \\ a}Q^{\pi_i}(s, a)\nu s\in S
     $$

  7. Bellman Equation(贝尔曼平衡)

     感觉给的定义比较模糊，意思大概就是用他来更新趴

