### Lectur e 1: Introduction to Reinforcement Learning  [[slides](./intro_RL.pdf)]

1. agent state, enviroment state, information state

   information state (Markov state) 包含历史中所有有用的信息
   $$
   \mathbb{P}[S_{t+1} | S_t] = \mathbb{P}[S_{t+1} | S_1,...,S_t]
   $$
   “The future is independent of the past given the present” 

   简而言之，未来与现在的过去状态无关

2. Major Components: Policy, Value function and Model
3. Value Based, Policy Based, Actor Critic ,Model Free and Model Based
4. Exploration and Exploitation

### Lecture 2: Markov Decision Process

1. Markov Process

   马尔可夫过程可以定义为一个元组($S,P$)

   $S$ 为有限的状态集

   $P$ 为状态转移矩阵
   $$
   P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s]
   $$

2. Markov Reward Process

   马尔可夫奖励过程可以定义为一个元组($S,P,{\color{red}R, \gamma}​$)

   $S $ 为有限的状态集

   $P​$ 为状态转移矩阵

$$
P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s]
$$

​	   **${\color{red}R \;为奖励函数，R_s = \mathbb{E}[R_{t+1}\;|\;S_t=s]}$**，可以理解为在状态t可获得的下一个状态的奖励

​	   $\color{red}{\gamma\;为衰减因子,\gamma \in [0,1]}​$

​	   定义$G_t​$，为随着次数变化的奖励衰减和
$$
G_t=r_t+γr_{t+1}+γ^2r_{t+2}+γ^3r_{t+3}+\cdots=\sum\limits_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$
​	   紧接着，我们定义Value Function，$v(s)​$ 表示在状态s的 $G_t​$
$$
v(s) = \mathbb{E}[G_t | S_t = s]
$$
​	   接着，推导出Bellman平衡(为递归定义)：

​	$v(s) = \mathbb{E}[G_t | S_t = s]$

​		$= \mathbb{E}[R_{t+1} + γR_{t+2} + γ^2R_{t+3} + \cdots| S_t = s]​$

​		$= \mathbb{E}[R_{t+1} + γ(R_{t+2} + γR_{t+3} + \cdots)| S_t = s]$

​		$= \mathbb{E}[R_{t+1} + γG_{t+1}| S_t = s]​$

​		$= \mathbb{E}[R_{t+1} + γv(S_{t+1})| S_t = s]$

​	$R$ 从t+1开始是因为，需要和环境交互之后才会得到一个reward

​	更为简便的，我们使用矩阵来计算和表示：
$$
v = R +\gamma Pv
$$

$$
\begin{equation}
\left[
    \begin{matrix}
    v(1)\\
    \vdots\\
    v(n)
    \end{matrix}
\right]
=
\left[
    \begin{matrix}
    R_1\\
    \vdots\\
    R_n
    \end{matrix}
\right]
+\gamma
\left[
    \begin{matrix}
    P_{11} & \cdots & P_{1n}\\
 	\vdots & \ddots & \vdots\\
 	P_{n1} & \cdots & P_{nn}
    \end{matrix}
\right]
\left[
    \begin{matrix}
    v(1)\\
    \vdots\\
    v(n)
    \end{matrix}
\right]
\end{equation}
$$
​	接着我们再做一个简单的转换吧：
$$
\qquad\qquad\qquad \;\;\ v = R+\gamma Pv\\
(1-\gamma P)v = R\\
\qquad\qquad\qquad\quad\quad \;\; v = (1-\gamma P)^{-1}R
$$
​	

3. Markov Decision Process

   马尔可夫决策过程是马尔可夫奖励过程+决策：

   马尔可夫决策过程可以表示为一个元组$(S,{\color{red}A},P,R,\gamma)​$

   $S ​$ 为有限的状态集

   ${\color{red}A \;为有限的动作集}​$

   $P​$ 为状态转移矩阵
   $$
   P_{ss'}^{\color{red}a} = \mathbb{P}[S_{t+1} = s' | S_t = {\color{red}a}]
   $$
    $R$ 为奖励函数*，$R_s^{\color{red}a} = \mathbb{E}[R_{t+1}\;|\;S_t=s,A_t={\color{red}a}]$*

    $\gamma$ 为衰减因子，$\gamma \in [0,1]$

4. Polices

   策略$\pi$，是一个从状态到动作的分布，只取决于当前状态
   $$
   \pi(a|s)=\mathbb{P}[A_t=a\;|\;S_t=s]
   $$

5. Bellman Expectation Equation

   为了引出贝尔曼平衡讲了很多其他的概念，但简单点理解吧，贴上一个简单点的公式
   $$
   v_\pi=R^\pi+\gamma P^\pi v_\pi
   $$

6. 找到一个 Optimal Policy

   对于每一个MDP，都存在一个确定的最佳的 Policy

   如果我们知道了$q_*(s,a)$ ，我们就能马上知道optimal policy
   $$
   f(n)=
   \begin{cases}
   1\quad if\;a={argmax\\a \in A}\;q_*(s,a) \\
   0\quad otherwise
   \end{cases}
   $$
   

## Lecture 3: Planning by Dynamic Programming

1. 如何提升一个Policy

    1）Evaluate一个Policy $\pi$ ，
   $$
   V_\pi(s)=\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\dots|S_t=s]
   $$
    2）使用 $v_\pi$ 贪婪的更新这个Policy
   $$
   \pi'=greedy(v_\pi)
   $$
   

    3）具体更新迭代步骤，进行单步更新操作。具体意思为：当前动作的最大的q，需要大于或等于当前所能采取的任意动作所获得的q
   $$
   \pi' (s)={argmax \\a\in A}\;q_\pi (s,a)\\
   q_\pi (s,\pi' (s)) = {max \\a\in A} q_\pi (s,a) ≥ q_\pi (s,π(s)) = v_\pi (s)
   $$
    4）稳定之后

$$
q_\pi (s,\pi' (s)) = {max \\a\in A} q_\pi (s,a) = q_\pi (s,π(s)) = v_\pi (s)
$$

2. 如何提升一个Value function

   1）更新迭代
   $$
   v_∗(s) \leftarrow {max\\ a\in A} R^a_s + \gamma \;\sum_{s'\in S}
   P^a_{ss'}v_∗(s')
   $$
   2）中间步骤无法代表任何一个policy

