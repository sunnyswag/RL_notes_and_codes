## 暂且记为力的方向向量问题？

1. 问题：```Pendulum-v0``` 环境收敛非常慢，```MountainCarContinuous-v0``` 环境学习非常快

   在debug的过程中，发现自己写的**对action添加噪声**的操作有问题。

   

   ```Algorithm ``` : ```DDPG(确定性策略)```

   ```Environment``` : ```MountainCarContinuous-v0(连续动作控制，spare reward)```

* 我的代码 :

```python
class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        # get 定义立即执行函数
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        # self.state += [0.0] + 0.3 * np.random.randn(1)
        self.state += self.theta * (self.mu - \			                                         			  	self.state)+np.random.randn(self.action_dim) 
        				# *np.sqrt(0.01) * self.sigma
        return self.state
    
    def get_action(self, action, t=0):
        # self.sigma = self.max_sigma
        self.sigma = self.max_sigma - (self.max_sigma \
                      - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + self.evolve_state(), self.low, self.high)
```

![DDPG_for_mountain_car.png](../assets/DDPG_for_mountain_car.png)

* (可能)正确的代码 :

```python
class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.1, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        # get 定义立即执行函数
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        # self.state += [0.0] + 0.3 * np.random.randn(1)
        self.state = self.state + self.theta * (self.mu - self.state)+ \
                        np.random.randn(self.action_dim) *np.sqrt(0.3) * self.sigma
        return self.state
    
    def get_action(self, action, t=0):
        # self.sigma = self.max_sigma
        self.sigma = self.max_sigma - (self.max_sigma \
                      - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + self.evolve_state(), self.low, self.high)
```

![DDPG_not_suit_for_mountain_car.png](../assets/DDPG_not_suit_for_mountain_car.png)

​	**收敛更快，可以有效地学习**

* 不同之处：

```python
class OUNoise:
    def evolve_state(self):
		self.state = self.state + xxx
        
class OUNoise:
    def evolve_state(self):
		self.state += xxx
```

2. Experiment 1

* Test 1

```python
class Test:
    def __init__(self):
        self.state = np.zeros(1)
    
    def sample_right(self):
        self.state = self.state + np.random.randn(1)
        return self.state
    
    def sample_error(self):
        self.state += np.random.randn(1)
        return self.state

plt.figure(figsize=(16, 6))

plt.subplot(121)
test = Test()
right_sample = [test.sample_right() for _ in range(1000)]
error_sample = [test.sample_error() for _ in range(1000)]

plt.plot(right_sample, label="a")
plt.plot(error_sample, label="b")
plt.legend()

plt.subplot(122)
test1 = Test()
right_sample = [test1.sample_right()[0] for _ in range(1000)]
error_sample = [test1.sample_error()[0] for _ in range(1000)]

plt.plot(right_sample, label="a")
plt.plot(error_sample, label="b")
plt.legend()

plt.show()
```

* Distinguish

```python
right_sample = [test.sample_right() for _ in range(1000)]
error_sample = [test.sample_error() for _ in range(1000)]

right_sample = [test1.sample_right()[0] for _ in range(1000)]
error_sample = [test1.sample_error()[0] for _ in range(1000)]
```

* Plot

![utils_ddpg_noise.png](../assets/utils_ddpg_noise.png)

3. 猜测

   在一个episode当中，所产生的噪声为一个确定值。也就是说，每个action都加上一个确定值。而这个确定值的噪声相当于作用在动作方向的一个向量。让小车在动作维度(一维)获得单个方向的力，帮助小车更快获得奖励。

4. Experiment 2

* use uniform noise

```python
for episode in range(episodes):
	noise = np.random.rand(1)
    for step in range(steps):
        action = xxx
        action = action + noise
        env.step(action)
```

![episod_uniform_noise.png](../assets/episod_uniform_noise.png)