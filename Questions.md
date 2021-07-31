# Q & A（中英结合）

## Diffences among DP、MC、TD

### Monte-Carlo Reinforcement Learning (MC Learning )

蒙特卡洛强化学习指：在**不清楚MDP状态转移及即时奖励**的情况下，代理直接从经历**完整的Episode**来学习状态价值，通常情况下某状态的价值等于在多个Episode中以该状态算得到的所有收获的平均。

Monte Carlo reinforcement learning means that without knowing the MDP state transition and immediate reward, the agent directly learns the state value from experiencing a complete episode through interaction with the environment. Generally, the value of a state is equal to the average of all gains calculated in multiple episodes.

1.没有模型（代理不知道状态MDP转换）

2.代理人从抽样的经验中学习

3.通过体验所有采样事件的平均收获，学习策略π下的状态值vπ（s）

4.只有在完成一个episode后，才会更新值。

5.没有 bootstrapping

6.只能用于情景问题

1. There is no model (agent does not know state MDP transitions)

2. agent **learns** from **sampled** experience

3. learn state value vπ(s) under policy π by experiencing **average** return from all sampled episodes 

   ![[公式]](https://www.zhihu.com/equation?tex=v_%7B%5Cpi%7D%28s%29+%3D+E_%7B%5Cpi%7D+%5B+G_%7Bt%7D+%7C+S_%7Bt%7D+%3D+s+%5D)

4. only after a **complete episode**, values are updated.

5. There is no bootstrapping

6. Only can be used in **episodic problems**



![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) ：实际收获，是基于某一策略的状态价值的**无偏**估计， ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻状态 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D) 的收获：

![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) ：The concept *return* is an unbiased estimate of the state value under under policy π. At time ![[公式]](https://www.zhihu.com/equation?tex=t) , the *return* of ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D) is defined as,

![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D+%3D+R_%7Bt%2B1%7D+%2B+%5Cgamma+R_%7Bt%2B2%7D+%2B+...+%2B+%5Cgamma%5E%7BT-1%7D+R_%7BT%7D)





 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bt%2B1%7D)：agent在![[公式]](https://www.zhihu.com/equation?tex=t) 时刻，在状态 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D) 执行一个行为 ![[公式]](https://www.zhihu.com/equation?tex=a) 后**离开该状态获得的即时奖励**。

 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bt%2B1%7D): The immediate reward obtained by an agent when leaving the state![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D) after performing an action.



MC使用实际的收获（return） ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) 来更新价值（Value)

MC updates the value with the actual return,

![[公式]](https://www.zhihu.com/equation?tex=V%28S_%7Bt%7D%29+%5Cleftarrow+V%28S_%7Bt%7D%29+%2B+%5Calpha+%28G_%7Bt%7D+-+V%28S_%7Bt%7D%29%29)

**BootStrapping** 指的就是TD目标值 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bt%2B1%7D+%2B+%5Cgamma+V%28S_%7Bt%2B1%7D%29+) 代替收获 ![[公式]](https://www.zhihu.com/equation?tex=G_t) 的过程，暂时把它翻译成“**引导”**。



###  Temporal-Difference Learning (TD)

时序差分学习：和蒙特卡洛学习一样，它也**从Episode学习**，**不需要**了解**模型**本身；但是它可以学习**不完整**的Episode，通过自身的**引导（bootstrapping）**，猜测Episode的结果，**持续更新**这个猜测。

TD, like Monte Carlo learning, also learns from episodes without understanding the model itself; However, it can learn incomplete episodes, guess the results of episodes through its bootstrapping, and continuously update the guess.

与MC学习更新value的公式相比，TD学习**将Gt替换为TD目标**

Compared with the formula that MC updates value, TD replaces the return ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) with TD target ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bt%2B1%7D+%2B+%5Cgamma+V%28S_%7Bt%2B1%7D%29+)

**BootStrapping**就是指这一**替换过程**

Bootstrapping refers to this replacement process



### Difference

#### 1.TD can learn in an ongoing environment before knowing the results , while MC must wait until the end of an episode.

TD学习可以在了解结果之前在持续的环境中学习，而MC学习必须等到一个episode结束。

#### 2.MC updates the value with the actual return, while TD updates the value by calculating the current estimated return based on the estimated value of the next state. The former return is unbiased estimation, while the latter is biased estimation.

2.MC用实际收益更新value，而TD则根据下一个状态的估计value计算当前估计return来更新value值。前者为无偏估计，后者为有偏估计。

#### 3....

##  DQN

## SARSA

