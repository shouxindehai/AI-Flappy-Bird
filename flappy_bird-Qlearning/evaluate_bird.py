from flappy import FlappyBird
import numpy as np
import matplotlib.pyplot as plt
from algorithms import q_learning

def evaluate_learning(
        sim,series_size=50, num_series=5, gamma=0.99, alpha=0.7,
        epsilon=0.0001):
    '''
    :param sim: The queuing object
    :param series_size: the size of series
    :param num_series: the number of series
    :return: a result policy
    '''
    # initialise Q values

    #Q = 0*np.ones((sim.num_states,sim.num_actions))  #选择重新开始训练，一开始初始化Q表为全0
    Q = np.load('save.npy')
    print(Q.shape)
    figrew, axrew = plt.subplots()
    total_reward_seq = [0]
    total_episodes = 0

    for series in range(num_series):
        print("series = %r/%d" % (series,num_series) ) # Print the current stage
        rewardlist=[]
        for episode in range(series_size):
            Q = q_learning(              # 调用Qlearning，这里面再调用游戏机制
                sim, gamma=gamma, alpha=alpha, epsilon=epsilon,
                num_episodes=1, Qtable=Q)
            rewardlist.append(sim.score)# 每次episode后得到一个reward值，接续到rewardlist末尾
            total_episodes += 1
        total_reward_seq.append(np.mean(np.array(rewardlist))) # 这一系列的episode的rewardlist，转化成数组求平均值，接续到总reward列表里
        np.save('save01.npy',Q)                  # 每一系列(series)中的所有episode结束，保存Q
    total_reward_seq = np.array(total_reward_seq) # 当一切训练结束，总reward列表转化成数组，方便plot打印图
    #下面是作图
    axrew.plot(
        np.arange(0, total_episodes+1, series_size),
        total_reward_seq)
    axrew.set_xlabel("episodes")
    axrew.set_ylabel("average score")
    return

#下面是主程序，首先调用“主要的函数evaluate_learning”..
evaluate_learning(sim=FlappyBird())#sim是FlappyBird类的实例，相当于环境，包含整个游戏机制，包括状态数动作数..
#下面是作图相关
plt.title('Average score per series')
plt.tight_layout()
plt.show()
