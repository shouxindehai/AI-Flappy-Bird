import numpy as np

def q_learning(
        env, gamma, alpha, epsilon, num_episodes,Qtable, max_steps=np.inf):
    num_actions = env.num_actions
    policy = get_soft_greedy_policy(epsilon, Qtable)
    for _ in range(num_episodes):
        # initialise state初始化状态
        s = env.reset()
        steps = 0
        while not env.is_terminal() and steps < max_steps:
            # choose the action选动作
            a = choose_from_policy(policy, s)
            next_s, r = env.next(a) # 获取下一个状态和本次reward
            # update the Q function estimate # qlearning公式更新
            Qtable[s, a] += alpha * (r + gamma * np.max(Qtable[next_s, :]) - Qtable[s, a])
            # update the policy # 更新policy，只需要对当前状态更新,只修改s那一行
            policy[s, :] = get_soft_greedy_policy(
                epsilon, Qtable[s,:].reshape(1,num_actions)) #实际上也只传了s那一行的Q值进去,注意仍然2维
            s = next_s
            steps += 1
    # return the policy
    return Qtable


def choose_from_policy(policy, state):   #根据目前policy选择最好的action
    num_actions = policy.shape[1]
    # np.random.choice：[0,..,action数-1]的数组中随机抽取元素选取一个action值，实际上就是0,1之前选
    result = np.random.choice(num_actions, p=policy[state, :])#参数p，是一位数组，对应数值决定对应选取的概率，
    # 因为policy中元素每行必定是一个0一个1，随机选取成了必然由policy矩阵决定action
    return result


def get_soft_greedy_policy(epsilon, Q):
    greedy_policy = get_greedy_policy(Q)#一行(本次状态)必定一个0一个1
    policy = (1 - epsilon) * greedy_policy + epsilon * np.ones(Q.shape) / Q.shape[1] #Q.shape[1]==2
    # epsilon贪心算法，注意右边np.ones(Q.shape)/Q.shape[1]形成的是全0.5的1X2矩阵，即等概率policy，再与贪心policy加权平均
    return policy


def get_greedy_policy(Q):
    num_states, num_actions = Q.shape #num_states其实等于1
    policy = np.zeros((num_states, num_actions))#同形状0矩阵，只接收了了1X2的矩阵，因此是1X2的
    dominant_actions = np.argmax(Q, axis=1) #取出那一行最大元素的列标号(从0开始，实际上0或者是1)，组成一位数组，元素数是状态数。
    policy[np.arange(num_states), dominant_actions] = 1.#0矩阵policy中每行最大的位置设为1
    return policy
