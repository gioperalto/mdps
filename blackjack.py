import gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def epsilon_greedy(env, Q, epsilon):
    """ Create epsilon greedy action policy
    @param env: Environment
    @param Q: Q table
    @param epsilon: Probability of selecting random action instead of the 'optimal' action
    @returns Epsilon-greedy-action Policy function with Probabilities of each action for each state
    """
    def policy(obs):
        P = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n  #initiate with same prob for all actions
        best_action = np.argmax(Q[obs])  #get best action
        P[best_action] += (1.0 - epsilon)
        return P
    return policy 

def decay(arg, min_arg):
    decay_rate = 0.9999
    return arg * decay_rate if (arg * decay_rate) >= min_arg else min_arg

def sarsa(env, n_episodes, seed, gamma=0.99, alpha=0.1, epsilon=0.4, policy=None, render=False):
    """
    @param alpha: learning rate
    @param gamma: decay factor
    @param epsilon: for exploration
    """
    np.random.seed(seed)
    env.seed(seed)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    if policy is None:
        pol = epsilon_greedy(env, Q, epsilon)

    wins, total_reward, decay_min = 0, 0, 0.001

    for episode in tqdm(range(n_episodes)):
        s = env.reset()
        p = pol(s)
        a = np.random.choice(np.arange(len(p)), p=p)
        done = False
        
        while not done:
            if render:
                env.render()
            
            s_, reward, done, _ = env.step(a)
            p_ = epsilon_greedy(env, Q, epsilon)(s_)
            a_ = np.random.choice(np.arange(len(p_)), p=p_)
            delta = reward + gamma * Q[s_][a_] - Q[s][a]
            Q[s][a] += alpha * delta
            s, a = s_, a_
            # Add to total reward
            total_reward += reward

        if reward == 1.0:
            wins += 1

        # Eps decay
        epsilon = decay(epsilon, decay_min)

    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward

if __name__ == "__main__":
    seed = 93 # Fixed seed
    # Number of episodes to play
    n_episodes = 500000
    
    env = gym.make('Blackjack-v0')
    # SARSA
    wins, total_reward, average_reward = sarsa(env, n_episodes, seed)
    print(f'SARSA :: number of wins over {n_episodes} episodes = {wins}')
    print(f'SARSA :: average reward over {n_episodes} episodes = {average_reward} \n\n')