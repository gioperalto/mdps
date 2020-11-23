import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

def q_learning(environment, gamma=0.95, alpha=0.1, epsilon=1.0, theta=0.0001, seed=93, policy=None, render=False):
        np.random.seed(seed)
        env.seed(seed)
        Q = defaultdict(lambda: np.zeros(env.action_space.n))

        if policy is None:
            pol = epsilon_greedy(env, Q, epsilon)

        decay_min, delta, deltas = 0.001, 0, []
        x, y, z = environment.observation_space
        n_states = x.n * y.n * z.n
        max_iters = n_states * 1000
        
        for i in range(max_iters):
            s = environment.reset()
            done = False
            
            while not done:
                if render:
                    environment.render()
                
                p = pol(s)
                a = np.random.choice(np.arange(len(p)), p=p)
                s_, reward, done, _ = environment.step(a)
                target = reward + gamma * np.max(Q[s_])
                td_delta = target - Q[s, a]
                Q[s, a] += alpha * td_delta
                delta = abs(td_delta[0])
                s = s_
                
            deltas.append(delta)
            # Eps decay
            epsilon = decay(epsilon, decay_min)
            if epsilon == decay_min and delta < theta: 
                print(f'Q-Learning converged in {i} iterations.')
                break
        
        return Q, deltas

def plot_data(info, data, n_episodes):
    plt.plot(np.arange(1, n_episodes+1, 1), data)
    plt.xlabel(info['x'])
    plt.ylabel(info['y'])
    plt.savefig('images/{}'.format(info['name']))
    plt.close()

def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0
    
    for episode in range(n_episodes):
        terminated = False
        state = environment.reset()
        
        while not terminated:
            action = np.argmax(policy[state])
            next_state, reward, terminated, info = environment.step(action)
            total_reward += reward
            state = next_state

            if terminated and reward == 1.0:
                wins += 1

    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward


if __name__ == "__main__":
    seed = 93 # Fixed seed
    # Number of episodes to play
    n_episodes = 10000

    solvers = [
        ('Q-Learning', q_learning),
    ]

    for iteration_name, iteration_func in solvers:
        env = gym.make('Blackjack-v0')
        policy, deltas = iteration_func(env)

        # Generate plots
        plot_data(
            {
                'x': 'Iterations',
                'y': 'Δ',
                'name': 'Deltas - Blackjack - {}'.format(iteration_name)
            },
            deltas,
            len(deltas)
        )

        wins, total_reward, average_reward = play_episodes(env, n_episodes, policy)

        print(f'{iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
        print(f'{iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')