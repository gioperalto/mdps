import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def epsilon_greedy(Q, epsilon, state):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param state current state
    """
    if np.random.random() <= epsilon:
        return np.random.randint(Q.shape[-1])
    else:
        return np.argmax(Q[state])

def truncate(decimal_value):
    finished, decimal_places, trunc_decimal_value = False, 0, decimal_value
    
    while not finished:
        decimal_places += 1
        trunc_decimal_value *= 10
        if trunc_decimal_value >= 1:
            finished = True

    decimal_places = 10 ** decimal_places

    return round(trunc_decimal_value) / decimal_places

def decay(arg, n_states, min_arg):
    state_ratio = (1/n_states**2.5)
    decay_rate = truncate(state_ratio)
    decayed_eps = arg - decay_rate
    return decayed_eps if decayed_eps >= min_arg else min_arg

def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    # Number of evaluation iterations
    evaluation_iterations = 1
    # Initialize a value function for each state as zero
    V = np.zeros(environment.nS)
    # Repeat until change in value is below the threshold
    for i in range(int(max_iterations)):
        # Initialize a change of value function as zero
        delta = 0
        # Iterate though each state
        for state in range(environment.nS):
            # Initial a new value of current state
            v = 0
            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(policy[state]):
                # Check how good next state will be
                for state_probability, next_state, reward, terminated in environment.P[state][action]:
                    # Calculate the expected value
                    v += action_probability * state_probability * (reward + discount_factor * V[next_state])
        
            # Calculate the absolute change of value function
            delta = max(delta, np.abs(V[state] - v))
            # Update value function
            V[state] = v
        evaluation_iterations += 1
        
        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return V

def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])
    return action_values

def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
    # Start with a random policy
    #num states x num actions / num actions
    policy = np.ones([environment.nS, environment.nA]) / environment.nA
    # Initialize counter of evaluated policies
    evaluated_policies = 1
    # Repeat until convergence or critical number of iterations reached
    for i in range(int(max_iterations)):
        stable_policy = True
        # Evaluate current policy
        V = policy_evaluation(policy, environment, discount_factor=discount_factor)
        # Go through each state and try to improve actions that were taken (policy Improvement)
        for state in range(environment.nS):
                # Choose the best action in a current state under current policy
            current_action = np.argmax(policy[state])
            # Look one step ahead and evaluate if current action is optimal
            # We will try every possible action in a current state
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            # Select a better action
            best_action = np.argmax(action_value)
            # If action didn't change
            if current_action != best_action:
                stable_policy = True
                # Greedy policy update
                policy[state] = np.eye(environment.nA)[best_action]
        evaluated_policies += 1
        # If the algorithm converged and policy is not changing anymore, then return final policy and value function
        if stable_policy:
            print(f'Evaluated {evaluated_policies} policies.')
            return policy

def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    # Initialize state-value function with zeros for each environment state
    V = np.zeros(environment.nS)
    for i in range(int(max_iterations)):
        # Early stopping condition
        delta = 0
        # Update each state
        for state in range(environment.nS):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_value)
            # Calculate change in value
            delta = max(delta, np.abs(V[state] - best_action_value))
            # Update the value function for current state
            V[state] = best_action_value
            # Check if we can stop
        if delta < theta:
            print(f'Value Iteration converged in {i} iterations.')
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([environment.nS, environment.nA])
    for state in range(environment.nS):
        # One step lookahead to find the best action for this state
        action_value = one_step_lookahead(environment, state, V, discount_factor)
        # Select best action based on the highest state-action value
        best_action = np.argmax(action_value)
        # Update the policy to perform a better action at a current state
        policy[state, best_action] = 1.0
    return policy

def q_learning(environment, gamma=0.95, alpha=0.1, epsilon=1.0, theta=0.0001, seed=93, policy=None, render=False):
        """
        @param policy strategy for choosing actions
        @param render whether to display UI
        @param verbose additional logging 
        """
        if policy is None:
            policy = epsilon_greedy

        np.random.seed(seed)
        environment.seed(seed)

        decay_min, delta, theta_chain = 0.001, 0, 0
        n_states, n_actions = environment.observation_space.n, environment.action_space.n
        Q, max_iters = np.zeros((n_states, n_actions)), n_states * 1000
        
        for i in range(max_iters):
            s = environment.reset()
            done = False
            
            while not done:
                if render:
                    environment.render()
                
                a = policy(Q=Q, epsilon=epsilon, state=s)
                s_, reward, done, _ = environment.step(a)
                target = reward + gamma * np.max(Q[s_])
                td_delta = target - Q[s, a]
                Q[s, a] += alpha * td_delta
                delta = abs(td_delta)
                s = s_

            # Eps decay
            epsilon = decay(epsilon, n_states, decay_min)

            if epsilon == decay_min and delta < theta: 
                print(f'Q-Learning converged in {i} iterations.')
                break
        
        # average_reward = total_reward / n_iter
        return Q

def plot_rewards(data, rewards, n_episodes):
    y = rewards/np.arange(1, n_episodes+1, 1)
    plt.plot(np.arange(1, n_episodes+1, 1), y)
    plt.xlabel(data['x'])
    plt.ylabel(data['y'])
    plt.title(data['title'])
    plt.save('images/{}'.format(data['name']))

def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
        state = environment.reset()
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(policy[state])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)
            # Summarize total reward
            total_reward += reward
            # Update current state
            state = next_state
            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward

if __name__ == "__main__":
    n_episodes, maps = 10000, {
        '4x4': [
            'SFFF',
            'FHFH',
            'FFFH',
            'HFFG'
        ],
        '8x8': [
            'SFFFFFFF',
            'FFFFFFFF',
            'FFFHFFFF',
            'FFFFFHFF',
            'FFFHFFFF',
            'FHHFFFHF',
            'FHFFHFHF',
            'FFFHFFFG'
        ],
    }

    for key, value in maps.items():
        # Functions to find best policy
        solvers = [
            ('Policy Iteration', policy_iteration),
            ('Value Iteration', value_iteration),
            ('Q-Learning', q_learning),
        ]
        for iteration_name, iteration_func in solvers:
            # Load a Frozen Lake environment
            environment = gym.make('FrozenLake-v0', desc=value)
            # Search for an optimal policy using policy iteration
            policy = iteration_func(environment)
            # Apply best policy to the real environment
            wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
            print(f'{key} {iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
            print(f'{key} {iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')