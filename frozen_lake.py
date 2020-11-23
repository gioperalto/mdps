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

def create_policy(Q):
    outputs = { 0: '<', 1: 'v', 2: '>', 3: '^' }
    mapped_actions = []
    inputs = np.argmax(Q, axis=1)

    for i in inputs:
        mapped_actions.append(outputs[i])

    return mapped_actions

def print_policy(actions, Q):
    c, s, rows = 0, '', int(Q.shape[0] ** 0.5)

    for x in range(rows):
        
        s += '\n'

        for y in range(rows):
            s += '| {} '.format(actions[c])
            c += 1
        s += '|'

    s += '\n'

    return s

def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    evaluation_iterations = 1
    n_states, n_actions = environment.observation_space.n, environment.action_space.n
    V, deltas = np.zeros(n_states), []
    for i in range(int(max_iterations)):
        delta = 0

        for state in range(n_states):
            v = 0

            for action, action_probability in enumerate(policy[state]):
                for state_probability, next_state, reward, terminated in environment.P[state][action]:
                    v += action_probability * state_probability * (reward + discount_factor * V[next_state])
        
            delta = max(delta, np.abs(V[state] - v))
            V[state] = v
        
        deltas.append(delta)
        evaluation_iterations += 1
        
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return V, deltas

def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])
    return action_values

def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
    n_states, n_actions = environment.observation_space.n, environment.action_space.n
    policy = np.ones([n_states, n_actions]) / n_actions
    evaluated_policies = 1

    for i in range(int(max_iterations)):
        stable_policy = True
        V, deltas = policy_evaluation(policy, environment, discount_factor=discount_factor)

        for state in range(n_states):
            current_action = np.argmax(policy[state])
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            best_action = np.argmax(action_value)

            if current_action != best_action:
                stable_policy = True
                policy[state] = np.eye(n_actions)[best_action]
        evaluated_policies += 1

        if stable_policy:
            print(f'Evaluated {evaluated_policies} policies.')
            return policy, deltas

def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    n_states, n_actions = environment.observation_space.n, environment.action_space.n
    V, deltas = np.zeros(n_states), []

    for i in range(int(max_iterations)):
        delta = 0

        for state in range(n_states):
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            best_action_value = np.max(action_value)
            delta = max(delta, np.abs(V[state] - best_action_value))
            V[state] = best_action_value
        
        deltas.append(delta)

        if delta < theta:
            print(f'Value Iteration converged in {i} iterations.')
            break

    policy = np.zeros([n_states, n_actions])
    for state in range(n_states):
        action_value = one_step_lookahead(environment, state, V, discount_factor)
        best_action = np.argmax(action_value)
        policy[state, best_action] = 1.0

    return policy, deltas

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

        decay_min, delta, deltas = 0.001, 0, []
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
                
            deltas.append(delta)
            # Eps decay
            epsilon = decay(epsilon, n_states, decay_min)

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
            policy, deltas = iteration_func(environment)
            plot_data(
                {
                    'x': 'Iterations',
                    'y': 'Δ',
                    'name': 'Deltas - {} - {}'.format(key, iteration_name)
                },
                deltas,
                len(deltas)
            )
            actions = create_policy(policy)
            # Print out policy
            print(print_policy(actions, policy))
            # Apply best policy to the real environment
            wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
            print(f'{key} {iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
            print(f'{key} {iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')