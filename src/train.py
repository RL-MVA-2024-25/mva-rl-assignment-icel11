import random
from collections import namedtuple, deque
from joblib import dump, load

import numpy as np
from gymnasium.wrappers import TimeLimit
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm

from env_hiv import HIVPatient

env1 = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
env2 = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

GAMMA = 0.95
N_ACTIONS = env1.action_space.n
N_OBSERVATIONS = env1.observation_space.shape[0]

class ProjectAgent:
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(N_ACTIONS)
        return greedy_action(self.q_function, observation, N_ACTIONS)

    def save(self, path):
        pass

    def load(self):
        print('Loading model...')
        with open('q_function_best.sav', 'rb') as file:
            self.q_function = load(file)
        print('Model loaded')


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity, env):
        self.memory = deque([], maxlen=capacity)
        self.fill(env, capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def fill(self, n_samples, q_model=None, epsilon=0.1):
        additions = 0
        while additions < n_samples:
            if np.random.uniform(0, 1) < 0.75:
                env = env1
            else:
                env = env2
            state = env.reset()
            done = False
            while not done:
                if q_model is None:
                    action = env.action_space.sample()
                else:
                    if isinstance(state, tuple):
                        state = state[0]
                    action = greedy_action(q_model, state, N_ACTIONS, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                done = terminated or truncated
                self.push(state, action, reward, next_state, done)
                additions += 1
                state = next_state

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

def rf_fqi(memory, iterations, nb_actions, gamma, q_function = None):
    if q_function is not None:
        last_q_function = q_function
        memory.fill(10000, q_model=last_q_function, epsilon=0.2)
    else:
        last_q_function = None
    for iter in tqdm(range(iterations)):
        states, actions, rewards, next_states, dones = zip(*memory.memory)
        states = [s[0] if isinstance(s, tuple) else s for s in states]
        next_states = [s[0] if isinstance(s, tuple) else s for s in next_states]
        S = np.array(states)
        A = np.array(actions)
        R = np.array(rewards)
        S2 = np.array(next_states)
        D = np.array(dones)
        nb_samples = S.shape[0]
        SA = np.column_stack((S, A))
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = last_q_function.predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = ExtraTreesRegressor(n_estimators=150, max_depth=20, random_state=42)
        Q.fit(SA, value)
        last_q_function = Q
        memory.fill(200, q_model=last_q_function, epsilon=0.1) 

    return last_q_function

def greedy_action(Q, s, nb_actions, epsilon=0):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(nb_actions)
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)

'''
if __name__ == "__main__":
    print('Starting set-up')
    print('Filling up the memory')
    memory_size = 10000
    with open('memory1_10000', 'rb') as file:
        memory = pickle.load(file)
    
    nb_iter = 250

    print('Training the model')
    #with open('q_function_best.sav', 'rb') as file:
    #    q_function = load(file)
    q_function = rf_fqi(memory, nb_iter, N_ACTIONS, GAMMA)#, q_function=q_function)

    # # iter, # estimators, max_depth
    filename = 'q_function_continued_250_150_075.sav'
    dump(q_function, filename, compress=9)
'''
