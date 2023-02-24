
from collections import namedtuple
from copy import deepcopy
import gym

import numpy as np

from MCTS.MCTS import MCTS, Node # MCTS is the package name. MCTS is the module name, MCTS is the class name....damn I fucked up 🤣


CART_X_BINS =np.flip(np.geomspace(0.001,0.15, 100)*-1) + np.geomspace(0.001,0.15, 100)
CART_V_BINS = np.flip(np.geomspace(0.001,0.15, 100)*-1) + np.geomspace(0.001,0.15, 100)
POLE_THETA_BINS =np.flip(np.geomspace(0.001,0.075, 100)*-1) + np.geomspace(0.001,0.075, 100)
POLE_V_BINS = np.flip(np.geomspace(0.001,0.1, 100)*-1) + np.geomspace(0.001,0.1, 100)

TOTAL_NUM_STATES = len(CART_X_BINS) * len(CART_V_BINS) * len(POLE_THETA_BINS) * len(POLE_V_BINS)

CART_X_HISTO = np.zeros(len(CART_X_BINS))
CART_V_HISTO = np.zeros(len(CART_V_BINS))
POLE_THETA_HISTO = np.zeros(len(POLE_THETA_BINS))
POLE_V_HISTO = np.zeros(len(POLE_V_BINS))

def discretize(observation):
    cart_x, cart_v, pole_theta, pole_v = observation
    cart_x = np.digitize(cart_x, CART_X_BINS)
    cart_v = np.digitize(cart_v, CART_V_BINS)
    pole_theta = np.digitize(pole_theta, POLE_THETA_BINS)
    pole_v = np.digitize(pole_v, POLE_V_BINS)
    return cart_x, cart_v, pole_theta, pole_v

_CartPole = namedtuple("_CartPole", ["cart_x", "cart_v", "pole_theta", "pole_v", "done", "env"])

class CartPole(_CartPole, Node):
    '''Node representation of a cart pole environment.'''

    def IS_TWO_PLAYER():
        return False
    
    def get_children(cart_pole):
        if cart_pole.done:
            return set()
        else:
            return {cart_pole.make_move(0), cart_pole.make_move(1)}
        
    def make_move(cart_pole, index):
        if cart_pole.env.render_mode=="human":
            env_spawn = cart_pole.env
        else:
            env_spawn = deepcopy(cart_pole.env)
        obs, reward, done, truncated, info = env_spawn.step(index)
        cart_x, cart_v, pole_theta, pole_v = discretize(obs)
        return CartPole(cart_x, cart_v, pole_theta, pole_v, done, env_spawn)
    
    def get_random_child(cart_pole):
        return cart_pole.make_move(cart_pole.env.action_space.sample())
    
    def is_terminal(cart_pole):
        return cart_pole.done
    
    def render(cart_pole):
        cart_pole.env.render()

def play_cartpole():
    '''Play cartpole using MCTS.
    Returns:
        trial_rewards: list of rewards for each trial
        tree: MCTS tree
    '''
    trial_rewards=[]
    tree = MCTS(exploration_weight=1)
    human_env = gym.make("CartPole-v1", render_mode="human")
    agent_env = gym.make("CartPole-v1")
    # training
    for trials in range(5000):
        print(f"{trials=}")
        for _ in range(10):
            obs, _= agent_env.reset()
            obs = discretize(obs)
            # fill histograms to see what states are mostly occupied
            CART_X_HISTO[obs[0]] += 1
            CART_V_HISTO[obs[1]] += 1
            POLE_THETA_HISTO[obs[2]] += 1
            POLE_V_HISTO[obs[3]] += 1
            cartpole = CartPole(*obs, False, agent_env)
            tree.do_rollout(cartpole)

        
       
        if trials%50==0:
            ave_reward = []
            for _ in range(10):
                print(f"trial {trials}")
                obs, _ = human_env.reset(seed=42)
                obs = discretize(obs)
                cartpole = CartPole(*obs, False, human_env)
                time_step=0
                while not cartpole.done:
                    old_cartpole=cartpole
                    cartpole = tree.choose(cartpole)
                    cartpole.render()
                    time_step+=1
                print(f"reward: {time_step}")
                print(f"reward, number of visits: {tree.total_rewards[old_cartpole]}, {tree.number_of_visits[old_cartpole]}")
                ave_reward.append(time_step)
            trial_rewards.append(np.mean(ave_reward))
    
    return trial_rewards, tree

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Playing CartPole using MCTS")
    print("Discretized state space size:", TOTAL_NUM_STATES)
    print("Bining cart x:", CART_X_BINS)
    print("Bining cart v:", CART_V_BINS)
    print("Bining pole theta:", POLE_THETA_BINS)
    print("Bining pole v:", POLE_V_BINS)

    rewards, tree = play_cartpole()
    plt.plot(np.array(range(len(rewards)))*50, rewards)

    # plot the histograms
    plt.figure()
    plt.plot(CART_X_BINS, CART_X_HISTO)
    plt.title("Cart X")
    plt.figure()
    plt.plot(CART_V_BINS, CART_V_HISTO)
    plt.title("Cart V")
    plt.figure()
    plt.plot(POLE_THETA_BINS, POLE_THETA_HISTO)
    plt.title("Pole Theta")
    plt.figure()
    plt.plot(POLE_V_BINS, POLE_V_HISTO)
    plt.title("Pole V")
    plt.show()
